import math
from typing import Union
from dataclasses import dataclass

import torch
import scipy.optimize as sco

from .utils import _cos, _l2, logger, log_exception_with_traceback
from .task import GaussianMixtureTask, GaussianMixtureSample


def _preprocess_scale(scale, batch_size, d, n_components):
    # Pre-process scale
    if scale is None:
        scale = 1.0
    if isinstance(scale, float):
        covariances = (
            torch.stack([torch.eye(d) for _ in range(batch_size)], dim=0) * scale
        )  # [b, d, d]
    elif scale.ndim == 1:
        covariances = torch.stack(
            [torch.eye(d) for _ in range(batch_size)], dim=0
        ) * scale.view(
            -1, 1, 1
        )  # [b, d, d]
    elif scale.ndim == 3:
        # This corresponds to the anisotropic case with [batch_size, n_components, dim]
        # TODO: there could be the case that the semantics were [batch_size, dim, dim]
        covariances = torch.einsum("bkd, de -> bkde", scale, torch.eye(d))
    elif scale.ndim == 4:
        covariances = scale
    else:
        raise ValueError
    if covariances.ndim == 3:
        covariances = torch.stack(
            [covariances for _ in range(n_components)], dim=1
        )  # [b, k, d, d]
    return covariances


def _compute_gmm_ll(X, mu, alpha, scale: Union[float, torch.Tensor] = None):
    r"""Compute gaussian log likelihood in a batched fashion,
    calculations are performed in a general style, supports (potentially)
    full-covariance matrices.

    Args:
        X (torch.Tensor): batched input of shape [batch_size, n_sample, dim]
        mu (torch.Tensor): means of shape [batch_size, n_components, dim]
        alpha (torch.Tensor): weights of shape [batch_size, n_components]
        scale (Union[float, torch.Tensor]): scale, could be of
            - float: Indicates isotropic Gaussian that is identical across tasks
            - torch.Tensor: An array of shape [batch_size], indicating
                isotropic for each task.
            - torch.Tensor: An array of shape [batch_size, n_components, dim],
                indicating an anisotropic Gaussian for each task.
            - torch.Tensor: A matrix of shape [batch_size, n_components, dim],
                indicating an anisotropic Gaussian for each task.
            - torch.Tensor: A matrix of shape [batch_size, n_components, dim, dim],
                indicating a specified covariance matrix for each task.

    Returns:
        likelihood tensor of shape [batch_size],
        estimated assignment tensor of shape [batch_size, n_sample]
    """
    batch_size, n_components, d = mu.size()
    n_sample = X.size(1)
    covariances = _preprocess_scale(scale, batch_size, d, n_components)
    if scale is None:
        inv_covariances = covariances
        log_det = (
            torch.log(2 * torch.pi * torch.ones(batch_size, n_components)) * d
        )  # [b, k]
    elif isinstance(scale, torch.Tensor) and scale.ndim == 3:
        inv_covariances = torch.einsum(
            "bkd, de -> bkde", 1 / (scale + 1e-15), torch.eye(d)
        )
        log_det = torch.log(torch.tensor(2 * torch.pi)) + scale.log().sum(dim=-1)
    else:
        inv_covariances = torch.inverse(covariances)  # [b, k, d, d]
        log_det = torch.logdet(2 * torch.pi * covariances)  # [b, k]
    diff = (X.unsqueeze(2) - mu.unsqueeze(1)).permute((0, 2, 1, 3))  # [b, k, n, d]
    exponent = -0.5 * ((diff @ inv_covariances) * diff).sum(dim=-1)  # [b, k, n]
    log_prob_density = exponent - 0.5 * log_det.unsqueeze(-1)  # [b, k, n]
    log_responsibilities = log_prob_density + torch.log(alpha + 1e-15).unsqueeze(-1)
    batch_ll = (
        torch.logsumexp(log_responsibilities, dim=-1).sum(dim=-1) / n_sample
    )  # [b]
    return batch_ll, torch.argmax(log_responsibilities, dim=1)


@dataclass
class GMMEvaluationResult:
    l2_error_means: Union[torch.Tensor, float]
    l2_error_weights: Union[torch.Tensor, float]
    l2_error_scale: Union[torch.Tensor, float]
    log_likelihood: Union[torch.Tensor, float]
    cluster_acc: Union[torch.Tensor, float]

    def summary(self):
        out_str = ""
        for k, v in self.__dict__.items():
            _out = f"{k}: "
            if isinstance(v, torch.Tensor):
                mean = v.mean().item()
                std = v.std().item()
                _out += f"{mean:.4f} ({std:.4f})"
            else:
                _out += f"{v:.4f}"
            out_str += f"{_out}\n"
        return out_str

    def _maybe_filter_abnormal_likelihoods(self, threshold: float = -100.0):
        all_tensor_check = all(
            isinstance(item, torch.Tensor) for item in self.__dict__.values()
        )
        if all_tensor_check:
            valid_indices = self.log_likelihood > threshold
            self.log_likelihood = self.log_likelihood[valid_indices]
            self.l2_error_means = self.l2_error_means[valid_indices]
            self.l2_error_weights = self.l2_error_weights[valid_indices]
            self.cluster_acc = self.cluster_acc[valid_indices]

    def summary_for_wandb(self):
        out_dict = {}
        self._maybe_filter_abnormal_likelihoods()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                mean = v.mean().item()
                std = v.std().item()
                out_dict[f"{k}_mean"] = mean
                out_dict[f"{k}_std"] = std
            else:
                out_dict[f"{k}"] = v
        return out_dict


class GMMEvaluator(object):
    r"""For evaluating GMM estimations"""

    def __init__(
        self,
        task: GaussianMixtureTask,
        ground_truth: GaussianMixtureSample,
        distance="l2",
    ):
        self.task = task
        self.ground_truth = ground_truth
        if distance == "cos":
            self.distance_fn = _cos
        elif distance == "l2":
            self.distance_fn = _l2
        else:
            raise NotImplementedError

    def _align(self, mu_est: torch.Tensor, alpha_est: torch.Tensor):
        cost_matrix = (
            self.distance_fn(self.ground_truth.gaussian_means, mu_est).cpu().numpy()
        )
        batch_size = mu_est.size(0)
        for i in range(batch_size):
            _, perm = sco.linear_sum_assignment(cost_matrix[i])
            mu_est[i] = mu_est[i][perm, :]  # in-place operation
            alpha_est[i] = alpha_est[i][perm]

    def __call__(
        self,
        mu_est: torch.Tensor,
        alpha_est: torch.Tensor,
        scale_est: torch.Tensor = None,
        in_sample_eval: bool = False,
        **kwargs,
    ):
        try:
            return self._call(mu_est, alpha_est, scale_est, in_sample_eval, **kwargs)
        except Exception as e:
            log_exception_with_traceback(logger)
            return GMMEvaluationResult(
                l2_error_means=math.nan,
                l2_error_weights=math.nan,
                l2_error_scale=math.nan,
                log_likelihood=math.nan,
                cluster_acc=math.nan,
            )

    def _call(
        self,
        mu_est: torch.Tensor,
        alpha_est: torch.Tensor,
        scale_est: torch.Tensor = None,
        in_sample_eval: bool = False,
        **kwargs,
    ):
        # Do the alignment
        self._align(mu_est, alpha_est)
        # l2 error of estimated means and weights
        l2_error_means = (
            (self.ground_truth.gaussian_means - mu_est).square().mean(dim=[1, 2])
        )
        l2_error_weights = (
            (self.ground_truth.mixture_probs - alpha_est).square().mean(dim=-1)
        )
        l2_error_scale = (
            ((self.ground_truth.scale - scale_est).square().mean(dim=-1))
            if scale_est is not None
            else torch.zeros_like(l2_error_means)
        )
        if in_sample_eval:
            X = self.ground_truth.sample
            true_assignments = self.ground_truth.assignment
        else:
            b, n, d = self.ground_truth.sample.size()
            n_sample = kwargs.pop("n_sample", n)
            batch_size = kwargs.pop("batch_size", b)
            new_sample = self.task.resample_from(
                task_sample=self.ground_truth,
                n_sample=n_sample,
                batch_size=batch_size,
            ).to("cpu")
            X = new_sample.sample
            true_assignments = new_sample.assignment
        log_likelihood, cluster_assignments = _compute_gmm_ll(
            X=X,
            mu=mu_est,
            alpha=alpha_est,
            scale=scale_est,
        )
        cluster_acc = (cluster_assignments == true_assignments).float().mean(dim=-1)
        return GMMEvaluationResult(
            l2_error_means=l2_error_means,
            l2_error_weights=l2_error_weights,
            l2_error_scale=l2_error_scale,
            log_likelihood=log_likelihood,
            cluster_acc=cluster_acc,
        )
