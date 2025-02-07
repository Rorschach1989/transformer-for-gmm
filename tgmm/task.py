from typing import List, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import (
    default_device,
    _cos,
    sequence_length_to_mask,
)


class Task(object):
    r"""Base class for task objects."""

    def sample(self, n_sample, batch_size, *args, **kwargs):
        raise NotImplementedError


@dataclass
class IsotropicGaussianMixtureSample(object):
    r"""For holding an isotropic Gaussian sample"""

    mixture_probs: torch.Tensor
    assignment: torch.Tensor
    gaussian_means: torch.Tensor
    sample: torch.Tensor
    scale: torch.Tensor
    mask_length: torch.Tensor = None
    mask_components: torch.Tensor = None

    def to(self, device):
        kwargs = {}
        for k, v in self.__dict__.items():
            if v is not None:
                kwargs[k] = v.to(device)
        return IsotropicGaussianMixtureSample(**kwargs)

    def pad(self, pad_to_length):
        batch_size, length = self.mixture_probs.size()
        if length < pad_to_length:
            diff = pad_to_length - length
            _ones = torch.ones_like(
                self.mixture_probs,
                device=self.mixture_probs.device
            )
            self.mixture_probs = F.pad(
                self.mixture_probs,
                (0, diff),
                mode="constant",
                value=0.0,
            )
            self.gaussian_means = F.pad(
                self.gaussian_means,
                (0, 0, 0, diff, 0, 0),
                mode="constant",
                value=0.0,
            )
            self.mask_components = F.pad(
                _ones,
                (0, diff),
                mode="constant",
                value=0.0,
            )
        else:
            self.mask_components = torch.ones_like(
                self.mixture_probs,
                device=self.mixture_probs.device
            )


def concat_task_sample(sample_list: List[IsotropicGaussianMixtureSample]):
    pad_to_length = max(item.mixture_probs.size(1) for item in sample_list)
    for item in sample_list:
        item.pad(pad_to_length)
    kwargs = {
        k: [] for k in sample_list[0].__dict__
    }
    for item in sample_list:
        for k, v in item.__dict__.items():
            if v is not None:
                kwargs[k].append(v)
    for k in list(kwargs):
        if kwargs[k]:
            # TODO: Align length masks
            kwargs[k] = torch.cat(kwargs[k], dim=0)
        else:
            kwargs[k] = None
    return IsotropicGaussianMixtureSample(**kwargs)


@dataclass
class IsotropicGaussianMixtureTask(Task):
    r"""Task for sampling IsotropicGaussianMixture.

    **Notes**
    Memos for reasonable sampling of Gaussian mixtures
    - Components shall NOT be two close
    - Mixture probabilities shall NOT be two extreme
    """

    n_components: int
    dim: int
    scale: float = None
    _default_scale: float = 1.

    def _sample_mean(self, batch_size):
        # TODO: too much heuristics here, can we be more rigorous?
        _batch_size = 4 * batch_size  # Expand batch size to create some buffer
        gaussian_means = (torch.rand(_batch_size, self.dim, self.n_components) - 0.5) * 10
        self_sim = -_cos(gaussian_means.permute(0, 2, 1), gaussian_means.permute(0, 2, 1))
        mask = ~torch.eye(self.n_components, dtype=torch.bool)
        mask = mask.unsqueeze(0).expand(_batch_size, -1, -1)
        off_diagonal_entries = self_sim[mask].view(_batch_size, -1)
        indices, = torch.where(off_diagonal_entries.max(dim=-1)[0] < 0.8)
        return gaussian_means[indices][:batch_size, :, :]

    def _sample_mixture_probs(self, batch_size):
        return F.normalize(
            torch.sort(
                torch.rand(batch_size, self.n_components) * 0.6 + 0.2,
                dim=-1
            )[0],
            p=1,
        )

    def _sample_scale(self, batch_size):
        r"""Sample a batch of isotropic Gaussian scales"""
        # TODO: enable custom samplers
        scale = self.scale or self._default_scale
        return torch.ones(batch_size) * scale

    def _sample_seq_mask(self, batch_size, n_sample):
        n_max = n_sample
        n_min = n_sample // 2  # TODO: use more flexible strategies
        seq_lens = torch.randint(n_min, n_max + 1, (batch_size,))
        return sequence_length_to_mask(seq_lens, max_len=n_sample)

    def _sample(self, n_sample, batch_size, mixture_probs, gaussian_means, scale):
        assignment = torch.multinomial(
            mixture_probs, n_sample, replacement=True
        )  # [batch_size, n_sample]
        assignment_one_hot = F.one_hot(
            assignment, self.n_components
        ).float()  # [batch_size, n_sample, n_components]
        _sample = torch.randn(
            batch_size, n_sample, self.dim, self.n_components
        ) * scale.view(-1, 1, 1, 1) + gaussian_means.unsqueeze(1)
        sample = torch.einsum("bndk,bnk->bnd", _sample, assignment_one_hot)
        return IsotropicGaussianMixtureSample(
            mixture_probs=mixture_probs.to(default_device),
            assignment=assignment.to(default_device),
            gaussian_means=torch.permute(gaussian_means.to(default_device), (0, 2, 1)),
            sample=sample.to(default_device),
            scale=scale.to(default_device),
        )

    def resample_from(
        self,
        task_sample: IsotropicGaussianMixtureSample,
        n_sample=None,
        batch_size=None,
    ):  # TODO: refine this util
        _batch_size, _n_sample, _ = task_sample.sample.size()
        n_sample = n_sample or _n_sample
        batch_size = batch_size or _batch_size
        return self._sample(
            n_sample=n_sample,
            batch_size=batch_size,
            mixture_probs=task_sample.mixture_probs,
            gaussian_means=task_sample.gaussian_means,
            scale=task_sample.scale,
        )

    def sample(
        self,
        n_sample,
        batch_size,
        *args,
        **kwargs
    ):
        gen_mask = kwargs.pop("gen_mask", True)
        mixture_probs = self._sample_mixture_probs(batch_size)
        gaussian_means = self._sample_mean(batch_size)
        scale = self._sample_scale(batch_size)
        task_sample = self._sample(
            n_sample,
            batch_size,
            mixture_probs,
            gaussian_means,
            scale,
        )
        if gen_mask:
            mask_length = self._sample_seq_mask(
                batch_size, n_sample
            ).to(default_device)
            task_sample.mask_length = mask_length
        return task_sample


class MixedComponentGMMTask(Task):
    r"""Isotropic GMM task that contains a mixture of tasks with
    different components."""

    def __init__(self, tasks: List[IsotropicGaussianMixtureTask]):
        dim = tasks[0].dim
        assert all(task.dim == dim for task in tasks)
        self.tasks = tasks
        self.dim = dim
        self.subtask_components = [task.n_components for task in self.tasks]
        self.max_n_components = max(self.subtask_components)

    @property
    def n_subtasks(self):
        return len(self.tasks)

    def sample(self, n_sample, batch_size, *args, **kwargs):
        sample_list = [
            task.sample(n_sample, batch_size, *args, **kwargs)
            for task in self.tasks
        ]
        return concat_task_sample(sample_list)
