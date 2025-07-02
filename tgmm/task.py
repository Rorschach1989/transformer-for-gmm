import math
from typing import List, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# For type hint of InstructTGMM tokenization
from transformers import PreTrainedTokenizerFast

from .utils import (
    _cos,
    sequence_length_to_mask,
)
from .utils.prompt import translate_to_prompt


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

    def to_dict(self):
        return {
            key: val.tolist() if isinstance(val, torch.Tensor) else val
            for key, val in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d):
        for key in d:
            if d[key] is not None:
                d[key] = torch.tensor(d[key], dtype=torch.float)
        return cls(**d)

    def clone(self):
        sample_kwargs = {}
        for k, v in self.__dict__.items():
            if v is not None:
                sample_kwargs[k] = v.clone()
        return self.__class__(**sample_kwargs)

    def to(self, device, **kwargs):
        sample_kwargs = {}
        for k, v in self.__dict__.items():
            if v is not None:
                sample_kwargs[k] = v.to(device, **kwargs)
        return self.__class__(**sample_kwargs)

    def pad(self, pad_to_length):
        if self.mask_components is not None:
            return  # Avoid re-padding
        batch_size, length = self.mixture_probs.size()
        if length < pad_to_length:
            diff = pad_to_length - length
            _ones = torch.ones_like(
                self.mixture_probs, device=self.mixture_probs.device
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
                self.mixture_probs, device=self.mixture_probs.device
            )


@dataclass
class AnisotropicGaussianMixtureSample(IsotropicGaussianMixtureSample):
    r"""For holding an anisotropic Gaussian sample"""

    def pad(self, pad_to_length):
        if self.mask_components is not None:
            return  # Avoid re-padding
        batch_size, length = self.mixture_probs.size()
        if length < pad_to_length:
            diff = pad_to_length - length
            _ones = torch.ones_like(
                self.mixture_probs, device=self.mixture_probs.device
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
            self.scale = F.pad(
                self.scale,
                (0, 0, 0, diff, 0, 0),
                mode="constant",
                value=1.0,
            )
            self.mask_components = F.pad(
                _ones,
                (0, diff),
                mode="constant",
                value=0.0,
            )
        else:
            self.mask_components = torch.ones_like(
                self.mixture_probs, device=self.mixture_probs.device
            )


GaussianMixtureSample = Union[
    IsotropicGaussianMixtureSample, AnisotropicGaussianMixtureSample
]


def concat_task_sample(sample_list: List[GaussianMixtureSample]):
    sample_cls = type(sample_list[0])
    pad_to_length = max(item.mixture_probs.size(1) for item in sample_list)
    for item in sample_list:
        item.pad(pad_to_length)
    kwargs = {k: [] for k in sample_list[0].__dict__}
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
    return sample_cls(**kwargs)


def concat_task_sample_hf(
    sample_list: List[Union[GaussianMixtureSample, Dict[str, Any]]],
):
    if sample_list and isinstance(sample_list[0], Dict):
        concat_sample = concat_task_sample(
            [
                # TODO: make this more configurable
                IsotropicGaussianMixtureSample(**sample)
                for sample in sample_list
            ]
        )
    else:
        concat_sample = concat_task_sample(sample_list)
    return concat_sample.__dict__


def concat_task_sample_instruct(
    sample_list: List[Union[GaussianMixtureSample, Dict[str, Any]]],
    tokenizer: PreTrainedTokenizerFast,
):
    concat_sample = concat_task_sample_hf(sample_list)
    messages = translate_to_prompt(
        gaussian_means=concat_sample["gaussian_means"],
        mask_components=concat_sample["mask_components"],
        mask_length=concat_sample["mask_length"],
        sample=concat_sample["sample"],
    )
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    concat_sample["input_ids"] = input_ids
    # Adjust input masks
    mask_length = concat_sample["mask_length"]
    if mask_length is not None:
        instruction_mask = torch.ones_like(
            input_ids,
            device=mask_length.device,
            dtype=mask_length.dtype,
        )
        mask_length = torch.cat([instruction_mask, mask_length], dim=1)
        concat_sample["mask_length"] = mask_length
    return concat_sample


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
    _default_scale: float = 1.0

    # Misc params reserved for rejection sampling
    _amplification_factor: float = 4.0
    _n_retries = 10

    def _sample_mean(self, batch_size):
        # TODO: too much heuristics here, can we be more rigorous?

        def _gen(b):
            # Expand batch size to create some buffer
            _batch_size = int(self._amplification_factor * b)
            gaussian_means = (
                torch.rand(_batch_size, self.dim, self.n_components) - 0.5
            ) * 10
            self_sim = -_cos(
                gaussian_means.permute(0, 2, 1), gaussian_means.permute(0, 2, 1)
            )
            mask = ~torch.eye(self.n_components, dtype=torch.bool)
            mask = mask.unsqueeze(0).expand(_batch_size, -1, -1)
            off_diagonal_entries = self_sim[mask].view(_batch_size, -1)
            (indices,) = torch.where(off_diagonal_entries.max(dim=-1)[0] < 0.8)
            return gaussian_means[indices][:b, :, :]

        means = _gen(b=batch_size)
        if means.size(0) < batch_size:
            n_retries = 0
            while n_retries < self._n_retries:
                _patch = _gen(b=batch_size - means.size(0))
                means = torch.cat((means, _patch), dim=0)
                if means.size(0) == batch_size:
                    break
                n_retries += 1
        # If maximum number of retries exceeded, return as-is with some post-fix
        return means

    def _sample_mixture_probs(self, batch_size):
        return F.normalize(
            torch.sort(torch.rand(batch_size, self.n_components) * 0.6 + 0.2, dim=-1)[
                0
            ],
            p=1,
        )

    def _sample_scale(self, batch_size):
        r"""Sample a batch of isotropic Gaussian scales"""
        # TODO: enable custom samplers
        scale = self.scale or self._default_scale
        return (torch.ones(batch_size) * scale).view(-1, 1, 1)

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
        ) * scale.unsqueeze(1) + gaussian_means.unsqueeze(1)
        sample = torch.einsum("bndk,bnk->bnd", _sample, assignment_one_hot)
        return IsotropicGaussianMixtureSample(
            mixture_probs=mixture_probs,
            assignment=assignment,
            gaussian_means=torch.permute(gaussian_means, (0, 2, 1)),
            sample=sample,
            scale=scale,
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

    def sample(self, n_sample, batch_size, *args, **kwargs):
        gen_mask = kwargs.pop("gen_mask", True)
        mixture_probs = self._sample_mixture_probs(batch_size)
        gaussian_means = self._sample_mean(batch_size)
        scale = self._sample_scale(batch_size)
        # Check if batch_size shall be reduced
        effective_batch_size = min(gaussian_means.size(0), scale.size(0))
        if effective_batch_size < batch_size:
            batch_size = effective_batch_size
            gaussian_means = gaussian_means[:batch_size, ...]
            mixture_probs = mixture_probs[:batch_size, ...]
            scale = scale[:batch_size, ...]
        task_sample = self._sample(
            n_sample,
            batch_size,
            mixture_probs,
            gaussian_means,
            scale,
        )
        if gen_mask:
            mask_length = self._sample_seq_mask(batch_size, n_sample)
            task_sample.mask_length = mask_length
        return task_sample


@dataclass
class AnisotropicGaussianMixtureTask(IsotropicGaussianMixtureTask):
    r"""Task for sampling anisotropic Gaussian mixture"""

    def _sample_scale(self, batch_size):
        _mean = self._sample_mean(batch_size) / 5
        return F.softplus(_mean) * 2

    def sample(self, n_sample, batch_size, *args, **kwargs):
        _sample = super(AnisotropicGaussianMixtureTask, self).sample(
            n_sample,
            batch_size,
            *args,
            **kwargs,
        )
        return AnisotropicGaussianMixtureSample(
            mixture_probs=_sample.mixture_probs,
            assignment=_sample.assignment,
            sample=_sample.sample,
            gaussian_means=_sample.gaussian_means,
            scale=torch.permute(_sample.scale, (0, 2, 1)),
            mask_length=_sample.mask_length,
            mask_components=_sample.mask_components,
        )


@dataclass
class OODIsotropicGaussianMixtureTask(IsotropicGaussianMixtureTask):
    r"""Wrapping IsotropicGaussianMixtureTask with OOD perturbations
    Only use during evaluation."""

    perturbation_scale: float = 1.0

    @classmethod
    def from_id_task(
        cls, id_task: IsotropicGaussianMixtureTask, perturbation_scale: float = 1.0
    ):
        return cls(
            n_components=id_task.n_components,
            dim=id_task.dim,
            scale=id_task.scale,
            perturbation_scale=perturbation_scale,
        )

    # def _sample_mean(self, batch_size):
    #     benign_means = super(OODIsotropicGaussianMixtureTask, self)._sample_mean(
    #         batch_size
    #     )
    #     return (
    #         benign_means
    #         + torch.randn_like(benign_means, device=benign_means.device)
    #         * self.perturbation_scale
    #     )

    def _sample_mean(self, batch_size):
        # TODO: too much heuristics here, can we be more rigorous?

        def _gen(b):
            # Expand batch size to create some buffer
            _batch_size = int(self._amplification_factor * b)
            gaussian_means = (
                torch.rand(_batch_size, self.dim, self.n_components) - 0.5
            ) * 10
            gaussian_means = (
                gaussian_means
                + torch.randn_like(gaussian_means, device=gaussian_means.device)
                * self.perturbation_scale
            )
            self_sim = -_cos(
                gaussian_means.permute(0, 2, 1), gaussian_means.permute(0, 2, 1)
            )
            mask = ~torch.eye(self.n_components, dtype=torch.bool)
            mask = mask.unsqueeze(0).expand(_batch_size, -1, -1)
            off_diagonal_entries = self_sim[mask].view(_batch_size, -1)
            (indices,) = torch.where(off_diagonal_entries.max(dim=-1)[0] < 0.8)
            return gaussian_means[indices][:b, :, :]

        means = _gen(b=batch_size)
        if means.size(0) < batch_size:
            n_retries = 0
            while n_retries < self._n_retries:
                _patch = _gen(b=batch_size - means.size(0))
                means = torch.cat((means, _patch), dim=0)
                if means.size(0) == batch_size:
                    break
                n_retries += 1
        # If maximum number of retries exceeded, return as-is with some post-fix
        return means


@dataclass
class SphericalGaussianMixtureTask(IsotropicGaussianMixtureTask):
    r"""For conducting phase-transition experiments
    The methodology is inspired by the paper
    ``Sharp optimal recovery in the two component Gaussian mixture model``
    https://arxiv.org/abs/1812.08078
    """

    delta: float = 1.0  # Required param indicating l2-distance between means
    a: float = 1.0
    b: float = 1.0

    def _sample_mixture_probs(self, batch_size):
        r"""As recovery guarantees have nothing to do with mixture probs
        Use the optimistic setup"""
        return torch.ones(batch_size, self.n_components) / 2

    def _sample_mean(self, batch_size):
        r"""Samples points uniformly from the surface of an n-dimensional unit sphere."""
        gaussian_means = torch.randn(batch_size, self.dim)
        norm = gaussian_means.norm(dim=1, keepdim=True)
        _mean_normed = gaussian_means / norm
        mean_sampled = (
            torch.stack([_mean_normed, -_mean_normed], dim=-1) * self.delta / 2
        )
        # **Notes**: Transformer may take shortcuts to find that two opposite vector
        # should be learned, break this
        return mean_sampled + (
            torch.randn(batch_size, self.dim) * self.delta
        ).unsqueeze(-1)

    @classmethod
    def abn_config(cls, a, b, n):
        r"""The experimental configuration in the Ndaoud paper"""
        delta = math.sqrt((1 + math.sqrt(a)) * math.log(n))
        d = int(b * n * math.log(n))
        return cls(
            n_components=2,
            dim=d,
            delta=delta,
            a=a,
            b=b,
        )


GaussianMixtureTask = Union[
    IsotropicGaussianMixtureTask,
    AnisotropicGaussianMixtureTask,
    SphericalGaussianMixtureTask,
    OODIsotropicGaussianMixtureTask,
]


GaussianMixtureSample = Union[
    IsotropicGaussianMixtureSample,
    AnisotropicGaussianMixtureSample,
]


class MultiTaskGaussianMixtureTask(Task):
    r"""GMM task that contains a mixture of tasks with
    different components."""

    def __init__(
        self,
        tasks: List[GaussianMixtureTask],
    ):
        dim = tasks[0].dim
        assert all(task.dim == dim for task in tasks)
        self.tasks = tasks
        self.dim = dim
        self.subtask_components = [task.n_components for task in self.tasks]
        self.max_n_components = max(self.subtask_components)

    @classmethod
    def abn_config(cls, a_s, b, n):
        r"""The experimental configuration in the Ndaoud paper,
        with an array of deltas but fix p"""
        tasks = [SphericalGaussianMixtureTask.abn_config(a, b, n) for a in a_s]
        return cls(tasks)

    @property
    def n_subtasks(self):
        return len(self.tasks)

    def sample(self, n_sample, batch_size, *args, **kwargs):
        sample_list = [
            task.sample(n_sample, batch_size, *args, **kwargs) for task in self.tasks
        ]
        return concat_task_sample(sample_list)
