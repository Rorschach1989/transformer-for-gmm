from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import default_device


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


@dataclass
class IsotropicGaussianMixtureTask(Task):
    r"""Task for sampling IsotropicGaussianMixture."""

    n_components: int
    dim: int

    def sample(self, n_sample, batch_size, *args, **kwargs):
        mixture_probs = F.normalize(
            torch.sort(torch.rand(batch_size, self.n_components), dim=-1)[0],
            p=1,
        )
        # mixture_probs = F.normalize(
        #     torch.sort(torch.tensor([[1., 1.]]), dim=-1)[0],
        #     p=1,
        # )
        # print(mixture_probs)
        assignment = torch.multinomial(
            mixture_probs, n_sample, replacement=True
        )  # [batch_size, n_sample]
        assignment_one_hot = F.one_hot(
            assignment, self.n_components
        ).float()  # [batch_size, n_sample, n_components]
        gaussian_means = (torch.rand(batch_size, self.dim, self.n_components) - 0.5) * 4
        # gaussian_means = torch.tensor([[[2, 2], [-2, -2]]])
        _sample = torch.randn(
            batch_size, n_sample, self.dim, self.n_components
        ) * 0.2 + gaussian_means.unsqueeze(1)
        # sample = (_sample * assignment_one_hot.unsqueeze(2)).sum(dim=-1)  # [batch_size, n_sample, dim]
        sample = torch.einsum("bndk,bnk->bnd", _sample, assignment_one_hot)
        return IsotropicGaussianMixtureSample(
            mixture_probs=mixture_probs.to(default_device),
            assignment=assignment,
            gaussian_means=torch.permute(gaussian_means.to(default_device), (0, 2, 1)),
            sample=sample.to(default_device),
        )
