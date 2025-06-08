import torch
from ..task import IsotropicGaussianMixtureSample


def translate_to_prompt(
    sample: IsotropicGaussianMixtureSample,
):
    batch_size, _, n_dim = sample.gaussian_means.size()  # n_components can vary across samples

