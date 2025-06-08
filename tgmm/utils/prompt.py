from typing import List, Dict

import torch


def _get_prompt_v1(**kwargs):
    system_prompt = """
### Instruction
You need to solve a problem of isotropic Gaussian Mixture Model (GMM) with unit variance. You will be provided with the following information:
- The dimension of the problem.
- The number of GMM components required to solve the problem.
- The number of input GMM samples.

You will output the estimates of mixture probabilities and the means of each GMM component.
"""
    user_prompt = """
### Inputs
- dimension: {n_dim}
- number of GMM components: {n_component}
- number of input GMM samples: {n_sample}
""".format(**kwargs)
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]


def translate_to_prompt(
    gaussian_means: torch.Tensor,
    mask_length: torch.Tensor,
    mask_components: torch.Tensor,
    sample: torch.Tensor,
) -> List[List[Dict[str, str]]]:
    batch_size, _, n_dim = gaussian_means.size()  # n_components can vary across samples
    if mask_length is not None:
        n_samples = mask_length.sum(dim=1).tolist()
    else:  # During eval
        n_sample_ = sample.size(1)
        n_samples = [n_sample_ for _ in range(batch_size)]
    n_components = mask_components.sum(dim=1).tolist()
    batch_messages = []
    for n_sample, n_component in zip(n_samples, n_components):
        kwargs = {
            "n_dim": n_dim,
            "n_component": n_component,
            "n_sample": n_sample,
        }
        batch_messages.append(
            _get_prompt_v1(**kwargs)
        )
    return batch_messages
