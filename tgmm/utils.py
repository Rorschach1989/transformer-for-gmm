import random

import torch
import numpy as np


def _cos(truth: torch.Tensor, est: torch.Tensor):
    r"""Compute cosine similarity

    Args:
        truth (torch.Tensor): ground truth means of shape [batch_size, n_components, dim]
        est (torch.Tensor): estimated means of shape [batch_size, n_components, dim]

    Returns:
        NEGATIVE similarity matrix of shape [batch_size, n_components, n_components]
    """
    inner_product = torch.bmm(truth, est.permute((0, 2, 1)))
    norm_truth = truth.norm(dim=-1, p=2, keepdim=True)
    norm_est = est.norm(dim=-1, p=2, keepdim=True)
    denom = torch.bmm(norm_truth, norm_est.permute((0, 2, 1))) + 1e-15
    return - inner_product / denom


def _l2(truth: torch.Tensor, est: torch.Tensor):
    r"""Compute L2 distance"""
    return (truth.unsqueeze(1) - est.unsqueeze(2)).square().mean(dim=-1)


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.
    copied from the impl in ``torch_geometric`` library

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_default_device():
    """cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


default_device = _get_default_device()