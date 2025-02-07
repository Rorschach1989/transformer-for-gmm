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


# Implemented by Gemini 2.0 Flash thinking
def sequence_length_to_mask(sequence_lengths, max_len=None, dtype=torch.bool):
    """
    Converts a tensor of sequence lengths into a mask tensor.

    This function takes a 1D tensor of sequence lengths and generates a 2D mask tensor
    where each row corresponds to a sequence and indicates the valid elements
    within that sequence based on its length.

    Args:
        sequence_lengths (torch.Tensor): A 1D tensor of sequence lengths (dtype=torch.long or torch.int).
        max_len (int, optional):  The maximum sequence length to use for the mask.
                                   If None (default), it will be inferred from the maximum
                                   value in `sequence_lengths`. If provided, the mask will
                                   be created up to this length, padding with False if necessary.
        dtype (torch.dtype, optional): The desired data type of the mask tensor.
                                      Defaults to torch.bool. You can also use torch.uint8
                                      for integer masks (0 and 1).

    Returns:
        torch.Tensor: A 2D mask tensor of shape (batch_size, max_len) and dtype `dtype`.
                      Values are True (or 1) for valid positions and False (or 0) for
                      padding positions.

    Example:
        >>> sequence_lengths = torch.tensor([5, 2, 4])
        >>> mask = sequence_length_to_mask(sequence_lengths)
        >>> print(mask)
        tensor([[ True,  True,  True,  True,  True],
                [ True,  True, False, False, False],
                [ True,  True,  True,  True, False]])

        >>> sequence_lengths = torch.tensor([3, 1, 2])
        >>> mask_int = sequence_length_to_mask(sequence_lengths, dtype=torch.uint8)
        >>> print(mask_int)
        tensor([[1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0]], dtype=torch.uint8)

        >>> sequence_lengths = torch.tensor([3, 1, 2])
        >>> mask_fixed_len = sequence_length_to_mask(sequence_lengths, max_len=7)
        >>> print(mask_fixed_len)
        tensor([[ True,  True,  True, False, False, False, False],
                [ True, False, False, False, False, False, False],
                [ True,  True, False, False, False, False, False]])
    """

    if max_len is None:
        max_len = torch.max(sequence_lengths)  # Find the maximum length dynamically

    batch_size = sequence_lengths.size(0)
    row_indices = torch.arange(max_len, device=sequence_lengths.device).unsqueeze(0).repeat(batch_size, 1) # (batch_size, max_len)
    col_indices = sequence_lengths.unsqueeze(1) # (batch_size, 1)

    mask = row_indices < col_indices # Broadcasting comparison

    return mask.to(dtype=dtype)


default_device = _get_default_device()