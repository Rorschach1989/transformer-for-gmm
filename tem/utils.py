import torch


def _get_default_device():
    """cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


default_device = _get_default_device()