import torch


class _BatchFitMixin(object):
    r"""For wrapping some common utilities of batch fitting"""

    def fit(self, X, *args, **kwargs):
        raise NotImplementedError

    def fit_batch(self, batch_X: torch.Tensor, *args, **kwargs):
        r"""Batch fitting, only necessary for algorithms that
        operates in a task-wise fashion."""
        batch_size = batch_X.size(0)
        result_pack = []
        for i in range(batch_size):
            result_pack.append(self.fit(batch_X[i], *args, **kwargs))
        outputs = []
        for t in zip(*result_pack):
            outputs.append(torch.stack(t, dim=0))
        return outputs
