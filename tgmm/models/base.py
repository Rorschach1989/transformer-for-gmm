import multiprocessing

import torch
from joblib import Parallel, delayed, parallel_config


class _BatchFitMixin(object):
    r"""For wrapping some common utilities of batch fitting"""

    n_jobs = multiprocessing.cpu_count() // 2

    def fit(self, X, *args, **kwargs):
        raise NotImplementedError

    def fit_batch(self, batch_X: torch.Tensor, *args, **kwargs):
        r"""Batch fitting, only necessary for algorithms that
        operates in a task-wise fashion.
        **Notes**: we use joblib to parallelize the fitting process
        which means cpu-only"""
        batch_size = batch_X.size(0)
        # result_pack = []
        # for i in range(batch_size):
        #     result_pack.append(self.fit(batch_X[i], *args, **kwargs))
        with parallel_config("loky"):
            result_pack = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit)(batch_X[i], *args, **kwargs) for i in range(batch_size)
            )
        outputs = []
        for t in zip(*result_pack):
            outputs.append(torch.stack(t, dim=0))
        return outputs
