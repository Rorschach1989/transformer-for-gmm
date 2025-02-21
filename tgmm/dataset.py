from dataclasses import dataclass

from torch.utils.data import IterableDataset

from .task import Task


@dataclass
class _TaskIterator(object):
    r"""Iterator class that returns samples"""

    batch_size: int
    task: Task
    n_sample: int

    def __iter__(self):
        return self

    def __next__(self):
        return self.task.sample(
            n_sample=self.n_sample,
            batch_size=self.batch_size,
        )


@dataclass
class GaussianMixtureDataset(IterableDataset):
    r"""Wrapping Gaussian mixture task utilities into a dataset"""

    batch_size: int
    task: Task
    n_sample: int

    def __iter__(self):
        return _TaskIterator(self.batch_size, self.task, self.n_sample)
