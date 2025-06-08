from dataclasses import dataclass

from torch.utils.data import Dataset, IterableDataset

from .task import Task, MultiTaskGaussianMixtureTask


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


@dataclass
class StaticGaussianMixtureDataset(Dataset):
    r"""A Gaussian mixture dataset that contains a static sample
    **Notes**: used for evaluation only"""

    dataset_size: int
    task: MultiTaskGaussianMixtureTask
    n_sample: int

    def __post_init__(self):
        super(Dataset, self).__init__()
        # Generate static sample
        sample = self.task.sample(
            n_sample=self.n_sample,
            batch_size=self.dataset_size,
            gen_mask=False,
        )
        self._sample = sample
        self.__dict__.update(sample.__dict__)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        exc_keys = {
            "dataset_size",
            "task",
            "n_sample",
            "_sample",
        }
        output = {}
        for key, value in self.__dict__.items():
            if key in exc_keys:
                continue
            output[key] = value[idx, ...].unsqueeze(0)\
                if value is not None else None
        return output

    @property
    def sample(self):
        return self._sample.to("cpu")

    @property
    def n_components(self):
        return self.task.n_components

    def pad(self, n_components):
        self._sample.pad(n_components)
        self.__dict__.update(self._sample.__dict__)
