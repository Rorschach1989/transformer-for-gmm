import sys
from dataclasses import dataclass

from torch.utils.data import Dataset

from .task import Task


@dataclass
class TGMMDataset(Dataset):
    r"""Wrapping TGMM task utilities into a dataset"""

    task: Task
    n_sample: int

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        r"""Ignore idx and directly sample a subsample"""

        return self.task.sample(
            n_sample=self.n_sample,
            batch_size=1,
        )
