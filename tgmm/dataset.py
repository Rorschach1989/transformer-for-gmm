import os
import json
from typing import Union, Dict, Any
from dataclasses import dataclass

from torch.utils.data import Dataset, IterableDataset

from .task import (
    Task,
    IsotropicGaussianMixtureTask,
)
from .utils import seed_everything


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
    task: IsotropicGaussianMixtureTask
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
        self._sample_raw = sample.clone()
        self.__dict__.update(sample.__dict__)

    def load_from(self, sample: Union[str, Dict[str, Any]], device):
        # **Notes**: We do not check task coherence here
        sample_cls = self._sample.__class__
        if isinstance(sample, str):
            assert os.path.isfile(sample) and sample.endswith(".json")
            with open(sample, "r") as fr:
                sample = json.loads(fr)
        external_sample = sample_cls.from_dict(sample)
        self._sample = external_sample.to(device)
        self._sample_raw = external_sample.clone().to(device)

    def save_to(self, path):
        sample_dict = self._sample_raw.to_dict()
        with open(path, "w") as fw:
            json.dump(sample_dict, fw, indent=4)

    def to_dict(self):
        return self._sample_raw.to_dict()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        exc_keys = {
            "dataset_size",
            "task",
            "n_sample",
            "_sample",
            "_sample_raw",
        }
        output = {}
        for key, value in self.__dict__.items():
            if key in exc_keys:
                continue
            output[key] = value[idx, ...].unsqueeze(0) if value is not None else None
        return output

    @property
    def sample(self):
        return self._sample_raw.to("cpu")

    @property
    def n_components(self):
        return self.task.n_components

    def pad(self, n_components):
        self._sample.pad(n_components)
        self.__dict__.update(self._sample.__dict__)


def _parse_dataset_path(path: str):
    path = path.split("/")[-1].replace(".json", "")
    pieces = path.split("+")
    k_conf, rests = pieces[0], pieces[1:]
    ks = [int(k) for k in k_conf.split("=")[1].split("-")]
    conf = {"k": ks}
    for piece in rests:
        key, value = piece.split("=")
        conf[key] = int(value)
    return conf


def check_or_create_static_dataset(dataset_path, seed=7777777):
    r"""Check if the static dataset exists, if not, create it."""
    if os.path.isfile(dataset_path):
        return
    dirname = os.path.dirname(dataset_path)
    os.makedirs(dirname, exist_ok=True)
    conf = _parse_dataset_path(dataset_path)
    ks = conf.get("k", [2, 3, 4])
    d = conf.get("d", 8)
    n_sample = conf.get("n_sample", 32)
    dataset_size = conf.get("dataset_size", 128)
    seed_everything(seed)
    task_list = [IsotropicGaussianMixtureTask(n_components=n, dim=d) for n in ks]
    dataset_dict = {}
    for task in task_list:
        dataset_name = f"{task.n_components}_{n_sample}"
        dataset_dict[dataset_name] = StaticGaussianMixtureDataset(
            n_sample=n_sample, task=task, dataset_size=dataset_size
        ).to_dict()
    with open(dataset_path, "w") as f:
        json.dump(dataset_dict, f, indent=4)
