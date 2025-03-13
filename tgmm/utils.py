import os
import json
import time
import random
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Union, Callable

import torch
import numpy as np
from omegaconf import OmegaConf
from torchmetrics.aggregation import RunningMean


@dataclass
class WandbProfile(object):
    api_key: str = None
    entity: str = None
    project: str = None


def _get_root():
    return os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def get_wandb_api_key():
    root = _get_root()
    data_path = os.path.join(root, "data", "wandb.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            profile_dict = json.load(f)
            return WandbProfile(**profile_dict)
    else:
        return WandbProfile()


wandb_profile = get_wandb_api_key()


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
    return -inner_product / denom


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
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_device(device_id: int = None):
    if (device_id is None) or (not torch.cuda.is_available()):
        return _get_default_device()
    else:
        return torch.device(f"cuda:{device_id}")


def get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


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
    row_indices = (
        torch.arange(max_len, device=sequence_lengths.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )  # (batch_size, max_len)
    col_indices = sequence_lengths.unsqueeze(1)  # (batch_size, 1)

    mask = row_indices < col_indices  # Broadcasting comparison

    return mask.to(dtype=dtype)


def setup_cfg(**kwargs):
    r"""Setup an experiment configuration"""
    cfg = OmegaConf.create()
    cfg.task = {}
    cfg.task.type = kwargs.get("task_type", "MultiTaskIsotropicGaussianMixture")
    if cfg.task.type == "IsotropicGaussianMixture":
        cfg.task.n_components = kwargs.get("n_components", 3)
        cfg.task.dim = kwargs.get("mixture_dim", 8)
    elif cfg.task.type == "MultiTaskIsotropicGaussianMixture":
        n_components_max = kwargs.get("n_components_max", 3)
        n_components_min = kwargs.get("n_components_min", 2)
        if n_components_max < n_components_min:
            n_components_max = n_components_min
        cfg.task.n_components = list(range(n_components_min, n_components_max + 1))
        cfg.task.dim = kwargs.get("mixture_dim", 8)
    elif cfg.task.type == "PhaseTransitionGaussianMixture":
        cfg.task.n_components = 2
        cfg.task.a_s = kwargs.get(
            "a_s",
            np.linspace(1.1, 11, 20).tolist()
        )
        cfg.task.b = kwargs.get("b", 1.)
    else:
        raise NotImplementedError
    cfg.model = {}
    cfg.model.model_type = kwargs.get("model_type", "transformer")
    # Transformer arguments
    if cfg.model.model_type == "transformer":
        cfg.model.n_positions = kwargs.get("n_positions", 4096)
        cfg.model.n_embd = kwargs.get("n_embd", 128)
        cfg.model.n_layer = kwargs.get("n_layer", 12)
        cfg.model.n_head = kwargs.get("n_head", 4)
    # Mamba2 arguments, note that naming conventions are indeed different
    else:
        cfg.model.hidden_size = kwargs.get("hidden_size", 128)
        cfg.model.num_heads = kwargs.get("num_heads", 8)
        cfg.model.head_dim = kwargs.get("head_dim", 64)
        cfg.model.state_size = kwargs.get("state_size", 16)
        cfg.model.n_groups = kwargs.get("n_groups", 2)
        cfg.model.expand = kwargs.get("expand", 4)
        # Except for num_hidden_layers, the rest params shall be adjusted in a
        # non-trivial way, keep this configuration for the time being
        cfg.model.num_hidden_layers = kwargs.get("n_layer", 12)
    cfg.train = {}
    cfg.train.verbose = kwargs.get("verbose", False)
    cfg.train.seed = kwargs.get("seed", 42)
    cfg.train.n_sample = kwargs.get("train_n_sample", 64)
    cfg.train.batch_size = kwargs.get("train_batch_size", 64)
    cfg.train.eval_every = kwargs.get("eval_every", 1000)
    # TODO: maybe set some advanced training schedules
    cfg.train.learning_rate = kwargs.get("learning_rate", 1e-3)
    cfg.train.weight_decay = kwargs.get("weight_decay", 0.)
    cfg.train.num_train_steps = kwargs.get("num_train_steps", 10001)
    cfg.eval = {}
    eval_n_sample = kwargs.get("eval_n_sample", "128").split(",")
    cfg.eval.n_sample = [int(n) for n in eval_n_sample]
    cfg.eval.batch_size = kwargs.get("eval_batch_size", 128)
    cfg.eval.ood_perturbation_scale = kwargs.get("ood_perturbation_scale", 0.)
    return cfg


def gen_name_from_cfg(cfg):
    if cfg.task.type in {
        "MultiTaskIsotropicGaussianMixture",
        "IsotropicGaussianMixture"
    }:
        out_fields = [
            cfg.task.type,
            cfg.task.n_components,
            cfg.task.dim,
            cfg.model.n_embd if cfg.model.model_type == "transformer" else cfg.model.hidden_size,
            cfg.model.n_layer if cfg.model.model_type == "transformer" else cfg.model.num_hidden_layers,
            cfg.train.batch_size,
            cfg.train.n_sample,
            cfg.eval.n_sample,
        ]
        if cfg.eval.ood_perturbation_scale > 0:
            out_fields.extend(f"{cfg.eval.ood_perturbation_scale:.1f}")
    else:
        a_conf = [min(cfg.task.a_s), max(cfg.task.a_s), len(cfg.task.a_s)]
        b_conf = f"{cfg.task.b:.4f}"
        out_fields = [
            cfg.task.type,
            cfg.task.n_components,
            a_conf,
            b_conf,
            cfg.model.n_embd,
            cfg.model.n_layer,
            cfg.train.batch_size,
            cfg.train.n_sample,
            cfg.eval.n_sample,
        ]
    return "-".join(map(str, out_fields))


class HyperParamManager(object):

    @staticmethod
    def _default_root(prefix="run"):
        log_dir = os.path.join(_get_root(), "logs")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return os.path.join(log_dir, f"{prefix}_{int(time.time())}")

    def __init__(self, root_dir=None, cfg_setter=setup_cfg):
        if root_dir is None:
            root_dir = HyperParamManager._default_root()
        self._param_store = OrderedDict()
        self._root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        self.cfg_setter = cfg_setter

    def register_field(self, key, value):
        if value is None:
            return
        if key not in self._param_store:
            self._param_store[key] = []
        if isinstance(value, list):
            self._param_store[key].extend(value)
        else:
            self._param_store[key].append(value)

    def __getitem__(self, item):
        return self._param_store[item]

    def dump(self, cfg, file_name=None):
        if file_name is None:
            file_name = f"{gen_name_from_cfg(cfg)}.yaml"
        file_path = os.path.join(self._root_dir, file_name)
        OmegaConf.save(cfg, file_path)

    def result_exists(self, cfg, file_name=None):
        if file_name is None:
            file_name = f"{gen_name_from_cfg(cfg)}.results.json"
        file_path = os.path.join(self._root_dir, file_name)
        return os.path.exists(file_path)

    def save_results(self, cfg, results, file_name=None):
        if file_name is None:
            file_name = f"{gen_name_from_cfg(cfg)}.results.json"
        file_path = os.path.join(self._root_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

    def iter_configs(self):
        for cfg_values in product(*self._param_store.values()):
            yield self.cfg_setter(**dict(zip(self._param_store.keys(), cfg_values)))

    def get_description_string(self, fields):
        descriptions = []
        for field in fields:
            field_val = "-".join(map(str, self[field]))
            descriptions.append(f"{field}_{field_val}")
        return "+".join(descriptions)

    def clone(self):
        new_manager = HyperParamManager(cfg_setter=self.cfg_setter)
        for key, val in self._param_store.items():
            new_manager.register_field(key, val)
        return new_manager


class StreamingLossMeter(object):
    r"""A streaming meter for training losses"""

    def __init__(self, n_metrics, window_size):
        self.n_metrics = n_metrics
        self.meters = []
        for _ in range(n_metrics):
            self.meters.append(RunningMean(window=window_size))
        self._counter = 0

    def to(self, device, **kwargs):
        for meter in self.meters:
            meter.to(device=device, **kwargs)
        return self

    def update(self, *losses):
        for i, loss in enumerate(losses):
            self.meters[i].update(loss)
        self._counter += 1

    def compute(self):
        return [meter.compute() if self._counter else None for meter in self.meters]


def _create_empty_list_structure(data):
    """
    Recursively creates a new dictionary with the same structure as the input dictionary,
    but with all values replaced by empty lists.

    Args:
        data: The input dictionary.

    Returns:
        A new dictionary with the same structure, but with empty lists as values.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = _create_empty_list_structure(value)
        return new_dict
    else:
        return []  # Replace non-dictionary values with empty lists


def _append_values_recursively(target_dict, source_dict):
    """
    Recursively appends the values from source_dict to the corresponding empty lists in target_dict.

    Args:
        target_dict: The dictionary with empty lists as values (created by create_empty_list_structure).
        source_dict: The dictionary containing the values to append.
    """
    if isinstance(target_dict, dict) and isinstance(source_dict, dict):
        for key, target_value in target_dict.items():
            source_value = source_dict.get(key)
            if source_value is not None:
                _append_values_recursively(target_value, source_value)
    elif isinstance(target_dict, list):
        if isinstance(source_dict, list):
            target_dict.extend(source_dict)
        else:
            target_dict.append(source_dict)


def _tail_mean(t=10):
    def f(seq):
        return np.mean(seq[-t:])
    return f


@dataclass
class ResultSummarizer(object):
    r"""For handling result summarization within a root directory"""

    L2_ERR_MU = "l2_error_means_mean"
    L2_ERR_ALPHA = "l2_error_weights_mean"
    LL = "log_likelihood_mean"
    CLUSTER_ACC = "cluster_acc_mean"
    EM_ITER = "em_iter"

    MIN = "min"
    MAX = "max"
    MEAN = "mean"

    _AGG_CONF = {
        L2_ERR_MU: _tail_mean(),
        L2_ERR_ALPHA: _tail_mean(),
        LL: _tail_mean(),
        CLUSTER_ACC: _tail_mean(),
        EM_ITER: _tail_mean(),
    }

    @staticmethod
    def _agg_seq(
        seq: list,
        mode: Union[str, Callable],
    ):
        seq = np.asarray(seq)
        seq = seq[~np.isnan(seq)]
        if isinstance(mode, str):
            if mode is ResultSummarizer.MIN:
                return f"{np.min(seq):.4f}"
            elif mode is ResultSummarizer.MAX:
                return f"{np.max(seq):.4f}"
            elif mode is ResultSummarizer.MEAN:
                return f"{np.mean(seq):.4f}"
            else:
                raise NotImplementedError
        else:
            return f"{mode(seq):.4f}"

    @staticmethod
    def _infer_mode(key):
        return ResultSummarizer._AGG_CONF.get(
            key.split(".")[-1]
        )

    def _apply_recursively(self, data, mode: str = None):
        """
        Recursively aggregate for each list in the dictionary.

        Args:
            data: The dictionary with lists as values.
            mode: The aggregation mode to use.

        Returns:
            A new dictionary with the same structure, but with aggregated values.
        """
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                agg_val = self._apply_recursively(
                    value,
                    mode=self._infer_mode(key)
                )
                if agg_val is not None:
                    new_dict[key] = agg_val
            return new_dict
        elif isinstance(data, list):
            if data and (mode is not None):  # check if list is not empty
                return self._agg_seq(data, mode)
            else:
                return None  # Or some other default value if you want to handle empty lists differently.
        else:
            return data  # If it is not a list or dict just return the value.

    root: str

    def _summarize(self, records):
        assert len(records) > 0
        summary_dict = _create_empty_list_structure(records[0])
        for record in records:
            _append_values_recursively(summary_dict, record)
        return self._apply_recursively(summary_dict)

    def summarize(self):
        for file_path in os.listdir(self.root):
            if file_path.endswith(".json"):
                file_path = os.path.join(self.root, file_path)
                conf_path = file_path.replace(".results.json", ".yaml")
                summary_path = file_path.replace(".results.json", ".summary.yaml")
                cfg = OmegaConf.load(conf_path)
                with open(file_path, "r") as f:
                    records = json.load(f)
                cfg.result_summary = self._summarize(records)
                OmegaConf.save(cfg, summary_path)
