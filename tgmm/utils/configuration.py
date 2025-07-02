import os
import json
import time
from collections import OrderedDict
from itertools import product

import numpy as np
from omegaconf import OmegaConf


def _get_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def setup_cfg(**kwargs):
    r"""Setup an experiment configuration"""
    cfg = OmegaConf.create()
    cfg.task = {}
    cfg.task.type = kwargs.get("task_type", "MultiTaskIsotropicGaussianMixture")
    if cfg.task.type == "IsotropicGaussianMixture":
        cfg.task.n_components = kwargs.get("n_components", 3)
        cfg.task.dim = kwargs.get("mixture_dim", 8)
    elif cfg.task.type in (
        "MultiTaskIsotropicGaussianMixture",
        "MultiTaskAnisotropicGaussianMixture",
    ):
        n_components_max = kwargs.get("n_components_max", 3)
        n_components_min = kwargs.get("n_components_min", 2)
        if n_components_max < n_components_min:
            n_components_max = n_components_min
        cfg.task.n_components = list(range(n_components_min, n_components_max + 1))
        cfg.task.dim = kwargs.get("mixture_dim", 8)
    elif cfg.task.type == "PhaseTransitionGaussianMixture":
        cfg.task.n_components = 2
        cfg.task.a_s = kwargs.get("a_s", np.linspace(1.1, 11, 20).tolist())
        cfg.task.b = kwargs.get("b", 1.0)
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
    cfg.train.weight_decay = kwargs.get("weight_decay", 0.0)
    cfg.train.num_train_steps = kwargs.get("num_train_steps", 10001)
    cfg.eval = {}
    eval_n_sample = kwargs.get("eval_n_sample", "128").split(",")
    cfg.eval.n_sample = [int(n) for n in eval_n_sample]
    cfg.eval.batch_size = kwargs.get("eval_batch_size", 128)
    cfg.eval.ood_perturbation_scale = kwargs.get("ood_perturbation_scale", 0.0)
    # The following options are currently only used for comparisons between
    # TGMM and InstructTGMM
    cfg.eval.strategy = kwargs.get("eval_strategy", "dynamic")
    cfg.eval.static_dataset_path = kwargs.get("static_dataset_path", None)
    return cfg


def gen_name_from_cfg(cfg):
    if cfg.task.type in {
        "MultiTaskIsotropicGaussianMixture",
        "MultiTaskAnisotropicGaussianMixture",
        "IsotropicGaussianMixture",
    }:
        out_fields = [
            cfg.task.type,
            cfg.task.n_components,
            cfg.task.dim,
            (
                cfg.model.n_embd
                if cfg.model.model_type == "transformer"
                else cfg.model.hidden_size
            ),
            (
                cfg.model.n_layer
                if cfg.model.model_type == "transformer"
                else cfg.model.num_hidden_layers
            ),
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
