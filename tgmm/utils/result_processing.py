import os
import json
from dataclasses import dataclass
from typing import Union, Callable

import numpy as np
from omegaconf import OmegaConf


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
        return ResultSummarizer._AGG_CONF.get(key.split(".")[-1])

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
                agg_val = self._apply_recursively(value, mode=self._infer_mode(key))
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
