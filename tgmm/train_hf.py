from typing import Optional, Union, Dict, List
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import Trainer

from .models.tgmm import TGMMOutput
from .dataset import StaticGaussianMixtureDataset
from .evaluation import GMMEvaluator


@dataclass
class TGMMTrainingArguments(object):
    r"""For holding some loss function balancing options"""
    tgmm_backbone_ckpt_path: str = field(  # This one is required
        metadata={"help": "path to a checkpoint of the TGMM backbone"}
    )
    mean_coefficient: float = field(
        default=1.0,
        metadata={"help": "coefficient for the loss function regarding mean component"}
    )
    prob_coefficient: float = field(
        default=1.0,
        metadata={"help": "coefficient for the loss function regarding probability component"}
    )
    scale_coefficient: float = field(
        default=1.0,
        metadata={"help": "coefficient for the loss function regarding scale component"}
    )
    # Use tgmm prefix to avoid confusions
    tgmm_task_dim: int = field(
        default=8,
        metadata={"help": "dimensionality of the GMM problems"}
    )
    tgmm_n_sample: int = field(
        default=32,
        metadata={"help": "number of samples in the GMM problems"}
    )
    tgmm_batch_size: int = field(
        default=4,
        metadata={"help": "batch size for TGMM training"}
    )
    tgmm_components: List[int] = field(
        default=None,
        metadata={"help": "number of components for TGMM training"},
    )
    tgmm_eval_datasize: int = field(
        default=128,
        metadata={"help": "number of eval cases in the GMM problems"}
    )


class TGMMHFTrainer(Trainer):
    r"""Customized Trainer for Huggingface-models-backbone training of TGMM."""

    def __init__(self, *args, tgmm_training_args: TGMMTrainingArguments, **kwargs) -> None:
        super(TGMMHFTrainer, self).__init__(*args, **kwargs)
        self.tgmm_training_args = tgmm_training_args
        self.tgmm_task_evaluators = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        r""""""
        # inputs = IsotropicGaussianMixtureSample(**inputs)
        outputs: TGMMOutput = model(**inputs)
        loss = outputs.mu_loss * self.tgmm_training_args.mean_coefficient + \
            outputs.alpha_loss * self.tgmm_training_args.prob_coefficient
        if outputs.scale_loss is not None:
            loss += outputs.scale_loss * self.tgmm_training_args.scale_coefficient
        return (loss, outputs.to_predictions()) if return_outputs else loss

    def _get_or_create_tgmm_task_evaluator(self, task_name, task, ground_truth):
        if task_name not in self.tgmm_task_evaluators:
            self.tgmm_task_evaluators[task_name] = GMMEvaluator(
                task=task,
                ground_truth=ground_truth,
            )
        return self.tgmm_task_evaluators[task_name]

    def evaluate(
        self,
        eval_dataset: Optional[Dict[int, StaticGaussianMixtureDataset]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        eval_dataset = eval_dataset or self.eval_dataset
        all_metrics_dict = {}
        max_n_components = max(d.n_components for d in eval_dataset.values())
        for task_name, subtask_dataset in eval_dataset.items():
            subtask = subtask_dataset.task
            evaluator = self._get_or_create_tgmm_task_evaluator(
                task_name=task_name,
                task=subtask,
                ground_truth=subtask_dataset.sample,
            )
            subtask_dataset.pad(max_n_components)
            eval_dataloader = self.get_eval_dataloader(subtask_dataset)
            output = self.evaluation_loop(
                eval_dataloader,
                description="TGMMEvaluation",
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            if len(output.predictions) == 2:
                alpha_est, mu_est = output.predictions
                scale_est = None
            elif len(output.predictions) == 3:
                alpha_est, mu_est, scale_est = output.predictions
            else:
                raise ValueError(f"Not supported prediction shape {len(output.predictions)}")
            alpha_est = torch.from_numpy(alpha_est[:, : subtask.n_components])
            mu_est = torch.from_numpy(mu_est[:, : subtask.n_components, :])
            if scale_est is not None:
                scale_est = torch.from_numpy(scale_est[:, : subtask.n_components, :])
            eval_results_tgmm = evaluator(
                mu_est=mu_est.cpu(),
                alpha_est=F.softmax(alpha_est.cpu(), dim=-1),
                scale_est=(
                    scale_est.cpu()
                    if scale_est is not None
                    else None
                ),
                in_sample_eval=True,
            )
            metrics = eval_results_tgmm.summary_for_wandb()
            all_metrics_dict[f"K={task_name}"] = metrics
        self.log(all_metrics_dict)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, all_metrics_dict)
        return all_metrics_dict
