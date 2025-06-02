from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import Trainer

from .models.tgmm import TGMMOutput
from .task import IsotropicGaussianMixtureSample


@dataclass
class TGMMTrainingArguments(object):
    r"""For holding some loss function balancing options"""
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


class TGMMQwenTrainer(Trainer):
    r"""Customized Trainer for Huggingface-models-backbone training of TGMM."""

    def __init__(self, *args, tgmm_training_args: TGMMTrainingArguments, **kwargs) -> None:
        super(TGMMQwenTrainer, self).__init__(*args, **kwargs)
        self.tgmm_training_args = tgmm_training_args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        r""""""
        inputs = IsotropicGaussianMixtureSample(**inputs)
        outputs: TGMMOutput = model(inputs)  # We do not need dict-valued inputs
        loss = outputs.mu_loss * self.tgmm_training_args.mean_coefficient + \
            outputs.alpha_loss * self.tgmm_training_args.prob_coefficient
        if outputs.scale_loss is not None:
            loss += outputs.scale_loss * self.tgmm_training_args.scale_coefficient
        return (loss, outputs) if return_outputs else loss
