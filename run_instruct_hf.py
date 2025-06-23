import os
os.environ["WANDB_DISABLED"] = "true"
from functools import partial

from transformers import TrainingArguments, AutoProcessor, HfArgumentParser

from tgmm.dataset import GaussianMixtureDataset, StaticGaussianMixtureDataset
from tgmm.models.tgmm import MultiTaskInstructTGMMModel
from tgmm.task import (
    IsotropicGaussianMixtureTask,
    MultiTaskGaussianMixtureTask,
    concat_task_sample_instruct,
)
from tgmm.train_hf import TGMMTrainingArguments, TGMMHFTrainer
from tgmm.utils.logger import logger


local_rank = None


def rank0_log(*args):
    if local_rank == 0:
        logger.info(*args)


def main():
    global local_rank

    parser = HfArgumentParser(
        (TrainingArguments, TGMMTrainingArguments),
    )
    training_args, tgmm_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ.get("RANK", 0))

    rank0_log("Start constructing Tasks...")

    task_list = [
        IsotropicGaussianMixtureTask(
            n_components=n,
            dim=tgmm_args.tgmm_task_dim
        )
        for n in tgmm_args.tgmm_components
    ]
    task = MultiTaskGaussianMixtureTask(task_list)
    train_dataset = GaussianMixtureDataset(
        task=task,
        batch_size=tgmm_args.tgmm_batch_size,
        n_sample=tgmm_args.tgmm_n_sample,
    )
    eval_dataset = {
        t.n_components: StaticGaussianMixtureDataset(
            dataset_size=tgmm_args.tgmm_eval_datasize,
            task=t,
            n_sample=tgmm_args.tgmm_n_sample,
        )
        for t in task_list
    }
    model = MultiTaskInstructTGMMModel(
        task=task,
        pretrained_ckpt_path=tgmm_args.tgmm_backbone_ckpt_path,
    )

    rank0_log("Freezing backbone and start stage1 training...")

    tokenizer = AutoProcessor.from_pretrained(tgmm_args.tgmm_backbone_ckpt_path)

    trainer = TGMMHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tgmm_training_args=tgmm_args,
        data_collator=partial(
            concat_task_sample_instruct,
            tokenizer=tokenizer,
        ),
    )
    trainer.train()

    rank0_log("Training finished!")


if __name__ == "__main__":
    main()
