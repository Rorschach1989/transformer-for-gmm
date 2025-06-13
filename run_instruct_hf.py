import os
os.environ["WANDB_DISABLED"] = "true"
from argparse import ArgumentParser
from functools import partial

from transformers import TrainingArguments, AutoProcessor

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


def main(args):
    global local_rank
    local_rank = int(os.environ.get("RANK", 0))

    rank0_log("Start constructing Tasks...")

    task_list = [
        IsotropicGaussianMixtureTask(n_components=n, dim=8)
        for n in range(2, 5)
    ]
    task = MultiTaskGaussianMixtureTask(task_list)
    train_dataset = GaussianMixtureDataset(
        task=task,
        batch_size=16,
        n_sample=32
    )
    eval_dataset = {
        t.n_components: StaticGaussianMixtureDataset(
            dataset_size=128,
            task=t,
            n_sample=32
        )
        for t in task_list
    }
    minor_options = TGMMTrainingArguments()
    model = MultiTaskInstructTGMMModel(
        task=task,
        pretrained_ckpt_path=args.pretrained_ckpt_path,
    )

    rank0_log("Freezing backbone and start stage1 training...")

    tokenizer = AutoProcessor.from_pretrained(args.pretrained_ckpt_path)

    training_args = TrainingArguments(
        output_dir="./output_stage1",
        label_names=[
            "mixture_probs",
            "assignment",
            "gaussian_means",
            "scale",
        ],
        max_steps=10000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_total_limit=1,
        save_only_model=True,
        save_steps=1000,
        gradient_accumulation_steps=2,
    )

    trainer = TGMMHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tgmm_training_args=minor_options,
        data_collator=partial(
            concat_task_sample_instruct,
            tokenizer=tokenizer,
        ),
    )
    trainer.train()

    rank0_log("Training finished!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_ckpt_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
