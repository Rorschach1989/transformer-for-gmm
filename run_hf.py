import os
os.environ["WANDB_DISABLED"] = "true"
from argparse import ArgumentParser

from transformers import TrainingArguments

from tgmm.dataset import GaussianMixtureDataset, StaticGaussianMixtureDataset
from tgmm.models.tgmm import HFMultiTaskTGMMModel
from tgmm.task import (
    IsotropicGaussianMixtureTask,
    MultiTaskGaussianMixtureTask,
    concat_task_sample_hf,
)
from tgmm.train_hf import TGMMTrainingArguments, TGMMHFTrainer


training_args = TrainingArguments(
    output_dir="./output",
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
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_total_limit=1,
    save_only_model=True,
    save_steps=1000,
    gradient_accumulation_steps=2,
)


def main(args):
    task_list = [
        IsotropicGaussianMixtureTask(n_components=n, dim=8)
        for n in range(2, 5)
    ]
    task = MultiTaskGaussianMixtureTask(task_list)
    train_dataset = GaussianMixtureDataset(task=task, batch_size=16, n_sample=32)
    eval_dataset = {
        t.n_components: StaticGaussianMixtureDataset(
            dataset_size=128,
            task=t,
            n_sample=32
        )
        for t in task_list
    }
    minor_options = TGMMTrainingArguments()
    model = HFMultiTaskTGMMModel(
        task=task,
        model_type="qwen3-0.6B",
        pretrained_ckpt_path=args.pretrained_ckpt_path,
    )
    trainer = TGMMHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tgmm_training_args=minor_options,
        data_collator=concat_task_sample_hf,
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_ckpt_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
