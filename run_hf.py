import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import TrainingArguments

from tgmm.dataset import GaussianMixtureDataset
from tgmm.models.tgmm import MultiTaskTGMMModel
from tgmm.task import (
    IsotropicGaussianMixtureTask,
    MultiTaskGaussianMixtureTask,
    concat_task_sample_hf,
)
from tgmm.train_hf import TGMMTrainingArguments, TGMMQwenTrainer


training_args = TrainingArguments(
    output_dir="./output",
    max_steps=10000,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=10,
)


def main():
    task_list = [
        IsotropicGaussianMixtureTask(n_components=n, dim=8)
        for n in range(2, 5)
    ]
    task = MultiTaskGaussianMixtureTask(task_list)
    dataset = GaussianMixtureDataset(task=task, batch_size=4, n_sample=32)
    minor_options = TGMMTrainingArguments()
    model = MultiTaskTGMMModel(
        task=task,
        model_type="qwen3-0.6B",
        pretrained_ckpt_path = "/Users/zekkarorschach/Projects/llm/Qwen3-0.6B",
    )
    trainer = TGMMQwenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tgmm_training_args=minor_options,
        data_collator=concat_task_sample_hf,
    )
    trainer.train()


if __name__ == "__main__":
    main()
