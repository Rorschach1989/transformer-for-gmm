from typing import Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from .models.tgmm import TGMMModel, MultiTaskTGMMModel
from .models.em import GaussianMixtureEM
from .models.spectral import GaussianMixtureSpectral
from .logger import logger
from .evaluation import GMMEvaluator
from .task import (
    IsotropicGaussianMixtureTask,
    OODIsotropicGaussianMixtureTask,
    SphericalGaussianMixtureTask,
    MultiTaskIsotropicGaussianMixtureTask,
    concat_task_sample
)
from .dataset import GaussianMixtureDataset
from .utils import seed_everything, wandb_profile, get_device, StreamingLossMeter


def _init_task_and_model(cfg):
    if cfg.task.type == "IsotropicGaussianMixture":
        task = IsotropicGaussianMixtureTask(
            n_components=cfg.task.n_components,
            dim=cfg.task.dim,
        )
        model = TGMMModel(
            task=task,
            n_positions=cfg.model.n_positions,
            n_embd=cfg.model.n_embd,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
        )
    elif cfg.task.type == "MultiTaskIsotropicGaussianMixture":
        task_list = [
            IsotropicGaussianMixtureTask(
                n_components=n,
                dim=cfg.task.dim,
            )
            for n in cfg.task.n_components
        ]
        task = MultiTaskIsotropicGaussianMixtureTask(task_list)
        if cfg.model.model_type == "transformer":
            model_args = {
                "n_positions": cfg.model.n_positions,
                "n_embd": cfg.model.n_embd,
                "n_layer": cfg.model.n_layer,
                "n_head": cfg.model.n_head,
            }
        else:
            # Mamba2 arguments, note that naming conventions are indeed different
            model_args = {
                "hidden_size": cfg.model.hidden_size,
                "num_heads": cfg.model.num_heads,
                "num_hidden_layers": cfg.model.num_hidden_layers,
                "head_dim": cfg.model.head_dim,
                "state_size": cfg.model.state_size,
                "n_groups": cfg.model.n_groups,
                "expand": cfg.model.expand,
            }
        model = MultiTaskTGMMModel(
            task=task,
            model_type=cfg.model.model_type,
            **model_args
        )
    elif cfg.task.type == "PhaseTransitionGaussianMixture":
        # Use a-b-n configuration from https://arxiv.org/abs/1812.08078
        task = MultiTaskIsotropicGaussianMixtureTask.abn_config(
            a_s=cfg.task.a_s,
            b=cfg.task.b,
            n=cfg.train.n_sample,
        )
        model = TGMMModel(
            task=task.tasks[0],  # Doesn't matter which specific task is
            n_positions=cfg.model.n_positions,
            n_embd=cfg.model.n_embd,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
        )
    else:
        raise ValueError(cfg.task.type)
    return task, model


def evaluate(
    task: MultiTaskIsotropicGaussianMixtureTask,
    model: Union[TGMMModel, MultiTaskTGMMModel],
    device,
    loss_meter: StreamingLossMeter,
    cfg,
    step,
):

    model.eval()
    summary_dict = {}

    def _eval(
        subtask: Union[
            IsotropicGaussianMixtureTask,
            SphericalGaussianMixtureTask
        ],
        eval_n_sample,
    ):
        if cfg.task.type == "PhaseTransitionGaussianMixture":
            prefix = f"a_{subtask.a:.4f}-b_{subtask.b:.4f}"
        else:
            prefix = f"K_{subtask.n_components}-N_{eval_n_sample}"
        with torch.no_grad():
            if cfg.eval.ood_perturbation_scale > 0.:
                subtask = OODIsotropicGaussianMixtureTask.from_id_task(
                    subtask,
                    perturbation_scale=cfg.eval.ood_perturbation_scale
                )
            task_sample = subtask.sample(
                n_sample=eval_n_sample,
                batch_size=cfg.eval.batch_size,
                gen_mask=False,
            ).to(device)
            task_sample_for_eval = task_sample.to("cpu")
            task_sample.pad(task.max_n_components)
            evaluator = GMMEvaluator(task=subtask, ground_truth=task_sample_for_eval)
            model_output = model(task_sample)
            # Adjust mask manually
            model_output.alpha_est = model_output.alpha_est[:, : subtask.n_components]
            model_output.mu_est = model_output.mu_est[:, : subtask.n_components, :]
            eval_results_tgmm = evaluator(
                mu_est=model_output.mu_est.cpu(),
                alpha_est=F.softmax(model_output.alpha_est.cpu(), dim=-1),
                in_sample_eval=True,
            )
            alpha_loss, mu_loss, total_loss = loss_meter.compute()
            summary = {
                "step": step,
                f"{prefix}.tgmm_summary": eval_results_tgmm.summary_for_wandb(),
                # Some auxiliary metrics
                f"{prefix}.alpha_loss": (
                    alpha_loss.cpu().item() if alpha_loss is not None else None
                ),
                f"{prefix}.mu_loss": (
                    mu_loss.cpu().item() if mu_loss is not None else None
                ),
                f"{prefix}.total_loss": (
                    total_loss.cpu().item() if total_loss is not None else None
                ),
            }
            if cfg.task.type != "PhaseTransitionGaussianMixture":
                gmm_em = GaussianMixtureEM(
                    n_components=subtask.n_components,
                    n_features=subtask.dim,
                    verbose=cfg.train.verbose,
                )
                gmm_spectral = GaussianMixtureSpectral(
                    n_components=subtask.n_components,
                    verbose=cfg.train.verbose,
                    # n_repeat=100,
                    # n_iteration=20,
                )
                alpha_est_em, mu_est_em, _, iter_em = gmm_em.fit_batch(
                    task_sample.sample.cpu()
                )
                alpha_est_spectral, mu_est_spectral, _ = gmm_spectral.fit_batch(
                    task_sample.sample.cpu()
                )
                eval_results_em = evaluator(
                    mu_est=mu_est_em,
                    alpha_est=alpha_est_em,
                    in_sample_eval=True,
                )
                eval_results_spectral = evaluator(
                    mu_est=mu_est_spectral,
                    alpha_est=alpha_est_spectral,
                    in_sample_eval=True,
                )
                mean_iter_em = iter_em.mean().item()
                baseline_summary = {
                    "step": step,
                    f"{prefix}.em_summary": eval_results_em.summary_for_wandb(),
                    f"{prefix}.spectral_summary": eval_results_spectral.summary_for_wandb(),
                    # Some auxiliary metrics
                    f"{prefix}.em_iter": mean_iter_em,
                }
                summary.update(baseline_summary)
        return summary

    # For legacy compatibility
    n_samples = [cfg.eval.n_sample] \
        if isinstance(cfg.eval.n_sample, int) else cfg.eval.n_sample
    for subtask in task.tasks:
        for n in n_samples:
            summary_dict.update(_eval(subtask, n))
    model.train()
    return summary_dict


def train(cfg, device_id, name: str = None):
    r"""Training pipeline"""

    device = get_device(device_id)
    seed_everything(cfg.train.seed)
    task, model = _init_task_and_model(cfg)
    model = model.to(device)
    dataset = GaussianMixtureDataset(
        batch_size=cfg.train.batch_size,
        task=task,
        n_sample=cfg.train.n_sample
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # Batch size is handled inside dataset iterator
        num_workers=4,
        pin_memory=True,
        collate_fn=concat_task_sample,
    )
    it = iter(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    num_steps = cfg.train.num_train_steps
    loss_meter = StreamingLossMeter(n_metrics=3, window_size=cfg.train.eval_every).to(
        device=device
    )
    model.train()
    pbar = tqdm(range(num_steps)) if cfg.train.verbose else range(num_steps)
    eval_results = []
    for step in pbar:
        if not step % cfg.train.eval_every:
            eval_results.append(evaluate(task, model, device, loss_meter, cfg, step))
        # task_sample = task.sample(
        #     n_sample=cfg.train.n_sample,
        #     batch_size=cfg.train.batch_size,
        # ).to(device)
        task_sample = next(it).to(device=device, non_blocking=True)
        optimizer.zero_grad()
        model_output = model(task_sample)
        total_loss = model_output.alpha_loss + model_output.mu_loss
        loss_meter.update(
            model_output.alpha_loss,
            model_output.mu_loss,
            total_loss,
        )
        total_loss.backward()
        optimizer.step()
        if cfg.train.verbose:
            pbar.set_description(
                f"alpha_loss: {model_output.alpha_loss.cpu().detach().numpy():.4f}\t"
                f"mu_loss: {model_output.mu_loss.cpu().detach().numpy():.4f}\t"
                f"total_loss: {total_loss.cpu().detach().numpy():.4f}"
            )
    return eval_results
