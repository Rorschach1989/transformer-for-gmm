import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models.tgmm import MultiTaskTGMMModel
from .models.em import GaussianMixtureEM
from .models.spectral import GaussianMixtureSpectral
from .logger import logger
from .evaluation import GMMEvaluator
from .task import IsotropicGaussianMixtureTask, MultiTaskIsotropicGaussianMixtureTask
from .utils import (
    seed_everything,
    wandb_profile,
    get_device,
    StreamingLossMeter
)


def evaluate(
    task: MultiTaskIsotropicGaussianMixtureTask,
    model: MultiTaskTGMMModel,
    device,
    loss_meter: StreamingLossMeter,
    cfg,
    step,
):
    # logger.info(f"Eval@step={step}")
    model.eval()
    summary_dict = {}

    def _eval(subtask: IsotropicGaussianMixtureTask):
        prefix = f"K_{subtask.n_components}"
        with torch.no_grad():
            task_sample = subtask.sample(
                n_sample=cfg.eval.n_sample * 2,
                batch_size=8,
                gen_mask=False,
            ).to(device)
            task_sample_for_eval = task_sample.to("cpu")
            task_sample.pad(task.max_n_components)
            gmm_em = GaussianMixtureEM(
                n_components=subtask.n_components,
                n_features=cfg.task.dim,
            )
            gmm_spectral = GaussianMixtureSpectral(
                n_components=subtask.n_components,
                # n_repeat=100,
                # n_iteration=20,
            )
            evaluator = GMMEvaluator(
                task=subtask,
                ground_truth=task_sample_for_eval
            )
            model_output = model(task_sample)
            # Adjust mask manually
            model_output.alpha_est = model_output.alpha_est[:, :subtask.n_components]
            model_output.mu_est = model_output.mu_est[:, :subtask.n_components, :]
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
            eval_results_tgmm = evaluator(
                mu_est=model_output.mu_est.cpu(),
                alpha_est=F.softmax(model_output.alpha_est.cpu(), dim=-1),
                in_sample_eval=True,
            )
            mean_iter_em = iter_em.mean().item()
            alpha_loss, mu_loss, total_loss = loss_meter.compute()
            summary = {
                f"{prefix}.em_summary": eval_results_em.summary_for_wandb(),
                f"{prefix}.spectral_summary": eval_results_spectral.summary_for_wandb(),
                f"{prefix}.tgmm_summary": eval_results_tgmm.summary_for_wandb(),
                # Some auxiliary metrics
                f"{prefix}.em_iter": mean_iter_em,
                f"{prefix}.alpha_loss": alpha_loss.cpu().item() if alpha_loss is not None else None,
                f"{prefix}.mu_loss": mu_loss.cpu().item() if mu_loss is not None else None,
                f"{prefix}.total_loss": total_loss.cpu().item() if total_loss is not None else None,
            }
        return summary

    for subtask in task.tasks:
        summary_dict.update(_eval(subtask))
    wandb.log(summary_dict, step=step)
    model.train()
    return summary_dict


def train(cfg, device_id, name):
    r"""Training pipeline"""
    device = get_device(device_id)
    wandb.init(
        project=wandb_profile.project,
        # entity=wandb_profile.entity,
        name=name,
    )
    seed_everything(cfg.train.seed)
    # Initialize task
    task_list = [
        IsotropicGaussianMixtureTask(
            n_components=n,
            dim=cfg.task.dim,
        )
        for n in cfg.task.n_components
    ]
    task = MultiTaskIsotropicGaussianMixtureTask(task_list)
    model = MultiTaskTGMMModel(
        task=task,
        n_positions=cfg.model.n_positions,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    num_steps = cfg.train.num_train_steps
    loss_meter = StreamingLossMeter(
        n_metrics=3,
        window_size=cfg.train.eval_every
    ).to(device=device)
    model.train()
    pbar = tqdm(range(num_steps)) if cfg.train.verbose else range(num_steps)
    eval_results = []
    for step in pbar:
        if not step % cfg.train.eval_every:
            eval_results.append(evaluate(task, model, device, loss_meter, cfg, step))
        task_sample = task.sample(
            n_sample=cfg.train.n_sample,
            batch_size=cfg.train.batch_size,
        ).to(device)
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
    wandb.finish()
    return eval_results
