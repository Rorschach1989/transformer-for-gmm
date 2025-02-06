import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from tgmm.task import IsotropicGaussianMixtureTask
from tgmm.models.tgmm import TGMMModel
from tgmm.models.em import GaussianMixtureEM
from tgmm.models.spectral import GaussianMixtureSpectral
from tgmm.logger import logger
from tgmm.evaluation import GMMEvaluator
from tgmm.utils import seed_everything


def setup_cfg(**kwargs):
    r"""Setup an experiment configuration"""
    cfg = OmegaConf.create()
    cfg.task = {}
    cfg.task.type = kwargs.get("task_type", "IsotropicGaussianMixture")
    cfg.task.n_sample = kwargs.get("n_sample", 64)
    if cfg.task.type == "IsotropicGaussianMixture":
        cfg.task.n_components = kwargs.get("n_components", 3)
        cfg.task.dim = kwargs.get("mixture_dim", 8)
    else:
        raise NotImplementedError
    cfg.model = {}
    cfg.model.n_positions = kwargs.get("n_positions", 4096)
    cfg.model.n_embd = kwargs.get("n_embd", 128)
    cfg.model.n_layer = kwargs.get("n_layer", 12)
    cfg.model.n_head = kwargs.get("n_head", 4)
    cfg.train = {}
    cfg.train.seed = kwargs.get("seed", 42)
    cfg.train.batch_size = kwargs.get("batch_size", 64)
    cfg.train.eval_every = kwargs.get("eval_every", 1000)
    cfg.train.learning_rate = kwargs.get("learning_rate", 1e-3)
    cfg.train.num_train_steps = kwargs.get("num_train_steps", 10001)
    return cfg


def evaluate(task, model, cfg, step):
    logger.info(f"Eval@step={step}")
    model.eval()
    with torch.no_grad():
        task_sample = task.sample(
            n_sample=cfg.task.n_sample,
            batch_size=8,
        )
        gmm_em = GaussianMixtureEM(
            n_components=cfg.task.n_components,
            n_features=cfg.task.dim,
        )
        gmm_spectral = GaussianMixtureSpectral(
            n_components=cfg.task.n_components,
            # n_repeat=100,
            # n_iteration=20,
        )
        evaluator = GMMEvaluator(
            task=task,
            ground_truth=task_sample.to("cpu")
        )
        model_output = model(task_sample)
        alpha_est_em, mu_est_em, _ = gmm_em.fit_batch(task_sample.sample.cpu())
        alpha_est_spectral, mu_est_spectral, _ = gmm_spectral.fit_batch(task_sample.sample.cpu())
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
        logger.info(f"EM result: \n{eval_results_em.summary()}")
        logger.info(f"Spectral result: \n{eval_results_spectral.summary()}")
        logger.info(f"TGMM result: \n{eval_results_tgmm.summary()}")
    model.train()


def train(cfg):
    r"""Training pipeline"""
    seed_everything(cfg.train.seed)
    # Initialize task
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    num_steps = cfg.train.num_train_steps
    model.train()
    pbar = tqdm(range(num_steps))
    for step in pbar:
        if not step % cfg.train.eval_every:
            evaluate(task, model, cfg, step)
        task_sample = task.sample(
            n_sample=cfg.task.n_sample,
            batch_size=cfg.train.batch_size,
        )
        optimizer.zero_grad()
        model_output = model(task_sample)
        total_loss = model_output.alpha_loss + model_output.mu_loss
        total_loss.backward()
        optimizer.step()
        pbar.set_description(
            f"alpha_loss: {model_output.alpha_loss.cpu().detach().numpy():.4f}\t"
            f"mu_loss: {model_output.mu_loss.cpu().detach().numpy():.4f}\t"
            f"total_loss: {total_loss.cpu().detach().numpy():.4f}"
        )


if __name__ == "__main__":
    cfg = setup_cfg()
    train(cfg)
