import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from tem.task import IsotropicGaussianMixtureTask
from tem.model import TEMModel
from tem.em import GaussianMixtureEM
from tem.spectral import GaussianMixtureSpectral
from tem.logger import logger


def setup_cfg(**kwargs):
    r"""Setup an experiment configuration"""
    cfg = OmegaConf.create()
    cfg.task = {}
    cfg.task.type = kwargs.get("task_type", "IsotropicGaussianMixture")
    cfg.task.n_sample = kwargs.get("n_sample", 64)
    if cfg.task.type == "IsotropicGaussianMixture":
        cfg.task.n_components = kwargs.get("n_components", 2)
        cfg.task.dim = kwargs.get("mixture_dim", 2)
    else:
        raise NotImplementedError
    cfg.model = {}
    cfg.model.n_positions = kwargs.get("n_positions", 4096)
    cfg.model.n_embd = kwargs.get("n_embd", 128)
    cfg.model.n_layer = kwargs.get("n_layer", 12)
    cfg.model.n_head = kwargs.get("n_head", 4)
    cfg.train = {}
    cfg.train.batch_size = kwargs.get("batch_size", 64)
    cfg.train.eval_every = kwargs.get("eval_every", 1000)
    cfg.train.learning_rate = kwargs.get("learning_rate", 1e-3)
    cfg.train.num_train_steps = kwargs.get("num_train_steps", 10000)
    return cfg


def evaluate(task, model, cfg, step):
    r"""Evaluation against both ground truth and EM-algorithm solutions"""
    logger.info(f"Eval@step={step}")
    model.eval()
    with torch.no_grad():
        task_sample = task.sample(
            n_sample=cfg.task.n_sample,
            batch_size=1,
        )
        gmm_em = GaussianMixtureEM(
            n_components=cfg.task.n_components,
            n_features=cfg.task.dim,
        )
        gmm_spectral = GaussianMixtureSpectral(
            n_components=cfg.task.n_components
        )
        X = task_sample.sample.squeeze(0)
        gmm_em.fit(X)
        w_spectral, mu_spectral = gmm_spectral.fit(X.cpu())
        model_output = model(task_sample)
        logger.info(f"{">" * 20} Ground truth {"<" * 20}")
        logger.info(f"mu:\n {task_sample.gaussian_means}")
        logger.info(f"alpha:\n {task_sample.mixture_probs}")
        logger.info(f"{">" * 20} Transformer prediction {"<" * 20}")
        logger.info(f"mu:\n {model_output.mu_est}")
        logger.info(f"alpha:\n {F.softmax(model_output.alpha_est, dim=-1)}")
        logger.info(f"{">" * 20} Standard EM {"<" * 20}")
        logger.info(f"mu:\n {gmm_em.means}")
        logger.info(f"alpha:\n {gmm_em.weights}")
        logger.info(f"{">" * 20} Spectral algorithm {"<" * 20}")
        logger.info(f"mu:\n {mu_spectral}")
        logger.info(f"alpha:\n {w_spectral}")
    model.train()


def train(cfg):
    r"""Training pipeline"""
    # Initialize task
    task = IsotropicGaussianMixtureTask(
        n_components=cfg.task.n_components,
        dim=cfg.task.dim,
    )
    model = TEMModel(
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
