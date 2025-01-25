from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

from .task import IsotropicGaussianMixtureSample, IsotropicGaussianMixtureTask
from .utils import default_device


class AttentivePooling(nn.Module):
    r"""Using attention mechanisms for pooling"""

    def __init__(self, d_in, d_out, n_out):
        super(AttentivePooling, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_out = n_out
        self.q = nn.Parameter(torch.empty(n_out, d_out), requires_grad=True)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        # TODO: by theory we do not need an out_proj, but could be helpful?
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q)

    def forward(self, x):
        r"""Pool x using an attentive fashion

        Args:
            x (torch.Tensor): input tensor of shape [batch_size, seq_len, d_in]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, n_out, d_out]
        """
        k, v = self.k_proj(x), self.v_proj(x)
        weights = F.softmax(k @ self.q.T, dim=1)  # [batch_size, seq_len, n_out]
        result = torch.einsum("bld,bln->bldn", v, weights).sum(dim=1)
        return torch.permute(result, (0, 2, 1))


@dataclass
class TEMOutput(object):
    r"""For wrapping outputs of TEMModel"""

    h: torch.Tensor
    alpha_est: torch.Tensor
    mu_est: torch.Tensor
    alpha_loss: torch.Tensor
    mu_loss: torch.Tensor


class TEMModel(nn.Module):
    r"""A wrapper of huggingface impl of GPT-2 model"""

    def __init__(
        self,
        task: IsotropicGaussianMixtureTask,
        n_positions=100,
        n_embd=128,
        n_layer=12,
        n_head=4,
    ):
        super(TEMModel, self).__init__()
        # TODO: Allow more transformer configurations
        transformer_config = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.task = task
        n_embd = transformer_config.n_embd
        self.read_in = nn.Linear(self.task.dim, n_embd)
        self.transformer = GPT2Model(transformer_config)
        d_out = self.n_components + self.task.dim
        self.read_out = AttentivePooling(
            d_in=n_embd, d_out=d_out, n_out=self.n_components
        )
        # Loss functions
        self.alpha_loss = nn.CrossEntropyLoss()
        self.mu_loss = nn.MSELoss()
        self.to(default_device)

    @property
    def n_components(self):
        return self.task.n_components

    def forward(self, inputs: IsotropicGaussianMixtureSample):
        x = inputs.sample
        embeds = self.read_in(x)
        h = self.transformer(inputs_embeds=embeds).last_hidden_state
        out_combn = self.read_out(h)
        alpha_est = out_combn[:, :, : self.n_components].mean(dim=1)
        mu_est = torch.permute(out_combn[:, :, self.n_components :], (0, 2, 1))
        alpha_loss_val = self.alpha_loss(alpha_est, inputs.mixture_probs)
        mu_loss_val = self.mu_loss(mu_est, inputs.gaussian_means)
        return TEMOutput(
            alpha_loss=alpha_loss_val,
            mu_loss=mu_loss_val,
            alpha_est=alpha_est,
            mu_est=mu_est,
            h=h,
        )
