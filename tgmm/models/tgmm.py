from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2Model,
    GPT2Config,
    Mamba2Model,
    Mamba2Config,
)
# Stuffs related to Qwen3
from transformers.models.qwen3 import (
    Qwen3Model
)

from ..task import (
    IsotropicGaussianMixtureSample,
    IsotropicGaussianMixtureTask,
    MultiTaskGaussianMixtureTask,
)


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

    def forward(self, x, mask: torch.Tensor = None):
        r"""Pool x using an attentive fashion

        Args:
            x (torch.Tensor): input tensor of shape [batch_size, n_sample, d_in]
            mask (torch.Tensor): mask tensor of shape [batch_size, n_sample]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, n_out, d_out]
        """
        k, v = self.k_proj(x), self.v_proj(x)
        weights_ = k @ self.q.T
        if mask is not None:
            attn_mask = mask.float().unsqueeze(-1)
            weights_ = weights_ - attn_mask * 1e9
        weights = F.softmax(weights_, dim=1)  # [batch_size, seq_len, n_out]
        result = torch.einsum("bld,bln->bldn", v, weights).sum(dim=1)
        return torch.permute(result, (0, 2, 1))


class _RNNReadout(nn.Module):
    r"""Wrapper for readout function in RNN mode like Mamba
    Only decoding the last hidden state

    **Notes**: According to some preliminary experiments, this
    method seems to yield inferior results to ``AttentivePooling``
    even with pure-RNN models.
    """

    def __init__(self, d_in, d_out, n_out):
        super(_RNNReadout, self).__init__()
        self.d_out = d_out
        self.n_out = n_out
        self.proj = nn.Linear(d_in, d_out * n_out, bias=False)

    def forward(self, x, mask: torch.Tensor = None):
        return self.proj(x[:, -1, :]).view(-1, self.n_out, self.d_out)


@dataclass
class TGMMOutput(object):
    r"""For wrapping outputs of TEMModel"""

    h: torch.Tensor
    alpha_est: torch.Tensor
    mu_est: torch.Tensor
    alpha_loss: torch.Tensor
    mu_loss: torch.Tensor
    # scale estimation is not necessary for Isotropic tasks
    scale_est: torch.Tensor = None
    scale_loss: torch.Tensor = None

    def to_predictions(self):
        out_dict = OrderedDict()  # Make sure a canonical unpacking order
        out_dict["alpha_est"] = self.alpha_est
        out_dict["mu_est"] = self.mu_est
        if self.scale_est is not None:
            out_dict["scale_est"] = self.scale_est
        return out_dict


class TGMMModel(nn.Module):
    r"""A wrapper of huggingface impl of GPT-2 model"""

    def __init__(
        self,
        task: IsotropicGaussianMixtureTask,
        n_positions=100,
        n_embd=128,
        n_layer=12,
        n_head=4,
    ):
        super(TGMMModel, self).__init__()
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
        self.n_components = self.task.n_components
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

    def forward(self, inputs: IsotropicGaussianMixtureSample):
        x = inputs.sample
        embeds = self.read_in(x)
        h = self.transformer(inputs_embeds=embeds).last_hidden_state
        out_combn = self.read_out(h)
        alpha_est = out_combn[:, :, : self.n_components].mean(dim=1)
        mu_est = out_combn[:, :, self.n_components :]
        alpha_loss_val = self.alpha_loss(alpha_est, inputs.mixture_probs)
        mu_loss_val = self.mu_loss(mu_est, inputs.gaussian_means)  # [b, n, d]
        return TGMMOutput(
            alpha_loss=alpha_loss_val,
            mu_loss=mu_loss_val,
            alpha_est=alpha_est,
            mu_est=mu_est,
            h=h,
        )


class MultiTaskTGMMModel(nn.Module):
    r"""Multi-task version of TGMMModel"""

    @staticmethod
    def _prepare_backbone(model_type, **model_args):
        if model_type == "transformer":
            n_positions = model_args.get("n_positions", 100)
            n_embd = model_args.get("n_embd", 128)
            n_layer = model_args.get("n_layer", 12)
            n_head = model_args.get("n_head", 4)
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
            return transformer_config, GPT2Model(transformer_config)
        elif model_type == "mamba2":
            mamba2_config = Mamba2Config(
                num_heads=model_args.get("num_heads", 8),
                head_dim=model_args.get("head_dim", 64),
                hidden_size=model_args.get("hidden_size", 128),
                state_size=model_args.get("state_size", 16),
                n_groups=model_args.get("n_groups", 2),
                expand=model_args.get("expand", 4),
                num_hidden_layers=model_args.get("num_hidden_layers", 12),
            )
            return mamba2_config, Mamba2Model(mamba2_config)
        elif model_type.startswith("qwen"):
            pretrained_ckpt_path = model_args.get("pretrained_ckpt_path")
            qwen_model = Qwen3Model.from_pretrained(pretrained_ckpt_path)
            # TODO: we shall allow for fine-grained control of parameter tunability
            return qwen_model.config, qwen_model
        else:
            raise NotImplementedError

    @staticmethod
    def _prepare_readout(model_type, **model_args):
        if model_type == "transformer":
            return AttentivePooling(**model_args)
        elif model_type == "mamba2":
            # return _RNNReadout(**model_args)
            return AttentivePooling(**model_args)
        elif model_type.startswith("qwen"):
            return AttentivePooling(**model_args)
        else:
            raise NotImplementedError

    def __init__(
        self,
        task: MultiTaskGaussianMixtureTask,
        model_type="transformer",
        n_task_embd=128,
        **kwargs,
    ):
        super(MultiTaskTGMMModel, self).__init__()
        # TODO: Allow more transformer configurations
        self.task = task
        self.is_isotropic = kwargs.pop("is_isotropic", True)
        self.n_components = self.task.max_n_components
        self.n_subtasks = self.task.n_subtasks
        self.subtask_components = self.task.subtask_components
        # Task embedding that embed components
        self.task_embedding = nn.Embedding(self.n_components, n_task_embd)
        if model_type.startswith("qwen"):
            n_embd = 1024
        else:
            n_embd = kwargs.get("n_embd", 128)
        # TODO: maybe enrich the method of injecting task information
        self.read_in = nn.Linear(self.task.dim + n_task_embd, n_embd)
        self.config, self.encoder = self._prepare_backbone(model_type, **kwargs)
        d_out = self.n_components + self.task.dim * (1 if self.is_isotropic else 2)
        self.read_outs = nn.ModuleList()
        readout_model_args = {"d_in": n_embd, "d_out": d_out, "n_out": self.n_components}
        for i in range(self.n_subtasks):
            self.read_outs.append(
                self._prepare_readout(model_type, **readout_model_args)
            )
        # Loss functions
        self.alpha_loss = nn.CrossEntropyLoss()
        self.mu_loss = nn.MSELoss(reduction="none")
        self.scale_loss = (
            nn.MSELoss(reduction="none") if not self.is_isotropic else None
        )

    def _map_component_ids(self, component_ids: torch.Tensor):
        # TODO: this is not the most efficient way
        return torch.stack(
            [
                i * (component_ids == n_components - 1).long()
                for i, n_components in enumerate(self.subtask_components)
            ],
            dim=1,
        ).sum(dim=1)

    def forward(self, inputs: IsotropicGaussianMixtureSample):
        x = inputs.sample
        component_ids = inputs.mask_components.sum(dim=1).long() - 1
        task_embeds = (
            self.task_embedding(component_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        )
        x = torch.cat([x, task_embeds], dim=-1)
        embeds = self.read_in(x)
        h = self.encoder(
            inputs_embeds=embeds,
            attention_mask=inputs.mask_length,
        ).last_hidden_state
        out_combn = torch.stack(
            [read_out(h, mask=inputs.mask_length) for read_out in self.read_outs], dim=1
        )
        results = torch.gather(
            out_combn,
            1,
            self._map_component_ids(component_ids)
            .view(-1, 1, 1, 1)
            .expand(-1, -1, out_combn.size(2), out_combn.size(3)),
        ).squeeze(dim=1)
        # Separate logics between isotropic and anisotropic cases
        # TODO: refine design
        if self.is_isotropic:
            alpha_est = results[:, :, : self.n_components].mean(dim=1)
            mu_est = results[:, :, self.n_components :]
            alpha_est = alpha_est - (1.0 - inputs.mask_components) * 1e9
            alpha_loss_val = self.alpha_loss(alpha_est, inputs.mixture_probs)
            mu_loss_val_ = self.mu_loss(mu_est, inputs.gaussian_means)  # [b, n, d]
            mask = inputs.mask_components
            mu_loss_sum_ = (mu_loss_val_ * mask.unsqueeze(-1)).mean(dim=-1).sum(dim=-1)
            mu_loss_val = (mu_loss_sum_ / mask.sum(dim=1)).mean()
            return TGMMOutput(
                alpha_loss=alpha_loss_val,
                mu_loss=mu_loss_val,
                alpha_est=alpha_est,
                mu_est=mu_est,
                h=h,
            )
        else:
            alpha_est = results[:, :, : self.n_components].mean(dim=1)
            mu_est = results[
                :, :, self.n_components : (self.n_components + self.task.dim)
            ]
            scale_est = F.softplus(results[:, :, (self.n_components + self.task.dim) :])
            alpha_est = alpha_est - (1.0 - inputs.mask_components) * 1e9
            alpha_loss_val = self.alpha_loss(alpha_est, inputs.mixture_probs)
            mu_loss_val_ = self.mu_loss(mu_est, inputs.gaussian_means)  # [b, n, d]
            scale_loss_val_ = self.scale_loss(scale_est, inputs.scale)
            mask = inputs.mask_components
            mu_loss_sum_ = (mu_loss_val_ * mask.unsqueeze(-1)).mean(dim=-1).sum(dim=-1)
            mu_loss_val = (mu_loss_sum_ / mask.sum(dim=1)).mean()
            scale_loss_sum_ = (
                (scale_loss_val_ * mask.unsqueeze(-1)).mean(dim=-1).sum(dim=-1)
            )
            scale_loss_val = (scale_loss_sum_ / mask.sum(dim=1)).mean()
            return TGMMOutput(
                alpha_loss=alpha_loss_val,
                mu_loss=mu_loss_val,
                alpha_est=alpha_est,
                mu_est=mu_est,
                h=h,
                scale_loss=scale_loss_val,
                scale_est=scale_est,
            )


class HFMultiTaskTGMMModel(MultiTaskTGMMModel):
    r"""A dispatching workaround for supporting TGMM
    in huggingface transformers Trainer"""

    def forward(
        self,
        mixture_probs,
        assignment,
        gaussian_means,
        sample,
        scale,
        mask_length,
        mask_components,
    ):
        inputs = IsotropicGaussianMixtureSample(
            mixture_probs=mixture_probs,
            assignment=assignment,
            gaussian_means=gaussian_means,
            sample=sample,
            scale=scale,
            mask_length=mask_length,
            mask_components=mask_components,
        )
        return super(HFMultiTaskTGMMModel, self).forward(inputs)
