"""DIWA on precomputed causal VLA features, without a vision checkpoint.

This shares the core, stochastic action head and influence objective with
DreamVLA. It trains the DIWA modules; the precomputed VLA encoder is frozen.
Use train.py for joint image/language/proprioception training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from models.action_model.action_model import ActionModel, ActionModelFM, DiT_models
from models.action_model.respace import space_timesteps
from .core import DIWACore
from .influence import intervention_influence_loss


def load_config(path):
    config = json.loads(Path(path).read_text())
    core, training, head = config["core"], config["training"], config["action_head"]
    if not core.get("require_measured_supervision", False):
        raise ValueError("feature training requires measured supervision")
    if not 0 < core["minimum_budget_ratio"] <= training["budget_ratio"] <= 1:
        raise ValueError("invalid query-budget bounds")
    if core["hidden_dim"] < 1 or core["num_heads"] < 1:
        raise ValueError("hidden_dim and num_heads must be positive")
    if core["hidden_dim"] % core["num_heads"]:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if training["batch_size"] < 2 or training["batch_size"] % 2:
        raise ValueError("batch_size must be even and >= 2 for cross-episode pairs")
    for name in (
        "sequence_length",
        "epochs",
        "gradient_accumulation",
        "batch_size",
        "window_stride",
    ):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"{name} must be positive")
    if (
        not 0
        <= training["critic_start"]
        <= training["influence_start"]
        <= training["regret_start"]
    ):
        raise ValueError("stages must satisfy 0 <= critic <= influence <= regret")
    if config["influence"]["num_probes"] < 2:
        raise ValueError("at least two influence probes are required")
    for name in (
        "covariance_weight",
        "policy_weight",
        "value_weight",
        "progress_weight",
    ):
        value = config["influence"][name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"influence.{name} must be finite and non-negative")
    if head["kind"] not in ("diffusion", "flow"):
        raise ValueError("action_head.kind must be diffusion or flow")
    if head["model_type"] not in DiT_models:
        raise ValueError(
            f"unknown action_head.model_type: {head['model_type']}"
        )
    if head["diffusion_steps"] < 2:
        raise ValueError("diffusion_steps must be at least two")
    if not 2 <= head["sampling_steps"] <= head["diffusion_steps"]:
        raise ValueError("sampling_steps must be in [2, diffusion_steps]")
    if head["kind"] == "diffusion":
        try:
            space_timesteps(
                head["diffusion_steps"], f"ddim{head['sampling_steps']}"
            )
        except ValueError as error:
            raise ValueError(
                "sampling_steps must be exactly representable by the DDIM "
                "stride for diffusion_steps"
            ) from error
    if head["training_repeats"] < 1:
        raise ValueError("action_head.training_repeats must be positive")
    if not math.isfinite(head["cfg_scale"]) or head["cfg_scale"] < 1.0:
        raise ValueError("action_head.cfg_scale must be finite and at least one")
    if not 0 <= training["warmup_epochs"] <= training["epochs"]:
        raise ValueError("warmup_epochs must be in [0, epochs]")
    for name in ("learning_rate", "max_grad_norm", "target_tau"):
        value = training[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be finite and positive")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be finite and non-negative")
    if not 0.0 < training["target_tau"] <= 1.0:
        raise ValueError("training.target_tau must be in (0, 1]")
    for name in (
        "budget_warmup",
        "budget_anneal",
        "teacher_forcing",
        "critic_start",
        "influence_start",
        "regret_start",
    ):
        if training[name] < 0:
            raise ValueError(f"training.{name} must be non-negative")
    expected_losses = {
        "proposal",
        "mask",
        "future",
        "influence",
        "budget",
        "critic",
        "progress",
        "regret",
        "contrastive",
    }
    if set(config["loss_weights"]) != expected_losses:
        raise ValueError(
            "loss_weights must contain exactly " + ", ".join(sorted(expected_losses))
        )
    for name, value in config["loss_weights"].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss_weights.{name} must be finite and non-negative")
    return config


def schedule(config, micro_step):
    """Schedule steps count minibatches, as in the original training loop."""
    t = config["training"]
    dense = micro_step < t["budget_warmup"]
    if dense:
        budget = 1.0
    else:
        progress = min(
            1.0, (micro_step - t["budget_warmup"]) / max(1, t["budget_anneal"])
        )
        budget = 1.0 + (t["budget_ratio"] - 1.0) * progress
        if t["budget_anneal"] == 0:
            budget = t["budget_ratio"]
    teacher = (
        max(0.0, 1.0 - micro_step / t["teacher_forcing"])
        if t["teacher_forcing"] > 0
        else 0.0
    )
    weights = dict(config["loss_weights"])
    for name in ("critic", "progress"):
        if micro_step < t["critic_start"]:
            weights[name] = 0.0
    for name in ("influence", "budget"):
        if micro_step < t["influence_start"]:
            weights[name] = 0.0
    for name in ("regret", "contrastive"):
        if micro_step < t["regret_start"]:
            weights[name] = 0.0
    return budget, dense, teacher, weights


class FeatureDIWAPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = DIWACore(**config["core"])
        h = config["action_head"]
        head_class = ActionModel if h["kind"] == "diffusion" else ActionModelFM
        self.action_model = head_class(
            token_size=self.core.hidden_dim,
            model_type=h["model_type"],
            in_channels=self.core.action_dim,
            future_action_window_size=self.core.action_pred_steps - 1,
            past_action_window_size=0,
            diffusion_steps=h["diffusion_steps"],
        )
        self.register_buffer("influence_scale", torch.tensor(1.0))

    def _context(self, batch, length):
        # Online features only see current/past observations. Future frames
        # are consumed exclusively by the detached EMA target encoder.
        objects = self.core.tokenize_objects(batch["visual_tokens"][:, :length])
        context = torch.cat(
            (batch["context_tokens"][:, :length], objects.tokens), dim=2
        )
        return context, objects

    def forward(self, batch, micro_step=0):
        """Return total training loss, named losses/diagnostics, and core output."""
        if not self.training:
            raise RuntimeError("use predict() for label-free evaluation")
        length = batch["actions"].shape[1]
        context, objects = self._context(batch, length)
        targets = self.core.tokenize_target_objects(batch["visual_tokens"]).tokens
        if targets.shape[1] < length + self.core.horizon:
            raise ValueError("training needs sequence_length + horizon observations")
        future_targets = torch.stack(
            [
                targets[:, offset : offset + length]
                for offset in range(1, self.core.horizon + 1)
            ],
            dim=2,
        )
        observation_valid = batch["observation_valid"].to(
            device=context.device, dtype=torch.bool
        )
        if observation_valid.shape != (
            context.shape[0],
            length + self.core.horizon,
        ):
            raise ValueError(
                "observation_valid must cover sequence_length + horizon"
            )
        valid = observation_valid[:, :length]
        future_valid = torch.stack(
            [
                valid & observation_valid[:, offset : offset + length]
                for offset in range(1, self.core.horizon + 1)
            ],
            dim=2,
        )
        budget, dense, teacher, weights = schedule(self.config, micro_step)
        output = self.core(
            context,
            current_object_tokens=objects.tokens,
            object_tokens=objects.tokens,
            object_assignments=objects.assignments,
            action_labels=batch["actions"],
            future_targets=future_targets,
            future_valid_mask=future_valid,
            rewards=batch["rewards"],
            dones=batch["dones"],
            progress_targets=batch["progress"],
            supervision_valid_mask=valid,
            candidate_actions=batch["candidate_actions"],
            candidate_q_values=batch["candidate_q_values"],
            task_ids=batch["task_ids"],
            episode_ids=batch["episode_ids"],
            budget_ratio=budget,
            force_dense_budget=dense,
            teacher_forcing_ratio=teacher,
            compute_counterfactual=(
                micro_step >= self.config["training"]["influence_start"]
            ),
        )
        influence, diagnostics = intervention_influence_loss(
            output,
            self.core.critic,
            self.influence_scale,
            action_model=self.action_model,
            **self.config["influence"],
        )
        repeats = self.config["action_head"]["training_repeats"]
        imitation = self.action_model.loss(
            batch["actions"].flatten(0, 1).repeat(repeats, 1, 1),
            output.action_features.flatten(0, 1).repeat(repeats, 1, 1),
        )
        losses = dict(output.aux_losses, influence=influence, imitation=imitation)
        total = imitation + sum(weights[name] * losses[name] for name in weights)
        losses.update(diagnostics)
        losses["retained_queries"] = output.selected_valid_mask.float().sum(-1).mean()
        return total, losses, output

    @torch.no_grad()
    def predict(self, batch, *, initial_noise=None):
        """Predict the last observed state's normalized action chunk.

        Accepts context_tokens / visual_tokens only. No future observations,
        rewards, candidate returns or intervention branches are needed.
        """
        if self.training:
            raise RuntimeError("call model.eval() before predict()")
        length = batch["visual_tokens"].shape[1]
        context, objects = self._context(batch, length)
        output = self.core(
            context[:, -1:],
            current_object_tokens=objects.tokens[:, -1:],
            budget_ratio=self.config["training"]["budget_ratio"],
            compute_counterfactual=False,
        )
        features = output.action_features[:, 0]
        shape = (features.shape[0], self.core.action_pred_steps, self.core.action_dim)
        noise = (
            torch.randn(shape, device=features.device, dtype=features.dtype)
            if initial_noise is None
            else initial_noise
        )
        if tuple(noise.shape) != shape:
            raise ValueError(f"initial_noise must have shape {shape}")
        h = self.config["action_head"]
        scale = h["cfg_scale"]
        if scale > 1:
            noise = torch.cat((noise, noise), dim=0)
            unconditional = self.action_model.net.z_embedder.uncondition
            z = torch.cat((features, unconditional[None].expand_as(features)), dim=0)
            kwargs = {"z": z, "cfg_scale": scale}
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            kwargs, sample_fn = {"z": features}, self.action_model.net.forward
        if h["kind"] == "flow":
            # Explicit Euler from t=0 to t=1. Reuses the supplied initial noise.
            raw = noise.clone()
            for step in range(h["sampling_steps"]):
                time = raw.new_full((raw.shape[0],), step / h["sampling_steps"])
                raw += sample_fn(raw, time, **kwargs) / h["sampling_steps"]
        else:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(h["sampling_steps"])
            raw = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise=noise,
                model_kwargs=kwargs,
                device=features.device,
                clip_denoised=False,
                progress=False,
                eta=0.0,
            )
        if scale > 1:
            raw = raw.chunk(2, dim=0)[0]
        continuous = self.core.continuous_action_dim
        actions = torch.cat(
            (raw[..., :continuous].clamp(-1, 1), raw[..., continuous:].clamp(0, 1)),
            dim=-1,
        )
        return {
            "actions": actions,
            "raw_actions": raw,
            "selected_indices": output.selected_indices[:, 0],
            "selected_valid_mask": output.selected_valid_mask[:, 0],
            "influence_scores": output.influence_probabilities[:, 0],
            "budget_ratio": output.adaptive_budget_ratio[:, 0],
            "expanded_queries": torch.tensor(
                output.world_tokens.shape[1], device=actions.device
            ),
        }
