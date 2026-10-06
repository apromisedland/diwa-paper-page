from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class CriticLossOutput:
    critic_loss: Tensor
    progress_loss: Tensor
    q_values: Tensor
    progress: Tensor


class DecisionCritic(nn.Module):
    """Twin TD critic and task-progress estimator used by DIWA."""

    def __init__(
        self,
        hidden_dim: int,
        action_pred_steps: int,
        action_dim: int = 7,
        discount: float = 0.99,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if action_pred_steps < 1:
            raise ValueError("action_pred_steps must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self.hidden_dim = hidden_dim
        self.action_pred_steps = action_pred_steps
        self.action_dim = action_dim
        self.flat_action_dim = action_pred_steps * action_dim
        self.discount = discount

        critic_input_dim = hidden_dim + self.flat_action_dim
        self.q1 = _mlp(critic_input_dim, hidden_dim, 1)
        self.q2 = _mlp(critic_input_dim, hidden_dim, 1)
        self.target_q1 = deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).requires_grad_(False)
        self.progress_head = _mlp(hidden_dim, hidden_dim, 1)

    def _critic_input(self, state: Tensor, actions: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"state must have shape [batch, {self.hidden_dim}]"
            )
        if actions.shape != (
            state.shape[0],
            self.action_pred_steps,
            self.action_dim,
        ):
            raise ValueError(
                "actions must have shape "
                f"[batch, {self.action_pred_steps}, {self.action_dim}]"
            )
        flat_actions = actions.reshape(actions.shape[0], self.flat_action_dim)
        return torch.cat((state, flat_actions.to(dtype=state.dtype)), dim=-1)

    def q_values(self, state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        critic_input = self._critic_input(state, actions)
        return self.q1(critic_input).squeeze(-1), self.q2(critic_input).squeeze(-1)

    def target_q_values(
        self, state: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        critic_input = self._critic_input(state, actions)
        return (
            self.target_q1(critic_input).squeeze(-1),
            self.target_q2(critic_input).squeeze(-1),
        )

    def minimum_q(self, state: Tensor, actions: Tensor, *, target: bool = False) -> Tensor:
        if target:
            q1, q2 = self.target_q_values(state, actions)
        else:
            q1, q2 = self.q_values(state, actions)
        return torch.minimum(q1, q2)

    def progress(self, state: Tensor) -> Tensor:
        return torch.sigmoid(self.progress_head(state).squeeze(-1))

    def losses(
        self,
        state: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        progress_targets: Tensor,
        valid_mask: Tensor,
        next_state: Tensor,
        next_actions: Tensor,
        next_valid_mask: Tensor,
    ) -> CriticLossOutput:
        q1, q2 = self.q_values(state, actions)
        with torch.no_grad():
            next_q = self.minimum_q(next_state, next_actions, target=True)
            td_target = rewards.float() + self.discount * (
                1.0 - dones.float()
            ) * next_q.float()
        transition_mask = valid_mask.bool() & (dones.bool() | next_valid_mask.bool())
        if transition_mask.any():
            critic_loss = F.smooth_l1_loss(
                q1[transition_mask].float(), td_target[transition_mask]
            ) + F.smooth_l1_loss(
                q2[transition_mask].float(), td_target[transition_mask]
            )
        else:
            critic_loss = state.new_zeros(())

        progress = self.progress(state)
        if valid_mask.any():
            progress_loss = F.binary_cross_entropy(
                progress[valid_mask].float(),
                progress_targets[valid_mask].float().clamp(0.0, 1.0),
            )
        else:
            progress_loss = state.new_zeros(())
        return CriticLossOutput(
            critic_loss=critic_loss.to(state.dtype),
            progress_loss=progress_loss.to(state.dtype),
            q_values=torch.minimum(q1, q2),
            progress=progress,
        )

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        for online, target in (
            (self.q1, self.target_q1),
            (self.q2, self.target_q2),
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.lerp_(online_parameter, tau)
