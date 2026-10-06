"""Dimension-aware state and action preprocessing."""

from __future__ import annotations

import torch


def binarize_discrete_action_channels(
    actions: torch.Tensor,
    *,
    continuous_action_dim: int,
) -> torch.Tensor:
    """Map the discrete suffix from native ``{-1, 1}`` to BCE ``{0, 1}``."""
    if not 0 < continuous_action_dim < actions.shape[-1]:
        raise ValueError(
            "continuous_action_dim must split the final action dimension"
        )
    converted = actions.clone()
    converted[..., continuous_action_dim:] = (
        converted[..., continuous_action_dim:] > 0
    ).to(converted.dtype)
    return converted


def select_policy_state(
    states: torch.Tensor,
    *,
    state_arm_dim: int,
    state_gripper_dim: int,
    gripper_width: bool,
) -> torch.Tensor:
    """Select the configured arm prefix and gripper suffix from native state."""
    if state_arm_dim < 1 or state_gripper_dim < 1:
        raise ValueError("state dimensions must be positive")
    required = state_arm_dim + state_gripper_dim
    if states.shape[-1] < required:
        raise ValueError(
            f"native state has {states.shape[-1]} values but {required} are required"
        )
    selected = torch.cat(
        (
            states[..., :state_arm_dim],
            states[..., -state_gripper_dim:],
        ),
        dim=-1,
    )
    if not gripper_width:
        if state_gripper_dim != 1:
            raise ValueError(
                "categorical gripper state requires state_gripper_dim=1"
            )
        selected = selected.clone()
        selected[..., state_arm_dim:] = (
            selected[..., state_arm_dim:] > 0
        ).to(selected.dtype)
    return selected


def decode_policy_actions(
    continuous: torch.Tensor,
    discrete_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Combine policy heads and map every discrete channel to ``{-1, 1}``."""
    discrete = (discrete_probabilities > 0.5).to(continuous.dtype)
    discrete = discrete.mul(2).sub(1)
    return torch.cat((continuous, discrete), dim=-1)
