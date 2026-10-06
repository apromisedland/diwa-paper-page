"""Shared utilities for simulator-measured DIWA rollout supervision."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateRollout:
    q_values: np.ndarray
    rewards: np.ndarray
    progress: np.ndarray
    dones: np.ndarray


def shaped_task_reward(
    previous_progress: float,
    next_progress: float,
    success: bool,
    *,
    success_bonus: float = 1.0,
) -> float:
    """Potential-difference reward backed by a measured task predicate."""
    previous = float(np.clip(previous_progress, 0.0, 1.0))
    following = float(np.clip(next_progress, 0.0, 1.0))
    return following - previous + success_bonus * float(success)


def measure_candidate_rollouts(
    candidate_actions: np.ndarray,
    *,
    restore_state: Callable[[], None],
    step_action: Callable[[np.ndarray], None],
    measure_progress: Callable[[], float],
    measure_success: Callable[[], bool],
    gamma: float = 0.95,
    success_bonus: float = 1.0,
) -> CandidateRollout:
    """Restore one simulator state and measure every candidate action chunk.

    ``candidate_actions`` has shape ``[candidate, action_step, action_dim]``.
    Each candidate is executed from the exact same restored simulator state.
    Q is the discounted sum of task-predicate potential differences and the
    official success predicate bonus. No learned critic participates.
    """
    candidates = np.asarray(candidate_actions, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[0] < 2:
        raise ValueError(
            "candidate_actions must have shape [candidate>=2, action_step, action_dim]"
        )
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")

    num_candidates, horizon = candidates.shape[:2]
    q_values = np.zeros(num_candidates, dtype=np.float32)
    rewards = np.zeros((num_candidates, horizon), dtype=np.float32)
    progress = np.zeros((num_candidates, horizon), dtype=np.float32)
    dones = np.zeros((num_candidates, horizon), dtype=bool)
    for candidate_index, action_chunk in enumerate(candidates):
        restore_state()
        previous = float(measure_progress())
        if not np.isfinite(previous):
            raise RuntimeError("initial task progress is non-finite")
        discount = 1.0
        for step_index, action in enumerate(action_chunk):
            step_action(action)
            following = float(measure_progress())
            success = bool(measure_success())
            reward = shaped_task_reward(
                previous,
                following,
                success,
                success_bonus=success_bonus,
            )
            rewards[candidate_index, step_index] = reward
            progress[candidate_index, step_index] = following
            dones[candidate_index, step_index] = success
            q_values[candidate_index] += discount * reward
            if success:
                break
            previous = following
            discount *= gamma

    if not (
        np.isfinite(q_values).all()
        and np.isfinite(rewards).all()
        and np.isfinite(progress).all()
    ):
        raise RuntimeError("candidate rollout measurement produced non-finite values")
    return CandidateRollout(q_values, rewards, progress, dones)


def validate_measured_candidate_q(
    q_values: np.ndarray,
    *,
    minimum_spread: float = 1e-7,
) -> dict[str, float | bool]:
    """Reject masked or decision-degenerate candidate supervision."""
    values = np.asarray(q_values, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("candidate Q values need a candidate dimension >= 2")
    if not np.isfinite(values).all():
        raise ValueError("measured candidate Q values must all be finite")
    spread = np.ptp(values, axis=-1)
    maximum_spread = float(spread.max(initial=0.0))
    if maximum_spread <= minimum_spread:
        raise ValueError(
            "measured candidate Q values are constant; no regret ordering exists"
        )
    return {
        "all_finite": True,
        "maximum_spread": maximum_spread,
        "nonconstant_fraction": float((spread > minimum_spread).mean()),
    }
