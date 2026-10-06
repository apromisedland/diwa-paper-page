from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def _require_finite_nonempty(name: str, values: Tensor) -> Tensor:
    if values.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _require_binary(name: str, values: Tensor) -> Tensor:
    values = _require_finite_nonempty(name, values)
    if not values.eq(values.round()).all() or not (
        (values >= 0) & (values <= 1)
    ).all():
        raise ValueError(f"{name} must contain binary 0/1 values")
    return values


def _require_unit_interval(name: str, values: Tensor) -> Tensor:
    values = _require_finite_nonempty(name, values)
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"{name} must lie in [0, 1]")
    return values


def _rank(values: Tensor) -> Tensor:
    """Return zero-based average ranks, including exact ties."""
    flat = values.flatten().double()
    order = torch.argsort(flat)
    sorted_values = flat[order]
    ranks = torch.empty_like(sorted_values)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while (
            end < sorted_values.numel()
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        ranks[start:end] = (start + end - 1) / 2.0
        start = end
    output = torch.empty_like(ranks)
    output[order] = ranks
    return output


def spearman_correlation(left: Tensor, right: Tensor) -> Tensor:
    if left.numel() != right.numel() or left.numel() == 0:
        raise ValueError("Spearman inputs must have the same nonzero size")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite")
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    denominator = left_rank.norm() * right_rank.norm()
    if denominator <= 1e-12:
        return left_rank.new_zeros(())
    return ((left_rank * right_rank).sum() / denominator).clamp(-1.0, 1.0)


def influence_identification_metrics(
    influence_scores: Tensor,
    measured_influence: Tensor,
    selected_mask: Tensor,
    *,
    topk: int = 12,
) -> Dict[str, float]:
    """Fixed Top-12 recall from the paper, separately from adaptive selection.

    Arrays are [states, object_time_candidates]. Flatten batch/time into
    states and horizon/slots into candidates before calling this function.
    Exact ties use ascending candidate index in both rankings.
    """
    if min(influence_scores.ndim, measured_influence.ndim, selected_mask.ndim) < 2:
        raise ValueError("influence tensors must have at least [states, candidates] dimensions")
    if (
        influence_scores.shape != measured_influence.shape
        or selected_mask.shape != influence_scores.shape
    ):
        raise ValueError("influence tensors and selected_mask must match")
    influence_scores = influence_scores.reshape(-1, influence_scores.shape[-1])
    measured_influence = measured_influence.reshape(-1, measured_influence.shape[-1])
    selected_mask = selected_mask.reshape(-1, selected_mask.shape[-1])
    if min(influence_scores.shape) < 1:
        raise ValueError("influence tensors must be nonempty")
    if not torch.isfinite(influence_scores).all() or not torch.isfinite(measured_influence).all():
        raise ValueError("influence measurements must be finite")
    if topk < 1:
        raise ValueError("topk must be positive")
    k = min(topk, influence_scores.shape[-1])
    predicted_top = influence_scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
    measured_top = measured_influence.argsort(dim=-1, descending=True, stable=True)[:, :k]
    fixed_overlap = predicted_top[:, :, None].eq(measured_top[:, None, :]).any(dim=-1).double().mean(-1)
    correlations = [spearman_correlation(p, m) for p, m in zip(influence_scores, measured_influence)]
    adaptive_precision = []
    for measured, selected in zip(measured_influence, selected_mask.bool()):
        count = int(selected.sum())
        if count == 0:
            adaptive_precision.append(measured.new_tensor(0.0))
            continue
        best = measured.argsort(descending=True, stable=True)[:count]
        adaptive_precision.append(selected[best].float().mean())
    result = {
        "influence_spearman": float(torch.stack(correlations).mean()),
        "influence_topk_precision": float(fixed_overlap.mean()),
        "influence_topk_recall": float(fixed_overlap.mean()),
        "influence_diagnostic_k": float(k),
        "selected_influence_precision": float(torch.stack(adaptive_precision).mean()),
        "influence_state_count": float(influence_scores.shape[0]),
    }
    if k == 12:
        result["influence_top12_recall"] = result["influence_topk_recall"]
    return result


def action_consistency_metrics(
    full_actions: Tensor,
    sparse_actions: Tensor,
    *,
    tolerance: float = 0.05,
) -> Dict[str, float]:
    if full_actions.shape != sparse_actions.shape:
        raise ValueError("full and sparse actions must have matching shapes")
    if full_actions.ndim < 2:
        raise ValueError("full and sparse actions must include a sample dimension")
    _require_finite_nonempty("full_actions", full_actions)
    _require_finite_nonempty("sparse_actions", sparse_actions)
    if tolerance < 0:
        raise ValueError("action tolerance must be non-negative")
    error = (full_actions.float() - sparse_actions.float()).flatten(1)
    per_sample_rmse = error.square().mean(dim=-1).sqrt()
    return {
        "topk_action_rmse": float(per_sample_rmse.mean()),
        "topk_action_consistency": float(
            (per_sample_rmse <= tolerance).float().mean()
        ),
    }


def regret_geometry_metrics(
    latent_states: Tensor,
    candidate_q_values: Tensor,
) -> Dict[str, float]:
    """Rank alignment and nearest-neighbor optimal-action retrieval."""
    if latent_states.ndim != 2 or candidate_q_values.ndim != 2:
        raise ValueError("latents and candidate Q values must be matrices")
    if latent_states.shape[0] != candidate_q_values.shape[0]:
        raise ValueError("latents and candidate Q values must align")
    if latent_states.shape[0] < 2:
        raise ValueError("at least two states are required")
    if not torch.isfinite(latent_states).all() or not torch.isfinite(candidate_q_values).all():
        raise ValueError("regret diagnostics require finite latents and measured returns")
    normalized = F.normalize(latent_states.double(), dim=-1)
    latent_distance = 1.0 - normalized @ normalized.transpose(0, 1)
    q = candidate_q_values.double()
    regret = q.amax(dim=-1, keepdim=True) - q
    regret_distance = torch.cdist(regret, regret, p=1)
    upper = torch.triu(
        torch.ones_like(latent_distance, dtype=torch.bool), diagonal=1
    )
    correlation = spearman_correlation(
        latent_distance[upper], regret_distance[upper]
    )
    nearest_distance = latent_distance.masked_fill(
        torch.eye(
            latent_distance.shape[0],
            device=latent_distance.device,
            dtype=torch.bool,
        ),
        torch.inf,
    )
    nearest = nearest_distance.argmin(dim=-1)
    optimal = candidate_q_values.argmax(dim=-1)
    retrieval = optimal.eq(optimal[nearest]).float().mean()
    return {
        "regret_distance_spearman": float(correlation),
        "optimal_action_retrieval_accuracy": float(retrieval),
        "regret_pair_count": float(upper.sum()),
        "regret_nonconstant_fraction": float((q.amax(-1) != q.amin(-1)).double().mean()),
        "regret_best_rule_tie_fraction": float((q.eq(q.amax(-1, keepdim=True)).sum(-1) > 1).double().mean()),
        "regret_distance_above_2_fraction": float((regret_distance[upper] > 2).double().mean()),
        "regret_geometry_absolute_error": float((latent_distance[upper] - regret_distance[upper]).abs().mean()),
    }


def efficiency_metrics(
    selected_mask: Tensor,
    *,
    latency_ms: Optional[Tensor] = None,
    peak_memory_mb: Optional[Tensor] = None,
    success: Optional[Tensor] = None,
) -> Dict[str, float | None]:
    if selected_mask.ndim < 2:
        raise ValueError("selected_mask must have [states, candidates] dimensions")
    raw_selected = selected_mask.reshape(-1, selected_mask.shape[-1])
    _require_binary("selected_mask", raw_selected.float())
    selected = raw_selected.bool()
    if min(selected.shape) < 1:
        raise ValueError("selected_mask must be nonempty")
    counts = selected.sum(dim=-1).float()
    result = {
        "expanded_tokens_mean": float(counts.mean()),
        "expanded_token_ratio": float(selected.float().mean()),
    }
    if success is not None:
        success = _require_binary(
            "success", success.float().reshape(-1)
        )
        if success.shape[0] != counts.shape[0]:
            raise ValueError("success must have one value per sample")
        successful = success > 0
        result["tokens_per_success"] = (
            float(counts[successful].mean())
            if successful.any()
            else None
        )
        result["successful_state_count"] = float(successful.sum())
    if latency_ms is not None:
        latency_ms = _require_finite_nonempty(
            "latency_ms", latency_ms.float().flatten()
        )
        if not (latency_ms > 0).all():
            raise ValueError("latency_ms must be positive")
        result["latency_ms_mean"] = float(latency_ms.mean())
        result["latency_ms_p95"] = float(
            torch.quantile(latency_ms, 0.95)
        )
        result["action_hz"] = float(
            1000.0 / latency_ms.mean().clamp_min(1e-6)
        )
    if peak_memory_mb is not None:
        peak_memory_mb = _require_finite_nonempty(
            "peak_memory_mb", peak_memory_mb.float().flatten()
        )
        if not (peak_memory_mb >= 0).all():
            raise ValueError("peak_memory_mb must be non-negative")
        result["peak_memory_mb"] = float(
            peak_memory_mb.max()
        )
    return result


def task_and_counterfactual_metrics(
    *,
    success: Optional[Tensor] = None,
    subgoal_completion: Optional[Tensor] = None,
    task_progress: Optional[Tensor] = None,
    standard_success: Optional[Tensor] = None,
    ood_success: Optional[Tensor] = None,
    baseline_return: Optional[Tensor] = None,
    high_intervention_return: Optional[Tensor] = None,
    counterfactual_prediction: Optional[Tensor] = None,
    counterfactual_target: Optional[Tensor] = None,
) -> Dict[str, float]:
    result = {}
    if success is not None:
        success = _require_binary("success", success.float().flatten())
        result["task_success_rate"] = float(success.mean())
        result["episode_success_sem"] = float(
            success.std(unbiased=False)
            / max(success.numel() ** 0.5, 1.0)
        )
    if subgoal_completion is not None:
        subgoal_completion = _require_unit_interval(
            "subgoal_completion", subgoal_completion.float()
        )
        result["subgoal_completion_rate"] = float(
            subgoal_completion.float().mean()
        )
    if task_progress is not None:
        task_progress = _require_unit_interval(
            "task_progress", task_progress.float()
        )
        result["average_task_progress"] = float(
            task_progress.float().mean()
        )
    if (standard_success is None) != (ood_success is None):
        raise ValueError(
            "standard_success and ood_success must be provided together"
        )
    if standard_success is not None:
        standard_success = _require_binary(
            "standard_success", standard_success.float()
        )
        ood_success = _require_binary("ood_success", ood_success.float())
        standard_rate = standard_success.mean()
        ood_rate = ood_success.mean()
        result["standard_success_rate"] = float(standard_rate)
        result["ood_success_rate"] = float(ood_rate)
    if (baseline_return is None) != (high_intervention_return is None):
        raise ValueError(
            "baseline_return and high_intervention_return must be paired"
        )
    if baseline_return is not None:
        if baseline_return.shape != high_intervention_return.shape:
            raise ValueError(
                "baseline and high-intervention returns must have matching shapes"
            )
        baseline_return = _require_finite_nonempty(
            "baseline_return", baseline_return.float()
        )
        high_intervention_return = _require_finite_nonempty(
            "high_intervention_return", high_intervention_return.float()
        )
        result["high_influence_intervention_drop"] = float(
            (
                baseline_return
                - high_intervention_return
            ).mean()
        )
    if (counterfactual_prediction is None) != (
        counterfactual_target is None
    ):
        raise ValueError(
            "counterfactual prediction and target must be paired"
        )
    if counterfactual_prediction is not None:
        if counterfactual_prediction.shape != counterfactual_target.shape:
            raise ValueError(
                "counterfactual prediction and target shapes must match"
            )
        if counterfactual_target.numel() == 0:
            raise ValueError("counterfactual decision labels cannot be empty")
        for labels in (counterfactual_prediction, counterfactual_target):
            if labels.is_floating_point() and (
                not torch.isfinite(labels).all()
                or not labels.eq(labels.round()).all()
            ):
                raise ValueError("counterfactual accuracy requires discrete labels, not continuous action vectors")
        result["counterfactual_decision_count"] = float(counterfactual_target.numel())
        result["counterfactual_decision_accuracy"] = float(
            counterfactual_prediction.eq(counterfactual_target).float().mean()
        )
    return result


def compute_diwa_metrics(
    *,
    influence_scores: Tensor,
    measured_influence: Tensor,
    selected_mask: Tensor,
    full_actions: Optional[Tensor] = None,
    sparse_actions: Optional[Tensor] = None,
    latent_states: Optional[Tensor] = None,
    candidate_q_values: Optional[Tensor] = None,
    latency_ms: Optional[Tensor] = None,
    peak_memory_mb: Optional[Tensor] = None,
    success: Optional[Tensor] = None,
    subgoal_completion: Optional[Tensor] = None,
    task_progress: Optional[Tensor] = None,
    standard_success: Optional[Tensor] = None,
    ood_success: Optional[Tensor] = None,
    baseline_return: Optional[Tensor] = None,
    high_intervention_return: Optional[Tensor] = None,
    counterfactual_prediction: Optional[Tensor] = None,
    counterfactual_target: Optional[Tensor] = None,
    action_tolerance: float = 0.05,
) -> Dict[str, float | None]:
    result = influence_identification_metrics(
        influence_scores, measured_influence, selected_mask
    )
    result.update(
        efficiency_metrics(
            selected_mask,
            latency_ms=latency_ms,
            peak_memory_mb=peak_memory_mb,
            success=success,
        )
    )
    if (full_actions is None) != (sparse_actions is None):
        raise ValueError(
            "full_actions and sparse_actions must be provided together"
        )
    if full_actions is not None:
        result.update(
            action_consistency_metrics(
                full_actions,
                sparse_actions,
                tolerance=action_tolerance,
            )
        )
    if (latent_states is None) != (candidate_q_values is None):
        raise ValueError(
            "latent_states and candidate_q_values must be provided together"
        )
    if latent_states is not None:
        result.update(
            regret_geometry_metrics(latent_states, candidate_q_values)
        )
    result.update(
        task_and_counterfactual_metrics(
            success=success,
            subgoal_completion=subgoal_completion,
            task_progress=task_progress,
            standard_success=standard_success,
            ood_success=ood_success,
            baseline_return=baseline_return,
            high_intervention_return=high_intervention_return,
            counterfactual_prediction=counterfactual_prediction,
            counterfactual_target=counterfactual_target,
        )
    )
    return result


def training_seed_statistics(scores: Tensor) -> Dict[str, float]:
    """Unrounded per-training-seed rates, in [0,1]; sample SD and SEM."""
    scores = scores.double()
    if scores.ndim != 1 or scores.numel() < 2 or not torch.isfinite(scores).all():
        raise ValueError("provide at least two finite per-training-seed success rates")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("seed success rates must be fractions in [0,1]")
    sd = scores.std(unbiased=True)
    return {"seed_success_mean": float(scores.mean()), "seed_success_sd": float(sd),
            "seed_success_sem": float(sd / scores.numel() ** 0.5), "training_seed_count": float(scores.numel())}


def physical_task_wilson_intervals(successes: Tensor, trials: Tensor) -> Tensor:
    """95% intervals for each task separately, conditional on its checkpoint."""
    successes, trials = successes.double(), trials.double()
    if successes.shape != trials.shape or successes.ndim != 1 or successes.numel() == 0:
        raise ValueError("successes and trials must be matching task-count vectors")
    if not (torch.isfinite(successes).all() and torch.isfinite(trials).all()):
        raise ValueError("physical trial counts must be finite")
    if not ((trials > 0) & (successes >= 0) & (successes <= trials) & successes.eq(successes.round()) & trials.eq(trials.round())).all():
        raise ValueError("counts must satisfy 0 <= successes <= positive trials")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z*z / trials
    center = (p + z*z / (2*trials)) / denominator
    radius = z * (p*(1-p)/trials + z*z/(4*trials*trials)).sqrt() / denominator
    return torch.stack((center-radius, center+radius), dim=-1)
