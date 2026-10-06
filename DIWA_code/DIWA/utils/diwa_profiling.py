"""Deterministic local and distributed summaries for DIWA profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch.distributed as dist


PROFILE_FIELDS = (
    "latency_ms",
    "peak_memory_mb",
    "expanded_tokens",
    "selected_ratio",
)


def _validated_profile(profile: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    missing = [name for name in PROFILE_FIELDS if name not in profile]
    if missing:
        raise ValueError(f"DIWA profile is missing {missing}")
    output = {}
    for name in PROFILE_FIELDS:
        values = np.asarray(profile[name], dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError(f"DIWA profile field {name} must be finite")
        output[name] = values.tolist()
    lengths = {len(values) for values in output.values()}
    if len(lengths) != 1:
        raise ValueError("DIWA profile fields must contain the same number of samples")
    return output


def summarize_profiles(
    profiles: Sequence[Mapping[str, Sequence[float]]],
    *,
    warmup_steps: int = 0,
) -> dict[str, Any]:
    """Merge rank-local samples after dropping warm-up steps on every rank."""
    if warmup_steps < 0:
        raise ValueError("profile warmup_steps must be non-negative")
    if not profiles:
        return {}
    validated = [_validated_profile(profile) for profile in profiles]
    rank_counts = [len(profile["latency_ms"]) for profile in validated]
    retained = [max(0, count - warmup_steps) for count in rank_counts]
    if not any(retained):
        return {
            "rank_count": len(validated),
            "rank_sample_counts": rank_counts,
            "warmup_steps_per_rank": warmup_steps,
            "profile_sample_count": 0,
        }
    merged = {
        name: np.concatenate(
            [
                np.asarray(profile[name][warmup_steps:], dtype=np.float64)
                for profile in validated
                if len(profile[name]) > warmup_steps
            ]
        )
        for name in PROFILE_FIELDS
    }
    latency = merged["latency_ms"]
    return {
        "rank_count": len(validated),
        "rank_sample_counts": rank_counts,
        "warmup_steps_per_rank": warmup_steps,
        "profile_sample_count": int(latency.size),
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_p95": float(np.quantile(latency, 0.95)),
        "action_hz": float(1000.0 / max(float(latency.mean()), 1e-6)),
        "peak_memory_mb": float(merged["peak_memory_mb"].max()),
        "expanded_tokens_mean": float(merged["expanded_tokens"].mean()),
        "expanded_token_ratio": float(merged["selected_ratio"].mean()),
    }


def gather_profile_summary(
    profile: Mapping[str, Sequence[float]],
    *,
    warmup_steps: int = 0,
) -> dict[str, Any] | None:
    """Gather rank-local profiles and return one summary on rank zero."""
    local = _validated_profile(profile)
    if not dist.is_available() or not dist.is_initialized():
        return summarize_profiles([local], warmup_steps=warmup_steps)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank != 0:
        return None
    return summarize_profiles(gathered, warmup_steps=warmup_steps)


def write_profile_report(
    path: str | Path,
    summary: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a standards-compliant profiling report atomically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": dict(summary), "metadata": dict(metadata or {})}
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(output)
