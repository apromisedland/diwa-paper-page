"""Compute the paper's offline DIWA quality and efficiency metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diwa.metrics import compute_diwa_metrics, training_seed_statistics, physical_task_wilson_intervals


REQUIRED_KEYS = (
    "influence_scores",
    "measured_influence",
    "selected_mask",
)
OPTIONAL_KEYS = (
    "full_actions",
    "sparse_actions",
    "latent_states",
    "candidate_q_values",
    "latency_ms",
    "peak_memory_mb",
    "success",
    "subgoal_completion",
    "task_progress",
    "standard_success",
    "ood_success",
    "baseline_return",
    "high_intervention_return",
    "counterfactual_prediction",
    "counterfactual_target",
)


def evaluate_archive(
    archive_path: Path,
    *,
    action_tolerance: float = 0.05,
) -> dict[str, float]:
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{archive_path} is missing {missing}")
        tensors = {
            key: torch.from_numpy(np.asarray(archive[key]))
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
            if key in archive
        }
        seeds = torch.from_numpy(archive["training_seed_success_rates"]) if "training_seed_success_rates" in archive else None
        physical = None
        if ("physical_task_successes" in archive) != ("physical_task_trials" in archive):
            raise ValueError("physical successes and trials must be provided together")
        if "physical_task_successes" in archive:
            physical = physical_task_wilson_intervals(
                torch.from_numpy(archive["physical_task_successes"]),
                torch.from_numpy(archive["physical_task_trials"]),
            ).tolist()
    metrics = compute_diwa_metrics(
        **tensors,
        action_tolerance=action_tolerance,
    )
    if seeds is not None:
        metrics.update(training_seed_statistics(seeds))
    if physical is not None:
        metrics["physical_task_wilson_95"] = physical
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ archive containing DIWA predictions and measurements",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--action_tolerance",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    metrics = evaluate_archive(
        args.input,
        action_tolerance=args.action_tolerance,
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
