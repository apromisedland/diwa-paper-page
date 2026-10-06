"""Validate and package measured DIWA supervision into per-step sidecars.

Input files must be named ``<episode_id>.npz`` and contain aligned reward,
termination, progress, candidate-action, measured candidate-Q and stable
candidate-rule identity arrays.
Values must come from simulator rollouts, task predicates, human annotations
or a separately audited reward pipeline. This utility deliberately does not
synthesize temporal proxies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.diwa_schema import validate_candidate_rule_ids  # noqa: E402


REQUIRED_KEYS = (
    "reward",
    "done",
    "progress",
    "candidate_actions",
    "candidate_q_values",
    "candidate_rule_ids",
)

TEMPORAL_KEYS = REQUIRED_KEYS[:-1]


def validate_episode(source: Path) -> dict[str, np.ndarray]:
    with np.load(source) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{source} is missing {missing}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    if any(
        array.ndim == 0 or array.shape[0] == 0
        for array in (arrays[key] for key in TEMPORAL_KEYS)
    ):
        raise ValueError(f"{source} supervision arrays must have a nonempty time axis")
    lengths = {arrays[key].shape[0] for key in TEMPORAL_KEYS}
    if len(lengths) != 1:
        raise ValueError(f"{source} has inconsistent supervision lengths")
    if arrays["reward"].ndim != 1 or arrays["done"].ndim != 1:
        raise ValueError(f"{source} reward/done arrays must be one-dimensional")
    if arrays["progress"].ndim != 1:
        raise ValueError(f"{source} progress array must be one-dimensional")
    if arrays["candidate_actions"].ndim != 4:
        raise ValueError(
            f"{source} candidate_actions must be [time, candidates, action_steps, 7]"
        )
    if arrays["candidate_actions"].shape[-1] != 7:
        raise ValueError(f"{source} candidate action dimension must be seven")
    if arrays["candidate_q_values"].ndim != 2:
        raise ValueError(
            f"{source} candidate_q_values must be [time, candidates]"
        )
    if (
        arrays["candidate_actions"].shape[:2]
        != arrays["candidate_q_values"].shape
    ):
        raise ValueError(f"{source} candidate action/Q dimensions do not match")
    validate_candidate_rule_ids(
        arrays["candidate_rule_ids"],
        candidate_count=arrays["candidate_actions"].shape[1],
    )
    if not np.isfinite(arrays["reward"]).all():
        raise ValueError(f"{source} contains non-finite rewards")
    if not np.isin(arrays["done"], (0, 1)).all():
        raise ValueError(f"{source} done must be binary")
    if not np.isfinite(arrays["progress"]).all():
        raise ValueError(f"{source} contains non-finite progress")
    if not np.isfinite(arrays["candidate_q_values"]).all():
        raise ValueError(f"{source} contains non-finite candidate Q values")
    if not np.isfinite(arrays["candidate_actions"]).all():
        raise ValueError(f"{source} contains non-finite candidate actions")
    if ((arrays["progress"] < 0) | (arrays["progress"] > 1)).any():
        raise ValueError(f"{source} progress must be in [0, 1]")
    return arrays


def package_episode(source: Path, output_root: Path) -> int:
    episode_id = source.stem
    arrays = validate_episode(source)
    output_steps = output_root / episode_id / "steps"
    output_steps.mkdir(parents=True, exist_ok=True)
    for step_index in range(arrays["reward"].shape[0]):
        output_path = output_steps / f"{step_index:04d}.npz"
        np.savez_compressed(
            output_path,
            reward=np.float32(arrays["reward"][step_index]),
            done=np.bool_(arrays["done"][step_index]),
            progress=np.float32(arrays["progress"][step_index]),
            candidate_actions=np.asarray(
                arrays["candidate_actions"][step_index], dtype=np.float32
            ),
            candidate_q_values=np.asarray(
                arrays["candidate_q_values"][step_index], dtype=np.float32
            ),
            candidate_rule_ids=np.asarray(arrays["candidate_rule_ids"]),
        )
    return arrays["reward"].shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing one measured <episode_id>.npz per episode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="target diwa_supervision directory",
    )
    args = parser.parse_args()
    sources = sorted(args.source.glob("*.npz"))
    if not sources:
        raise FileNotFoundError(f"no episode NPZ files found in {args.source}")
    total_steps = sum(package_episode(source, args.output) for source in sources)
    print(f"packaged {len(sources)} episodes and {total_steps} measured steps")


if __name__ == "__main__":
    main()
