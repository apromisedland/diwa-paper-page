"""Random tensors for code checks, never measured robot data or benchmarks."""

import json
from pathlib import Path

import numpy as np


def create_fixture(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    c, t = config["core"], config["training"]
    length = t["sequence_length"] + c["horizon"] + 2
    rules = [f"synthetic_rule_{index}" for index in range(c["regret_candidates"])]
    paths = []
    for episode in range(2):

        def actions(shape):
            value = rng.uniform(-1, 1, shape).astype(np.float32)
            value[..., c["continuous_action_dim"] :] = rng.integers(
                0, 2, value[..., c["continuous_action_dim"] :].shape
            )
            return value

        arrays = {
            "context_tokens": rng.normal(size=(length, 3, c["hidden_dim"])).astype(
                np.float32
            ),
            "visual_tokens": rng.normal(size=(length, 8, c["hidden_dim"])).astype(
                np.float32
            ),
            "actions": actions((length, c["action_pred_steps"], c["action_dim"])),
            "reward": rng.normal(size=length).astype(np.float32),
            "done": np.arange(length) == length - 1,
            "progress": rng.uniform(0, 1, length).astype(np.float32),
            "candidate_actions": actions(
                (
                    length,
                    c["regret_candidates"],
                    c["action_pred_steps"],
                    c["action_dim"],
                )
            ),
            "candidate_q_values": rng.uniform(
                -0.2, 0.2, (length, c["regret_candidates"])
            ).astype(np.float32),
            "task_id": np.asarray("synthetic_task"),
            "episode_id": np.asarray(f"synthetic_episode_{episode}"),
            "candidate_rule_ids": np.asarray(rules),
        }
        name = f"synthetic_episode_{episode}.npz"
        np.savez_compressed(directory / name, **arrays)
        paths.append(name)
        if episode == 0:
            # The inference example contains observed history only.
            np.savez_compressed(
                directory / "observations.npz",
                **{
                    name: arrays[name][: t["sequence_length"]]
                    for name in ("context_tokens", "visual_tokens")
                },
            )
    manifest = {
        "format_version": 1,
        "data_kind": "synthetic_fixture",
        "feature_encoder": "independent random tensor fixture; no pretrained encoder",
        "feature_causality": "past_and_current_only",
        "action_normalization": "continuous [-1,1]; gripper [0,1]",
        "return_definition": "random numbers for functional tests only; not measured returns",
        "candidate_rules": rules,
        "episodes": paths,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, directory / "observations.npz"
