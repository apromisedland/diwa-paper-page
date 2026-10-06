"""Strict episode features and same-task, cross-episode training batches."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


ARRAY_FIELDS = (
    "context_tokens",
    "visual_tokens",
    "actions",
    "reward",
    "done",
    "progress",
    "candidate_actions",
    "candidate_q_values",
)
POLICY_NAMES = {"reward": "rewards", "done": "dones"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_features(arrays, hidden_dim):
    for name in ("context_tokens", "visual_tokens"):
        value = arrays[name]
        if value.ndim != 3 or value.shape[-1] != hidden_dim or min(value.shape) < 1:
            raise ValueError(
                f"{name} must be [time, tokens, {hidden_dim}] with nonempty axes"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric features")
    if arrays["context_tokens"].shape[0] != arrays["visual_tokens"].shape[0]:
        raise ValueError("context and visual observation times must match")


def load_observations(path, hidden_dim):
    """Read label-free, unbatched observations for inference."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ("context_tokens", "visual_tokens")}
    _validate_features(arrays, hidden_dim)
    return {
        name: torch.as_tensor(value, dtype=torch.float32)[None]
        for name, value in arrays.items()
    }


class FeatureWindowDataset(Dataset):
    def __init__(self, manifest_path, config, *, allow_synthetic=False):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.config = config
        self.sequence_length = config["training"]["sequence_length"]
        self.window_length = self.sequence_length + config["core"]["horizon"]
        m = self.manifest
        if m.get("format_version") != 1:
            raise ValueError("manifest format_version must be 1")
        if m.get("data_kind") != "measured":
            if not (allow_synthetic and m.get("data_kind") == "synthetic_fixture"):
                raise ValueError(
                    "training requires data_kind=measured; synthetic fixtures are only for smoke tests"
                )
        if m.get("feature_causality") != "past_and_current_only":
            raise ValueError("feature_causality must declare past_and_current_only")
        for name in ("feature_encoder", "action_normalization", "return_definition"):
            if not isinstance(m.get(name), str) or not m[name].strip():
                raise ValueError(f"manifest requires a nonempty {name} description")
        rules = m.get("candidate_rules", [])
        if len(rules) != config["core"]["regret_candidates"] or len(set(rules)) != len(
            rules
        ):
            raise ValueError(
                "candidate_rules must give unique, aligned names for every candidate"
            )
        self.paths = [
            (self.manifest_path.parent / path).resolve()
            for path in m.get("episodes", [])
        ]
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("episodes must be a nonempty list without repeated files")
        self.windows = []
        self.windows_by_episode = {}
        self.episodes_by_task = {}
        self.task_ids = []
        task_map, episode_ids = {}, set()
        digest = hashlib.sha256(json.dumps(m, sort_keys=True).encode())
        digest.update(b"diwa-feature-windows-v2-terminal-lookahead-padding")
        feature_shape = None
        for episode_index, path in enumerate(self.paths):
            try:
                arrays, task, episode = self._read_and_validate(path)
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path.name}: {error}") from error
            if episode in episode_ids:
                raise ValueError(f"episode_id must be unique across files: {episode}")
            episode_ids.add(episode)
            shape = (
                arrays["context_tokens"].shape[1:],
                arrays["visual_tokens"].shape[1:],
            )
            if feature_shape is not None and shape != feature_shape:
                raise ValueError(
                    "all episodes must have the same feature token counts and width"
                )
            feature_shape = shape
            task_id = task_map.setdefault(task, len(task_map))
            self.task_ids.append(task_id)
            self.episodes_by_task.setdefault(task_id, []).append(episode_index)
            indices = []
            stop = arrays["actions"].shape[0] - self.sequence_length + 1
            stride = config["training"]["window_stride"]
            if stride < 1 or stop < 1:
                raise ValueError(
                    f"{path.name}: needs at least {self.sequence_length} "
                    "observations and positive stride"
                )
            starts = list(range(0, stop, stride))
            final_start = stop - 1
            if starts[-1] != final_start:
                starts.append(final_start)
            for start in starts:
                indices.append(len(self.windows))
                self.windows.append((episode_index, start))
            self.windows_by_episode[episode_index] = indices
            digest.update(file_sha256(path).encode())
        for task, episodes in self.episodes_by_task.items():
            if len(episodes) < 2:
                raise ValueError(
                    f"task {task} needs at least two distinct episodes for regret/intervention pairs"
                )
        self.fingerprint = digest.hexdigest()

    def _read_and_validate(self, path):
        with np.load(path, allow_pickle=False) as data:
            required = (*ARRAY_FIELDS, "task_id", "episode_id", "candidate_rule_ids")
            missing = set(required).difference(data.files)
            if missing:
                raise ValueError(f"missing {sorted(missing)}")
            arrays = {name: np.asarray(data[name]) for name in ARRAY_FIELDS}
            if data["task_id"].ndim or data["episode_id"].ndim:
                raise ValueError("task_id and episode_id must be scalars")
            task, episode = str(data["task_id"].item()), str(data["episode_id"].item())
            if data["candidate_rule_ids"].tolist() != self.manifest["candidate_rules"]:
                raise ValueError("candidate_rule_ids order differs from the manifest")
        c = self.config["core"]
        _validate_features(arrays, c["hidden_dim"])
        length = arrays["context_tokens"].shape[0]
        expected = {
            "actions": (length, c["action_pred_steps"], c["action_dim"]),
            "candidate_actions": (
                length,
                c["regret_candidates"],
                c["action_pred_steps"],
                c["action_dim"],
            ),
            "candidate_q_values": (length, c["regret_candidates"]),
            **{name: (length,) for name in ("reward", "done", "progress")},
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if not np.isin(arrays["done"], (0, 1)).all():
            raise ValueError("done must be binary")
        if arrays["done"][:-1].any():
            raise ValueError(
                "a file must contain one episode; termination is only allowed at its last observation"
            )
        if ((arrays["progress"] < 0) | (arrays["progress"] > 1)).any():
            raise ValueError("progress must be in [0, 1]")
        for name in ("actions", "candidate_actions"):
            continuous = arrays[name][..., : c["continuous_action_dim"]]
            gripper = arrays[name][..., c["continuous_action_dim"] :]
            if (np.abs(continuous) > 1 + 1e-6).any() or (
                (gripper < 0) | (gripper > 1)
            ).any():
                raise ValueError(
                    f"{name}: use continuous [-1,1] and gripper [0,1] normalization"
                )
        return arrays, task, episode

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        # NPZ is a simple interchange format. Large-scale runs should use the
        # image-dataset pipeline or a sharded feature reader with the same API.
        with np.load(self.paths[episode], allow_pickle=False) as data:
            sample = {}
            for name in ARRAY_FIELDS:
                length = (
                    self.window_length
                    if name.endswith("tokens")
                    else self.sequence_length
                )
                dtype = torch.bool if name == "done" else torch.float32
                values = data[name][start : start + length].copy()
                if name.endswith("tokens") and len(values) < length:
                    values = np.concatenate(
                        (values, np.repeat(values[-1:], length - len(values), axis=0)),
                        axis=0,
                    )
                sample[POLICY_NAMES.get(name, name)] = torch.as_tensor(
                    values, dtype=dtype
                )
            available_observations = min(
                self.window_length,
                data["visual_tokens"].shape[0] - start,
            )
        sample["observation_valid"] = torch.arange(self.window_length) < (
            available_observations
        )
        sample.update(
            task_ids=torch.tensor(self.task_ids[episode]),
            episode_ids=torch.tensor(episode),
        )
        return sample


class PairedBatchSampler(Sampler):
    """Every anchor is accompanied by a same-task, different-episode window.

    Each anchor is visited once per epoch; the last batch repeats anchors
    when necessary to keep all batch sizes equal. Pair sampling is seeded.
    """

    def __init__(self, dataset, batch_size, seed, epoch=0):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired batches must have even size >= 2")
        self.dataset, self.batch_size, self.seed, self.epoch = (
            dataset,
            batch_size,
            seed,
            epoch,
        )

    def __len__(self):
        return math.ceil(len(self.dataset) / (self.batch_size // 2))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        anchors = rng.permutation(len(self.dataset)).tolist()
        half = self.batch_size // 2
        total = len(self) * half
        anchors += [anchors[i % len(anchors)] for i in range(total - len(anchors))]
        for offset in range(0, len(anchors), half):
            batch = []
            for anchor in anchors[offset : offset + half]:
                episode, _ = self.dataset.windows[anchor]
                task = self.dataset.task_ids[episode]
                partners = [
                    e for e in self.dataset.episodes_by_task[task] if e != episode
                ]
                partner = int(rng.choice(partners))
                batch.extend(
                    (anchor, int(rng.choice(self.dataset.windows_by_episode[partner])))
                )
            yield batch
