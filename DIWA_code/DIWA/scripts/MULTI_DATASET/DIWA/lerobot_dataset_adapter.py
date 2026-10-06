"""Export tiny real DROID/OXE samples from public LeRobot mirrors.

The DreamVLA repository has native loaders for DROID and a fixed set of 12
Open X-Embodiment (OXE) datasets.  Their full releases are very large, so the
smoke protocol materializes only the first packed parquet files.  Camera
frames are decoded directly from the public packed videos with HTTP range
requests through FFmpeg; the video archives do not need to be downloaded in
full.

These are offline datasets rather than simulators.  Evaluation therefore
checks finite policy inference against the exported action contract; it does
not claim an environment rollout or task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np


ACTION_DIM = 7
CONTINUOUS_ACTION_DIM = 6
STATE_ARM_DIM = 6
CANDIDATE_RULE_IDS = (
    "demonstration",
    "invert_gripper",
    "invert_continuous_action",
    "suppress_translation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download")
    download.add_argument("--repo", required=True)
    download.add_argument("--output-dir", required=True, type=Path)
    download.add_argument("--endpoint", default="https://huggingface.co")

    export = commands.add_parser("export")
    export.add_argument("--repo", required=True)
    export.add_argument("--dataset-name", required=True)
    export.add_argument("--dataset-dir", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--endpoint", default="https://huggingface.co")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--export", required=True, type=Path)
    evaluate.add_argument("--action", required=True, type=Path)
    evaluate.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def resolve_url(endpoint: str, repo: str, path: str) -> str:
    encoded = "/".join(quote(part) for part in path.split("/"))
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/main/{encoded}?download=true"


def _download_file(url: str, output: Path) -> None:
    import requests

    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary.replace(output)


def materialize(repo: str, output_dir: Path, endpoint: str) -> dict:
    info_path = output_dir / "meta/info.json"
    _download_file(resolve_url(endpoint, repo, "meta/info.json"), info_path)
    info = json.loads(info_path.read_text())
    data_path = info["data_path"].format(chunk_index=0, file_index=0)
    paths = (
        data_path,
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    )
    for path in paths:
        _download_file(resolve_url(endpoint, repo, path), output_dir / path)
    return info


def select_camera_keys(info: dict) -> tuple[str, str]:
    video_keys = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError("dataset has no video features")
    wrist_markers = ("wrist", "hand", "gripper", "eye_in_hand")
    wrist = next(
        (key for key in video_keys if any(marker in key.lower() for marker in wrist_markers)),
        video_keys[-1],
    )
    primary = next((key for key in video_keys if key != wrist), video_keys[0])
    return primary, wrist


def canonical_action(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}-D action, got {actions.shape[-1]}")
    result = actions.copy()
    result[..., :CONTINUOUS_ACTION_DIM] = np.clip(
        result[..., :CONTINUOUS_ACTION_DIM], -1.0, 1.0
    )
    result[..., -1] = (result[..., -1] > 0.0).astype(np.float32)
    return result


def canonical_state(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.shape[-1] < STATE_ARM_DIM + 1:
        raise ValueError("state must contain six arm and one gripper coordinate")
    return np.concatenate((states[..., :STATE_ARM_DIM], states[..., -1:]), axis=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_value(metadata: dict, key: str, row: int):
    if key not in metadata:
        raise KeyError(f"episode metadata is missing {key}")
    return metadata[key][row]


def _video_url(
    info: dict,
    endpoint: str,
    repo: str,
    camera: str,
    file_index: int,
) -> str:
    path = info["video_path"].format(
        video_key=camera,
        chunk_index=file_index // 1000,
        file_index=file_index,
    )
    return resolve_url(endpoint, repo, path)


def read_remote_frames(
    url: str,
    start_seconds: float,
    count: int,
    height: int,
    width: int,
) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        url,
        "-frames:v",
        str(count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    expected = count * height * width * 3
    if len(raw) != expected:
        raise RuntimeError(f"decoded {len(raw)} bytes, expected {expected} from {url}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)


def _task_map(tasks_path: Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(tasks_path).to_pydict()
    indices = table.get("task_index", range(len(next(iter(table.values())))))
    names = table.get("task", table.get("language_instruction"))
    if names is None:
        names = next(
            (
                values
                for key, values in table.items()
                if key not in {"task_index", "index"}
            ),
            [],
        )
    return {int(index): str(name) for index, name in zip(indices, names)}


def export_batch(
    repo: str,
    dataset_name: str,
    dataset_dir: Path,
    output: Path,
    endpoint: str,
) -> None:
    import pyarrow.parquet as pq

    info = json.loads((dataset_dir / "meta/info.json").read_text())
    data_path = dataset_dir / info["data_path"].format(chunk_index=0, file_index=0)
    table = pq.read_table(data_path).to_pydict()
    episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"], dtype=np.int64)
    starts = np.flatnonzero(frame_indices == 0)
    selected = []
    for start in starts:
        episode = int(episode_indices[start])
        rows = np.flatnonzero(episode_indices == episode)[:3]
        has_droid_language = True
        if repo == "lerobot/droid_1.0.1":
            has_droid_language = any(
                str(table[key][start]).strip()
                for key in (
                    "language_instruction",
                    "language_instruction_2",
                    "language_instruction_3",
                )
                if key in table
            )
        if (
            len(rows) == 3
            and np.array_equal(rows, np.arange(start, start + 3))
            and has_droid_language
        ):
            selected.append((episode, rows))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("first parquet file does not contain two 3-frame episodes")

    primary_key, wrist_key = select_camera_keys(info)
    episode_metadata = pq.read_table(
        dataset_dir / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pydict()
    metadata_rows = {
        int(value): index
        for index, value in enumerate(episode_metadata["episode_index"])
    }
    tasks = _task_map(dataset_dir / "meta/tasks.parquet")

    primary, wrist, states, chunks, candidates, rewards, dones = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    task_indices, languages = [], []
    for episode, rows in selected:
        if {
            "action.cartesian_velocity",
            "action.gripper_position",
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
        }.issubset(table):
            # The official DROID LeRobot release exposes an 8-D joint-position
            # convenience feature, while DreamVLA's DROID loader is Cartesian:
            # six wrist controls plus one gripper coordinate.
            action_arm = np.asarray(
                [table["action.cartesian_velocity"][row] for row in rows],
                dtype=np.float32,
            )
            action_gripper = np.asarray(
                [table["action.gripper_position"][row] for row in rows],
                dtype=np.float32,
            ).reshape(-1, 1)
            state_arm = np.asarray(
                [table["observation.state.cartesian_position"][row] for row in rows],
                dtype=np.float32,
            )
            state_gripper = np.asarray(
                [table["observation.state.gripper_position"][row] for row in rows],
                dtype=np.float32,
            ).reshape(-1, 1)
            native_actions = canonical_action(
                np.concatenate((action_arm, action_gripper), axis=-1)
            )
            state = canonical_state(
                np.concatenate((state_arm, state_gripper), axis=-1)
            )
        else:
            native_actions = canonical_action(
                np.asarray([table["action"][row] for row in rows], dtype=np.float32)
            )
            state = canonical_state(
                np.asarray(
                    [table["observation.state"][row] for row in rows],
                    dtype=np.float32,
                )
            )
        chunk = np.stack((native_actions[:2], native_actions[1:3]))
        alternatives = np.repeat(chunk[:, None], 4, axis=1)
        alternatives[:, 1, :, -1] = 1.0 - alternatives[:, 1, :, -1]
        alternatives[:, 2, :, :CONTINUOUS_ACTION_DIM] *= -1.0
        alternatives[:, 3, :, :3] = 0.0

        reward_key = "next.reward" if "next.reward" in table else "reward"
        if reward_key in table:
            episode_rewards = np.asarray(
                [table[reward_key][row] for row in rows[:2]], dtype=np.float32
            )
        else:
            episode_rewards = np.zeros(2, dtype=np.float32)
        done_key = "next.done" if "next.done" in table else "is_last"
        if done_key in table:
            episode_dones = np.asarray(
                [table[done_key][row] for row in rows[:2]], dtype=bool
            )
        else:
            episode_dones = np.zeros(2, dtype=bool)

        meta_row = metadata_rows[episode]
        camera_frames = []
        for camera in (primary_key, wrist_key):
            file_index = int(
                _metadata_value(
                    episode_metadata, f"videos/{camera}/file_index", meta_row
                )
            )
            start_seconds = float(
                _metadata_value(
                    episode_metadata, f"videos/{camera}/from_timestamp", meta_row
                )
            )
            shape = info["features"][camera]["shape"]
            camera_frames.append(
                read_remote_frames(
                    _video_url(info, endpoint, repo, camera, file_index),
                    start_seconds,
                    count=3,
                    height=int(shape[0]),
                    width=int(shape[1]),
                )
            )

        task_index = int(table.get("task_index", [0] * len(episode_indices))[rows[0]])
        task_indices.append(task_index)
        language = tasks.get(task_index, "")
        if not language.strip():
            for key in (
                "language_instruction", "language_instruction_2", "language_instruction_3"
            ):
                if key not in table:
                    continue
                value = table[key][int(rows[0])]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if str(value).strip():
                    language = str(value)
                    break
        if not language.strip():
            raise ValueError(f"episode {episode} has no task instruction")
        languages.append(language)
        primary.append(camera_frames[0])
        wrist.append(camera_frames[1])
        states.append(state)
        chunks.append(chunk)
        candidates.append(alternatives)
        rewards.append(episode_rewards)
        dones.append(episode_dones)

    language = languages[0]
    rewards_array = np.stack(rewards)
    dones_array = np.stack(dones)
    # No measured progress/counterfactual returns exist in this offline export.
    # Preserve absence explicitly; the smoke path disables decision losses.
    progress = np.full_like(rewards_array, np.nan)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        primary=np.stack(primary).astype(np.uint8),
        wrist=np.stack(wrist).astype(np.uint8),
        state=np.stack(states).astype(np.float32),
        actions=np.stack(chunks).astype(np.float32),
        rewards=rewards_array,
        dones=dones_array,
        progress=progress,
        candidate_actions=np.stack(candidates).astype(np.float32),
        candidate_q_values=np.full((2, 2, 4), np.nan, dtype=np.float32),
        candidate_rule_ids=np.asarray(CANDIDATE_RULE_IDS),
        dataset=np.asarray(dataset_name),
        task=np.asarray(language),
        language=np.asarray(language),
        languages=np.asarray(languages),
        task_ids=np.asarray([
            int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
            & 0x7FFFFFFFFFFFFFFF for text in languages
        ], dtype=np.int64),
        source_task_indices=np.asarray(task_indices, dtype=np.int64),
        continuous_action_dim=np.asarray(CONTINUOUS_ACTION_DIM, dtype=np.int64),
        state_arm_dim=np.asarray(STATE_ARM_DIM, dtype=np.int64),
        gripper_width=np.asarray(True),
        supervision_source=np.asarray("public_lerobot_conversion_of_official_dataset"),
        candidate_q_source=np.asarray(
            "unavailable_offline_counterfactuals_explicitly_masked"
        ),
        require_measured_candidate_q=np.asarray(False),
        progress_source=np.asarray("unavailable_offline_progress_explicitly_masked"),
        source_repo=np.asarray(repo),
        source_parquet_sha256=np.asarray(_sha256(data_path)),
        episode_indices=np.asarray([episode for episode, _ in selected]),
        primary_camera=np.asarray(primary_key),
        wrist_camera=np.asarray(wrist_key),
    )
    print(f"LEROBOT_DATASET_EXPORT_OK {dataset_name} {output}")


def evaluate_action(export_path: Path, action_path: Path, report_path: Path) -> None:
    batch = np.load(export_path, allow_pickle=False)
    prediction = np.load(action_path).astype(np.float32)
    target = batch["actions"][0, 0].astype(np.float32)
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"prediction shape {prediction.shape} does not match target {target.shape}"
        )
    finite = bool(np.isfinite(prediction).all())
    if not finite:
        raise RuntimeError("offline policy action is non-finite")
    split = int(batch["continuous_action_dim"])
    report = {
        "status": "ok",
        "dataset": str(batch["dataset"]),
        "evaluation_kind": "offline_dataset_action_inference",
        "source_repo": str(batch["source_repo"]),
        "source_parquet_sha256": str(batch["source_parquet_sha256"]),
        "episode_indices": batch["episode_indices"].tolist(),
        "action_shape": list(prediction.shape),
        "finite_action": finite,
        "continuous_l1": float(np.abs(prediction[..., :split] - target[..., :split]).mean()),
        "gripper_accuracy": float(
            (
                (prediction[..., split:] >= 0.5)
                == (target[..., split:] >= 0.5)
            ).mean()
        ),
        "primary_camera": str(batch["primary_camera"]),
        "wrist_camera": str(batch["wrist_camera"]),
        "note": "integration metric only; the two-episode smoke batch is not a benchmark split",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_OFFLINE_DATASET_EVAL_OK")


def main() -> None:
    args = parse_args()
    if args.command == "download":
        materialize(args.repo, args.output_dir, args.endpoint)
        print(f"LEROBOT_DATASET_DOWNLOAD_OK {args.repo} {args.output_dir}")
    elif args.command == "export":
        export_batch(
            args.repo,
            args.dataset_name,
            args.dataset_dir,
            args.output,
            args.endpoint,
        )
    else:
        evaluate_action(args.export, args.action, args.report)


if __name__ == "__main__":
    main()
