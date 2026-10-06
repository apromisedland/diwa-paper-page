"""Measure DIWA reward, progress and candidate-Q labels in LIBERO.

Each demonstration simulator state is restored before every candidate rollout,
so candidate Q values are simulator measurements and never predictions from
the DIWA critic. The resulting episode NPZ files can be passed directly to
``build_diwa_supervision.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from utils.diwa_schema import stable_candidate_rule_ids  # noqa: E402


SUITE_NAMES = {
    "libero_10": "LIBERO_10",
    "libero_90": "LIBERO_90",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=tuple(SUITE_NAMES), required=True
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="official LIBERO suite directory containing *_demo.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument(
        "--progress-reward",
        type=float,
        default=1.0,
        help="terminal task-predicate progress shaping coefficient",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    args = parser.parse_args()
    if args.action_pred_steps < 1:
        parser.error("--action-pred-steps must be positive")
    if args.num_candidates < 4:
        parser.error("--num-candidates must be at least four")
    if not 0.0 < args.discount <= 1.0:
        parser.error("--discount must be in (0, 1]")
    if args.perturbation <= 0.0:
        parser.error("--perturbation must be positive")
    return args


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"_demo$", "", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def task_for_file(task_suite, source: Path):
    source_name = normalize_name(source.stem)
    exact = []
    partial = []
    for task_index in range(task_suite.n_tasks):
        task = task_suite.get_task(task_index)
        task_name = normalize_name(task.name)
        if source_name == task_name:
            exact.append((task_index, task))
        elif source_name.endswith(task_name) or task_name.endswith(source_name):
            partial.append((task_index, task))
    matches = exact or partial
    if len(matches) != 1:
        names = [task_suite.get_task(i).name for i in range(task_suite.n_tasks)]
        raise ValueError(
            f"cannot uniquely map {source.name} to a suite task; "
            f"available task names: {names}"
        )
    return matches[0]


def measured_progress(env) -> float:
    for candidate in (env, getattr(env, "env", None)):
        check = getattr(candidate, "_check_success", None)
        if check is not None:
            return float(bool(check()))
    raise RuntimeError("LIBERO environment does not expose task predicates")


def expert_chunk(actions: np.ndarray, step: int, length: int) -> np.ndarray:
    chunk = actions[step : step + length]
    if len(chunk) == 0:
        raise ValueError("empty expert action chunk")
    if len(chunk) < length:
        chunk = np.concatenate(
            (chunk, np.repeat(chunk[-1:], length - len(chunk), axis=0)),
            axis=0,
        )
    return np.asarray(chunk, dtype=np.float32)


def candidate_bank(
    expert: np.ndarray, count: int, perturbation: float
) -> np.ndarray:
    """Create stable expert/no-op/axis-perturbation candidate policies."""
    action_steps = expert.shape[0]
    candidates = np.zeros((count, action_steps, 7), dtype=np.float32)
    candidates[0] = expert
    candidates[1, :, 6] = expert[:, 6]
    for candidate_index in range(2, count):
        rule_index = candidate_index - 2
        axis = (rule_index // 2) % 6
        sign = 1.0 if rule_index % 2 == 0 else -1.0
        scale = 1.0 + (rule_index // 12)
        candidates[candidate_index] = expert
        candidates[candidate_index, :, axis] = np.clip(
            candidates[candidate_index, :, axis]
            + sign * scale * perturbation,
            -1.0,
            1.0,
        )
    candidates[..., 6] = np.where(candidates[..., 6] >= 0.0, 1.0, -1.0)
    return candidates


def measure_rollouts(
    env,
    state: np.ndarray,
    candidates: np.ndarray,
    discount: float,
    progress_reward: float,
) -> tuple[np.ndarray, float, bool, float]:
    q_values = np.zeros(candidates.shape[0], dtype=np.float32)
    demonstrated_reward = 0.0
    demonstrated_done = False
    progress = None
    for candidate_index, candidate in enumerate(candidates):
        # Reset non-simulator episode state (elapsed steps, done flags, etc.)
        # before restoring the same serialized MuJoCo state for every branch.
        env.reset()
        env.set_init_state(state)
        start_progress = measured_progress(env)
        if candidate_index == 0:
            progress = start_progress
        candidate_return = 0.0
        final_progress = start_progress
        steps_taken = 0
        for action_index, action in enumerate(candidate):
            _, reward, done, _ = env.step(action)
            steps_taken = action_index + 1
            candidate_return += (discount**action_index) * float(reward)
            final_progress = measured_progress(env)
            if candidate_index == 0 and action_index == 0:
                demonstrated_reward = float(reward)
                demonstrated_done = bool(done)
            if done:
                break
        candidate_return += (
            progress_reward
            * (discount**steps_taken)
            * max(0.0, final_progress - start_progress)
        )
        q_values[candidate_index] = candidate_return
    if progress is None:
        raise ValueError("candidate bank must contain at least one rollout")
    return q_values, demonstrated_reward, demonstrated_done, progress


def numeric_demo_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else 2**31 - 1, value)


def collect_demo(
    env,
    demo,
    action_pred_steps: int,
    num_candidates: int,
    perturbation: float,
    discount: float,
    progress_reward: float,
    max_steps: int | None,
) -> dict[str, np.ndarray]:
    if "states" not in demo or "actions" not in demo:
        raise ValueError("LIBERO demo must contain states and actions")
    states = np.asarray(demo["states"])
    actions = np.asarray(demo["actions"], dtype=np.float32)
    length = min(len(states), len(actions))
    if max_steps is not None:
        length = min(length, max_steps)
    if length < 1:
        raise ValueError("LIBERO demonstration contains no aligned states/actions")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("LIBERO demonstration actions must have shape [time, 7]")
    rewards = np.zeros(length, dtype=np.float32)
    dones = np.zeros(length, dtype=np.bool_)
    progress = np.zeros(length, dtype=np.float32)
    candidate_actions = np.zeros(
        (length, num_candidates, action_pred_steps, 7), dtype=np.float32
    )
    candidate_q_values = np.zeros(
        (length, num_candidates), dtype=np.float32
    )
    for step in range(length):
        expert = expert_chunk(actions, step, action_pred_steps)
        candidates = candidate_bank(
            expert, num_candidates, perturbation
        )
        q_values, reward, done, measured = measure_rollouts(
            env,
            states[step],
            candidates,
            discount,
            progress_reward,
        )
        rewards[step] = reward
        dones[step] = done
        progress[step] = measured
        candidate_actions[step] = candidates
        candidate_q_values[step] = q_values
    return {
        "reward": rewards,
        "done": dones,
        "progress": progress,
        "candidate_actions": candidate_actions,
        "candidate_q_values": candidate_q_values,
        "candidate_rule_ids": np.asarray(
            stable_candidate_rule_ids(num_candidates)
        ),
    }


def main() -> None:
    args = parse_args()
    libero_path = args.libero_path.expanduser().resolve()
    sys.path.insert(0, str(libero_path))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from utils.libero_compat import apply_robosuite_mujoco3_compat

    apply_robosuite_mujoco3_compat()

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_key = args.suite
    if benchmark_key not in benchmark_dict:
        raise KeyError(
            f"installed LIBERO does not provide benchmark {benchmark_key}"
        )
    task_suite = benchmark_dict[benchmark_key]()
    sources = sorted(args.dataset_dir.glob("*_demo.hdf5"))
    if not sources:
        raise FileNotFoundError(
            f"no *_demo.hdf5 files found in {args.dataset_dir}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    episode_id = 0
    stop = False
    for source in sources:
        task_index, task = task_for_file(task_suite, source)
        bddl_file = (
            libero_path
            / "libero"
            / "libero"
            / "bddl_files"
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        try:
            with h5py.File(source, "r") as handle:
                demos = handle["data"]
                for demo_key in sorted(demos.keys(), key=numeric_demo_key):
                    if (
                        args.max_episodes is not None
                        and episode_id >= args.max_episodes
                    ):
                        stop = True
                        break
                    env.reset()
                    arrays = collect_demo(
                        env,
                        demos[demo_key],
                        args.action_pred_steps,
                        args.num_candidates,
                        args.perturbation,
                        args.discount,
                        args.progress_reward,
                        args.max_steps_per_episode,
                    )
                    output_path = args.output / f"{episode_id:06d}.npz"
                    np.savez_compressed(output_path, **arrays)
                    manifest.append(
                        {
                            "episode_id": f"{episode_id:06d}",
                            "source": str(source),
                            "demo": demo_key,
                            "task_index": task_index,
                            "task": task.name,
                            "steps": int(arrays["reward"].shape[0]),
                            "candidate_rule_ids": arrays[
                                "candidate_rule_ids"
                            ].tolist(),
                        }
                    )
                    episode_id += 1
        finally:
            env.close()
        if stop:
            break
    if not manifest:
        raise RuntimeError("no supervision episodes were collected")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"collected {len(manifest)} LIBERO episodes into {args.output}"
    )


if __name__ == "__main__":
    main()
