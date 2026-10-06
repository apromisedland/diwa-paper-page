"""RoboCasa simulator-measured training and official action-step adapter.

Target-human actions are used as policy-label priors. Every observation,
task-predicate reward/progress value, and candidate return is measured in the
official ``robocasa/OpenCabinet`` MuJoCo environment. Candidate chunks branch
from an identical serialized ``MjSimState``; learned critic predictions never
enter the exported Q labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from measured_supervision import (  # noqa: E402
    measure_candidate_rollouts,
    shaped_task_reward,
    validate_measured_candidate_q,
)


TASK_INDEX = 5
TASK_NAME = "OpenCabinet"
CONTINUOUS_NATIVE_INDICES = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])
DISCRETE_NATIVE_INDICES = np.asarray([4, 11])
CANDIDATE_RULE_IDS = (
    "demonstration",
    "invert_gripper_close",
    "invert_control_mode",
    "invert_continuous_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--dataset", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--action", required=True, type=Path)
    evaluate.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def canonical_action(native: np.ndarray) -> np.ndarray:
    native = np.asarray(native, dtype=np.float32)
    continuous = np.clip(native[..., CONTINUOUS_NATIVE_INDICES], -1.0, 1.0)
    discrete = (native[..., DISCRETE_NATIVE_INDICES] > 0.0).astype(np.float32)
    return np.concatenate((continuous, discrete), axis=-1)


def action_dict(canonical: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(canonical, dtype=np.float32).copy()
    action[:10] = np.clip(action[:10], -1.0, 1.0)
    action[10:] = (action[10:] >= 0.5).astype(np.float32)
    return {
        "action.base_motion": action[:4],
        "action.end_effector_position": action[4:7],
        "action.end_effector_rotation": action[7:10],
        "action.control_mode": action[10:11],
        "action.gripper_close": action[11:12],
    }


def make_env(seed: int, layout_and_style_ids=((1, 1),)):
    import gymnasium as gym
    import robocasa  # noqa: F401 - registers official environments

    return gym.make(
        "robocasa/OpenCabinet",
        split=None,
        seed=seed,
        layout_and_style_ids=list(layout_and_style_ids),
        obj_instance_split="pretrain",
        obj_registries=("lightwheel",),
        generative_textures=False,
    )


def inner_env(env):
    return env.unwrapped.env


def state_vector(observation: dict) -> np.ndarray:
    return np.concatenate(
        (
            observation["state.base_position"],
            observation["state.base_rotation"],
            observation["state.end_effector_position_relative"],
            observation["state.end_effector_rotation_relative"],
            observation["state.gripper_qpos"],
        )
    ).astype(np.float32)


def task_progress(env) -> float:
    """Measured cabinet-open predicate with a reach-to-fixture stage."""
    task = inner_env(env)
    if hasattr(task.fxtr, "get_door_state"):
        door_state = task.fxtr.get_door_state(task)
    else:
        door_state = task.fxtr.get_joint_state(
            task, task.fxtr.door_joint_names
        )
    door_fraction = float(np.mean(list(door_state.values())))
    robot = task.robots[0]
    arm = robot.arms[0]
    eef = np.asarray(task.sim.data.site_xpos[robot.eef_site_id[arm]])
    fixture = np.asarray(task.fxtr.pos)
    reach = float(np.exp(-np.linalg.norm(eef - fixture) / 0.75))
    return float(np.clip(0.75 * door_fraction + 0.25 * reach, 0.0, 1.0))


def task_success(env) -> bool:
    return bool(inner_env(env)._check_success())


def capture_state(env) -> dict:
    task = inner_env(env)
    return {
        "sim": copy.deepcopy(task.sim.get_state()),
        "timestep": int(task.timestep),
        "done": bool(task.done),
    }


def restore_state(env, snapshot: dict) -> None:
    task = inner_env(env)
    task.sim.set_state(copy.deepcopy(snapshot["sim"]))
    task.sim.forward()
    task.timestep = snapshot["timestep"]
    task.done = snapshot["done"]
    for robot in task.robots:
        robot.composite_controller.reset()


def candidate_bank(chunk: np.ndarray) -> np.ndarray:
    alternatives = np.repeat(np.asarray(chunk, dtype=np.float32)[None], 4, axis=0)
    alternatives[1, :, -1] = 1.0 - alternatives[1, :, -1]
    alternatives[2, :, -2] = 1.0 - alternatives[2, :, -2]
    alternatives[3, :, :10] *= -1.0
    return alternatives


def load_action_priors(dataset: Path) -> tuple[list[np.ndarray], np.ndarray]:
    import pyarrow.parquet as pq

    data = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pydict()
    episode = np.asarray(data["episode_index"], dtype=np.int64)
    frame = np.asarray(data["frame_index"], dtype=np.int64)
    task = np.asarray(data["task_index"], dtype=np.int64)
    starts = np.flatnonzero((frame == 0) & (task == TASK_INDEX))[:2]
    if len(starts) != 2:
        raise RuntimeError(f"expected two {TASK_NAME} action-prior episodes")
    priors = []
    for row in starts:
        ep = episode[row]
        rows = np.flatnonzero(episode == ep)[:3]
        native = np.asarray([data["action"][index] for index in rows], dtype=np.float32)
        priors.append(canonical_action(native))
    return priors, episode[starts]


def collect_episode(actions: np.ndarray, seed: int) -> dict:
    env = make_env(seed)
    try:
        observation, _ = env.reset(seed=seed)
        observations = [observation]
        chunks = []
        candidates = []
        candidate_q = []
        rewards = []
        dones = []
        progress_values = []
        for step_index in range(2):
            chunk = np.stack((actions[step_index], actions[step_index + 1]))
            alternatives = candidate_bank(chunk)
            snapshot = capture_state(env)
            measured = measure_candidate_rollouts(
                alternatives,
                restore_state=lambda: restore_state(env, snapshot),
                step_action=lambda action: env.step(action_dict(action)),
                measure_progress=lambda: task_progress(env),
                measure_success=lambda: task_success(env),
            )
            chunks.append(chunk)
            candidates.append(alternatives)
            candidate_q.append(measured.q_values)

            restore_state(env, snapshot)
            previous = task_progress(env)
            observation, _, terminated, truncated, _ = env.step(
                action_dict(actions[step_index])
            )
            following = task_progress(env)
            success = task_success(env)
            rewards.append(shaped_task_reward(previous, following, success))
            dones.append(bool(terminated or truncated or success))
            progress_values.append(following)
            observations.append(observation)

        return {
            "primary": np.stack(
                [obs["video.robot0_agentview_left"] for obs in observations]
            ).astype(np.uint8),
            "wrist": np.stack(
                [obs["video.robot0_eye_in_hand"] for obs in observations]
            ).astype(np.uint8),
            "state": np.stack([state_vector(obs) for obs in observations]),
            "actions": np.stack(chunks),
            "candidate_actions": np.stack(candidates),
            "candidate_q_values": np.stack(candidate_q),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "progress": np.asarray(progress_values, dtype=np.float32),
        }
    finally:
        env.close()


def export_batch(dataset: Path, output: Path) -> None:
    priors, episode_indices = load_action_priors(dataset)
    episodes = [collect_episode(actions, seed) for seed, actions in enumerate(priors)]
    q_values = np.stack([episode["candidate_q_values"] for episode in episodes])
    q_diagnostics = validate_measured_candidate_q(q_values)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        primary=np.stack([episode["primary"] for episode in episodes]),
        wrist=np.stack([episode["wrist"] for episode in episodes]),
        state=np.stack([episode["state"] for episode in episodes]),
        actions=np.stack([episode["actions"] for episode in episodes]),
        rewards=np.stack([episode["rewards"] for episode in episodes]),
        dones=np.stack([episode["dones"] for episode in episodes]),
        progress=np.stack([episode["progress"] for episode in episodes]),
        candidate_actions=np.stack(
            [episode["candidate_actions"] for episode in episodes]
        ),
        candidate_q_values=q_values,
        candidate_rule_ids=np.asarray(CANDIDATE_RULE_IDS),
        dataset=np.asarray("RoboCasa"),
        task=np.asarray(TASK_NAME),
        language=np.asarray("open the cabinet door"),
        continuous_action_dim=np.asarray(10, dtype=np.int64),
        state_arm_dim=np.asarray(14, dtype=np.int64),
        gripper_width=np.asarray(True),
        supervision_source=np.asarray("official_mujoco_branched_rollouts"),
        reward_source=np.asarray("task_predicate_potential_delta"),
        candidate_q_source=np.asarray(
            "discounted_official_simulator_task_predicate_return"
        ),
        require_measured_candidate_q=np.asarray(True),
        candidate_q_maximum_spread=np.asarray(
            q_diagnostics["maximum_spread"], dtype=np.float32
        ),
        episode_indices=episode_indices,
    )
    print(json.dumps(q_diagnostics, indent=2))
    print(f"ROBOCASA_MEASURED_EXPORT_OK {output}")


def evaluate_action(action_path: Path, report_path: Path) -> None:
    env = make_env(seed=0)
    try:
        before, _ = env.reset(seed=0)
        before_progress = task_progress(env)
        policy_action = np.load(action_path)[0].astype(np.float32)
        after, _, terminated, truncated, info = env.step(action_dict(policy_action))
        after_progress = task_progress(env)
        success = bool(info.get("success", task_success(env)))
        before_state = state_vector(before)
        after_state = state_vector(after)
        finite = bool(
            np.isfinite(after_state).all()
            and np.isfinite(after["video.robot0_agentview_left"]).all()
            and np.isfinite(after["video.robot0_eye_in_hand"]).all()
        )
        if not finite:
            raise RuntimeError("RoboCasa produced a non-finite observation")
        report = {
            "status": "ok",
            "dataset": "RoboCasa",
            "environment": "robocasa/OpenCabinet",
            "submitted_canonical_action": policy_action.tolist(),
            "reward": shaped_task_reward(
                before_progress, after_progress, success
            ),
            "progress_before": before_progress,
            "progress_after": after_progress,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
            "robot_state_changed": bool(not np.allclose(before_state, after_state)),
            "primary_shape": list(after["video.robot0_agentview_left"].shape),
            "wrist_shape": list(after["video.robot0_eye_in_hand"].shape),
            "finite_observation": finite,
        }
    finally:
        env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_ROBOCASA_ENV_SMOKE_OK")


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_batch(args.dataset, args.output)
    else:
        evaluate_action(args.action, args.report)


if __name__ == "__main__":
    main()
