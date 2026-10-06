"""RoboTwin simulator export and official SAPIEN action-step adapter.

RoboTwin's ALOHA embodiment exposes native qpos actions in the order
``left_arm(6), left_gripper, right_arm(6), right_gripper``.  DreamVLA-DIWA
keeps continuous controls first, so this adapter uses the canonical order
``left_arm(6), right_arm(6), left_gripper, right_gripper``.

The official RoboTwin task imports cuRobo even when only qpos control is used.
For this smoke test the optional cuRobo imports are stubbed and the task's
existing qpos control loop is supplied with a deterministic linear TOPP
implementation.  SAPIEN scene construction, rendering, robot drives, task
actors, stability checks, success checks, and action execution remain the
official RoboTwin implementations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from measured_supervision import (  # noqa: E402
    shaped_task_reward,
    validate_measured_candidate_q,
)


TASK_NAME = "beat_block_hammer"
ACTION_DIM = 14
CONTINUOUS_ACTION_DIM = 12
CANDIDATE_RULE_IDS = (
    "demonstration",
    "invert_left_gripper",
    "invert_right_gripper",
    "invert_both_arm_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--robotwin", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--robotwin", required=True, type=Path)
    evaluate.add_argument("--action", required=True, type=Path)
    evaluate.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def canonical_action(native: np.ndarray) -> np.ndarray:
    """Move the two native gripper coordinates behind both arm vectors."""
    native = np.asarray(native, dtype=np.float32)
    if native.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-D RoboTwin action")
    return np.concatenate(
        (native[..., :6], native[..., 7:13], native[..., 6:7], native[..., 13:14]),
        axis=-1,
    )


def native_action(canonical: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical, dtype=np.float32)
    if canonical.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-D canonical action")
    return np.concatenate(
        (
            canonical[..., :6],
            canonical[..., 12:13],
            canonical[..., 6:12],
            canonical[..., 13:14],
        ),
        axis=-1,
    )


def _install_curobo_import_stub() -> None:
    """Make RoboTwin's optional cuRobo class definition importable.

    The placeholder classes are never instantiated: ``Robot.set_planner`` is
    replaced below before an environment is constructed.
    """
    if "curobo" in sys.modules:
        return

    module_names = (
        "curobo",
        "curobo.types",
        "curobo.types.math",
        "curobo.types.robot",
        "curobo.wrap",
        "curobo.wrap.reacher",
        "curobo.wrap.reacher.motion_gen",
        "curobo.util",
        "curobo.util.logger",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for name in ("curobo", "curobo.types", "curobo.wrap", "curobo.wrap.reacher"):
        modules[name].__path__ = []

    class _UnavailableCurobo:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cuRobo is unavailable in the qpos smoke environment")

    motion = modules["curobo.wrap.reacher.motion_gen"]
    for name in (
        "MotionGen",
        "MotionGenConfig",
        "MotionGenPlanConfig",
        "PoseCostMetric",
    ):
        setattr(motion, name, _UnavailableCurobo)
    modules["curobo.types.math"].Pose = _UnavailableCurobo
    modules["curobo.types.robot"].JointState = _UnavailableCurobo
    modules["curobo.util.logger"].setup_logger = lambda *args, **kwargs: None
    modules["curobo.util"].logger = modules["curobo.util.logger"]
    sys.modules.update(modules)


class _LinearQposPlanner:
    """Small deterministic trajectory parameterizer for qpos smoke steps."""

    def TOPP(self, path, dt, verbose=False):  # noqa: N802 - RoboTwin API
        path = np.asarray(path, dtype=np.float64)
        if path.shape[0] < 2:
            raise ValueError("TOPP path needs a start and target")
        position = np.linspace(path[0], path[-1], 16, dtype=np.float64)
        velocity = np.gradient(position, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        times = np.arange(position.shape[0], dtype=np.float64) * dt
        return times, position, velocity, acceleration, float(times[-1])

    def plan_grippers(self, now_val, target_val):
        values = np.linspace(now_val, target_val, 16, dtype=np.float64)
        return {
            "num_step": len(values),
            "per_step": float(target_val - now_val) / len(values),
            "result": values,
        }


def _smoke_set_planner(robot, scene=None) -> None:
    planner = _LinearQposPlanner()
    robot.communication_flag = False
    robot.left_planner = planner
    robot.right_planner = planner
    robot.left_mplib_planner = planner
    robot.right_mplib_planner = planner


def _task_args(robotwin: Path) -> dict:
    with (robotwin / "task_config/demo_clean.yml").open(encoding="utf-8") as file:
        args = yaml.safe_load(file)
    robot_file = robotwin / "assets/embodiments/aloha-agilex"
    with (robot_file / "config.yml").open(encoding="utf-8") as file:
        embodiment = yaml.safe_load(file)
    args.update(
        {
            "task_name": TASK_NAME,
            "left_robot_file": str(robot_file),
            "right_robot_file": str(robot_file),
            "left_embodiment_config": embodiment,
            "right_embodiment_config": embodiment,
            "dual_arm_embodied": True,
            "embodiment_name": "aloha-agilex",
            "task_config": "demo_clean",
            "save_path": str(robotwin / "smoke_output"),
            "need_plan": False,
            "save_data": False,
            "eval_mode": True,
            "eval_video_save_dir": None,
            "render_freq": 0,
        }
    )
    return args


def make_task(robotwin: Path, seed: int, episode: int):
    robotwin = robotwin.resolve()
    os.chdir(robotwin)
    if str(robotwin) not in sys.path:
        sys.path.insert(0, str(robotwin))
    _install_curobo_import_stub()

    from envs.robot.robot import Robot
    from envs.beat_block_hammer import beat_block_hammer

    Robot.set_planner = _smoke_set_planner
    task = beat_block_hammer()
    task.setup_demo(seed=seed, now_ep_num=episode, **_task_args(robotwin))
    return task


def close_task(task) -> None:
    try:
        task.close_env(clear_cache=True)
    finally:
        del task
        gc.collect()


def _joint_limits(task) -> tuple[np.ndarray, np.ndarray]:
    def limits(joints) -> np.ndarray:
        result = []
        for joint in joints:
            value = np.asarray(joint.get_limits(), dtype=np.float32).reshape(-1, 2)[0]
            result.append(value)
        return np.asarray(result)

    return limits(task.robot.left_arm_joints), limits(task.robot.right_arm_joints)


def clamp_native_action(task, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    left_limits, right_limits = _joint_limits(task)
    action[:6] = np.clip(action[:6], left_limits[:, 0], left_limits[:, 1])
    action[7:13] = np.clip(action[7:13], right_limits[:, 0], right_limits[:, 1])
    action[[6, 13]] = np.clip(action[[6, 13]], 0.0, 1.0)
    return action


def target_from_observation(task, observation: dict, direction: float) -> np.ndarray:
    target = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).copy()
    target[0] += 0.035 * direction
    target[7] -= 0.035 * direction
    return clamp_native_action(task, target)


def _camera(observation: dict, name: str) -> np.ndarray:
    image = np.asarray(observation["observation"][name]["rgb"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"RoboTwin returned an invalid {name} image: {image.shape}")
    return image.astype(np.uint8, copy=False)


def _pose_position(pose) -> np.ndarray:
    """Return xyz for both SAPIEN Pose objects and RoboTwin's 7D pose lists."""
    value = pose.p if hasattr(pose, "p") else pose
    position = np.asarray(value, dtype=np.float32).reshape(-1)
    if position.size < 3:
        raise RuntimeError(f"RoboTwin returned an invalid pose: {position.shape}")
    return position[:3]


def task_progress(task) -> float:
    """Measured reach/grasp/alignment predicate for hammering the block."""
    hammer = _pose_position(task.hammer.get_functional_point(0, "pose"))
    block = _pose_position(task.block.get_functional_point(1, "pose"))
    left = _pose_position(task.robot.get_left_ee_pose())
    right = _pose_position(task.robot.get_right_ee_pose())
    reach_distance = min(np.linalg.norm(left - hammer), np.linalg.norm(right - hammer))
    alignment_distance = np.linalg.norm(hammer - block)
    reach = float(np.exp(-reach_distance / 0.25))
    alignment = float(np.exp(-alignment_distance / 0.25))
    contact = float(
        task.check_actors_contact(task.hammer.get_name(), task.block.get_name())
    )
    return float(np.clip(0.55 * reach + 0.35 * alignment + 0.10 * contact, 0.0, 1.0))


def candidate_bank(actions: np.ndarray) -> np.ndarray:
    chunks = np.stack((actions[:2], actions[1:3]))
    alternatives = np.repeat(chunks[:, None], 4, axis=1)
    alternatives[:, 1, :, 12] = 1.0 - alternatives[:, 1, :, 12]
    alternatives[:, 2, :, 13] = 1.0 - alternatives[:, 2, :, 13]
    alternatives[:, 3, :, :12] *= -1.0
    return alternatives


def collect_episode(robotwin: Path, seed: int, episode: int) -> dict:
    task = make_task(robotwin, seed, episode)
    try:
        observations = [task.get_obs()]
        actions = []
        rewards = []
        dones = []
        progress_values = []
        for direction in (1.0, -1.0):
            action = target_from_observation(task, observations[-1], direction)
            actions.append(action)
            previous = task_progress(task)
            task.take_action(action, action_type="qpos")
            observations.append(task.get_obs())
            success = bool(task.check_success())
            following = task_progress(task)
            rewards.append(shaped_task_reward(previous, following, success))
            dones.append(success)
            progress_values.append(following)
        actions.append(target_from_observation(task, observations[-1], 1.0))
        return {
            "primary": np.stack([_camera(obs, "head_camera") for obs in observations]),
            "wrist": np.stack([_camera(obs, "left_camera") for obs in observations]),
            "state": np.stack(
                [canonical_action(obs["joint_action"]["vector"]) for obs in observations]
            ),
            "actions": canonical_action(np.stack(actions)),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "progress": np.asarray(progress_values, dtype=np.float32),
        }
    finally:
        close_task(task)


def measure_candidate_q(
    robotwin: Path,
    seed: int,
    episode_index: int,
    label_actions: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Rebuild deterministic tasks so each branch has an identical base state."""
    values = np.zeros((2, 4), dtype=np.float32)
    for state_index in range(2):
        prefix = label_actions[:state_index]
        for candidate_index, chunk in enumerate(candidates[state_index]):
            task = make_task(
                robotwin,
                seed=seed,
                episode=episode_index * 100 + state_index * 10 + candidate_index,
            )
            try:
                for prefix_action in prefix:
                    task.take_action(
                        clamp_native_action(task, native_action(prefix_action)),
                        action_type="qpos",
                    )
                previous = task_progress(task)
                discount = 1.0
                for action in chunk:
                    task.take_action(
                        clamp_native_action(task, native_action(action)),
                        action_type="qpos",
                    )
                    following = task_progress(task)
                    success = bool(task.check_success())
                    values[state_index, candidate_index] += discount * shaped_task_reward(
                        previous, following, success
                    )
                    if success:
                        break
                    previous = following
                    discount *= 0.95
            finally:
                close_task(task)
    return values


def export_batch(robotwin: Path, output: Path) -> None:
    episodes = [
        collect_episode(robotwin, seed, index)
        for index, seed in enumerate((0, 1))
    ]
    chunks = []
    candidates = []
    candidate_q = []
    for episode_index, (seed, episode) in enumerate(zip((0, 1), episodes)):
        action = episode["actions"]
        alternatives = candidate_bank(action)
        chunks.append(np.stack((action[:2], action[1:3])))
        candidates.append(alternatives)
        candidate_q.append(
            measure_candidate_q(
                robotwin,
                seed,
                episode_index,
                action,
                alternatives,
            )
        )
    q_values = np.stack(candidate_q)
    q_diagnostics = validate_measured_candidate_q(q_values)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        primary=np.stack([episode["primary"] for episode in episodes]),
        wrist=np.stack([episode["wrist"] for episode in episodes]),
        state=np.stack([episode["state"] for episode in episodes]).astype(np.float32),
        actions=np.stack(chunks).astype(np.float32),
        rewards=np.stack([episode["rewards"] for episode in episodes]),
        dones=np.stack([episode["dones"] for episode in episodes]),
        progress=np.stack([episode["progress"] for episode in episodes]),
        candidate_actions=np.stack(candidates).astype(np.float32),
        candidate_q_values=q_values,
        candidate_rule_ids=np.asarray(CANDIDATE_RULE_IDS),
        dataset=np.asarray("RoboTwin"),
        task=np.asarray(TASK_NAME),
        language=np.asarray("use the hammer to beat the red block"),
        continuous_action_dim=np.asarray(CONTINUOUS_ACTION_DIM, dtype=np.int64),
        state_arm_dim=np.asarray(CONTINUOUS_ACTION_DIM, dtype=np.int64),
        gripper_width=np.asarray(True),
        supervision_source=np.asarray("official_sapien_branched_rollouts"),
        reward_source=np.asarray("task_predicate_potential_delta"),
        candidate_q_source=np.asarray(
            "discounted_official_simulator_task_predicate_return"
        ),
        require_measured_candidate_q=np.asarray(True),
        candidate_q_maximum_spread=np.asarray(
            q_diagnostics["maximum_spread"], dtype=np.float32
        ),
        seeds=np.asarray([0, 1], dtype=np.int64),
    )
    print(json.dumps(q_diagnostics, indent=2))
    print(f"ROBOTWIN_MEASURED_EXPORT_OK {output}")


def evaluate_action(robotwin: Path, action_path: Path, report_path: Path) -> None:
    task = make_task(robotwin, seed=2, episode=0)
    try:
        before = task.get_obs()
        before_progress = task_progress(task)
        policy_action = np.load(action_path)[0].astype(np.float32)
        policy_action[:CONTINUOUS_ACTION_DIM] = np.clip(
            policy_action[:CONTINUOUS_ACTION_DIM], -1.0, 1.0
        )
        policy_action[CONTINUOUS_ACTION_DIM:] = (
            policy_action[CONTINUOUS_ACTION_DIM:] >= 0.5
        ).astype(np.float32)
        submitted = clamp_native_action(task, native_action(policy_action))
        task.take_action(submitted, action_type="qpos")
        after = task.get_obs()
        before_state = canonical_action(before["joint_action"]["vector"])
        after_state = canonical_action(after["joint_action"]["vector"])
        primary = _camera(after, "head_camera")
        wrist = _camera(after, "left_camera")
        finite = bool(
            np.isfinite(after_state).all()
            and np.isfinite(primary).all()
            and np.isfinite(wrist).all()
        )
        if not finite:
            raise RuntimeError("RoboTwin produced a non-finite observation")
        success = bool(task.check_success())
        after_progress = task_progress(task)
        report = {
            "status": "ok",
            "dataset": "RoboTwin",
            "environment": TASK_NAME,
            "submitted_canonical_action": policy_action.tolist(),
            "submitted_native_action": submitted.tolist(),
            "reward": shaped_task_reward(
                before_progress, after_progress, success
            ),
            "progress_before": before_progress,
            "progress_after": after_progress,
            "success": success,
            "robot_state_changed": bool(not np.allclose(before_state, after_state)),
            "primary_shape": list(primary.shape),
            "wrist_shape": list(wrist.shape),
            "finite_observation": finite,
            "qpos_control_loop": "official Base_Task.take_action",
            "trajectory_parameterizer": "linear smoke TOPP",
        }
    finally:
        close_task(task)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_ROBOTWIN_ENV_SMOKE_OK")


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_batch(args.robotwin, args.output)
    else:
        evaluate_action(args.robotwin, args.action, args.report)


if __name__ == "__main__":
    main()
