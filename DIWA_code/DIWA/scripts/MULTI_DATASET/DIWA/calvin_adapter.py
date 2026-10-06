"""CALVIN simulator-measured training and action-step adapter.

The public demonstrations provide action priors. Observations, shaped task
rewards, progress, and every candidate Q value are collected from the official
PyBullet environment. Each candidate chunk is branched from the exact same
``robot_obs`` / ``scene_obs`` state with ``PlayTableSimEnv.reset``.
"""

from __future__ import annotations

import argparse
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


TASK_NAME = "turn_on_lightbulb"
LANGUAGE = "move the light switch to turn on the yellow light"
CANDIDATE_RULE_IDS = (
    "demonstration",
    "no_cartesian_motion_keep_gripper",
    "invert_gripper",
    "signed_cartesian_counter_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--dataset", required=True, type=Path)
    export.add_argument("--calvin-env", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--calvin-env", required=True, type=Path)
    evaluate.add_argument("--action", required=True, type=Path)
    evaluate.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def convert_action(actions: np.ndarray) -> np.ndarray:
    converted = np.asarray(actions, dtype=np.float32).copy()
    converted[..., :6] = np.clip(converted[..., :6], -1.0, 1.0)
    converted[..., 6] = (converted[..., 6] > 0).astype(np.float32)
    return converted


def native_action(action: np.ndarray) -> np.ndarray:
    native = np.asarray(action, dtype=np.float64).copy()
    native[:6] = np.clip(native[:6], -1.0, 1.0)
    native[6] = 1.0 if native[6] >= 0.5 else -1.0
    return native


def make_env(calvin_env: Path):
    import hydra
    from hydra import compose, initialize_config_dir
    import calvin_env.envs.play_table_env as play_table_env

    play_table_env.get_git_commit_hash = lambda _path: "github-source-archive"
    with initialize_config_dir(config_dir=str(calvin_env.resolve() / "conf")):
        config = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                "record=false",
                "cameras=static_and_gripper",
                "env.use_egl=false",
            ],
        )
    return hydra.utils.instantiate(config.env)


def close_env(env) -> None:
    env.close()
    # CALVIN's destructor otherwise disconnects the same PyBullet client twice.
    env.ownsPhysicsClient = False


def state_vector(observation: dict) -> np.ndarray:
    robot = np.asarray(observation["robot_obs"], dtype=np.float32)
    return np.concatenate((robot[:6], robot[-1:]))


def _light_switch(env):
    return next(
        (switch for switch in env.scene.switches if switch.effect == "lightbulb"),
        env.scene.switches[0],
    )


def task_progress(env) -> float:
    """Measured reach/switch/light predicate for ``turn_on_lightbulb``."""
    observation = env.get_state_obs()
    tcp = np.asarray(observation["robot_obs"][:3], dtype=np.float64)
    switch = _light_switch(env)
    link = env.p.getLinkState(
        switch.uid,
        switch.joint_index,
        physicsClientId=env.cid,
    )
    switch_position = np.asarray(link[0], dtype=np.float64)
    reach = float(np.exp(-np.linalg.norm(tcp - switch_position) / 0.30))

    value = switch.get_state()
    if switch.initial_state <= switch.trigger_threshold:
        denominator = max(switch.trigger_threshold - switch.initial_state, 1e-6)
        switch_fraction = (value - switch.initial_state) / denominator
    else:
        denominator = max(switch.initial_state - switch.trigger_threshold, 1e-6)
        switch_fraction = (switch.initial_state - value) / denominator
    switch_fraction = float(np.clip(switch_fraction, 0.0, 1.0))
    light = next(
        (light for light in env.scene.lights if light.name == "lightbulb"),
        env.scene.lights[0],
    )
    light_on = float(bool(light.get_state()))
    return float(np.clip(0.35 * reach + 0.40 * switch_fraction + 0.25 * light_on, 0.0, 1.0))


def task_success(env) -> bool:
    light = next(
        (light for light in env.scene.lights if light.name == "lightbulb"),
        env.scene.lights[0],
    )
    return bool(light.get_state())


def candidate_bank(chunk: np.ndarray) -> np.ndarray:
    candidates = np.repeat(np.asarray(chunk, dtype=np.float32)[None], 4, axis=0)
    # Stable candidate rules: demonstration, no Cartesian displacement,
    # gripper inversion, and signed Cartesian counter-action.
    candidates[1, :, :6] = 0.0
    candidates[2, :, 6] = 1.0 - candidates[2, :, 6]
    candidates[3, :, :6] *= -1.0
    return candidates


def _episode_actions(dataset: Path, episode_index: int) -> np.ndarray:
    import pyarrow.parquet as pq

    per_episode = (
        dataset
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )
    if per_episode.is_file():
        data = pq.read_table(per_episode, columns=["action"]).to_pydict()
        actions = np.asarray(data["action"][:3], dtype=np.float32)
    else:
        packed = dataset / "data/chunk-000/file-000.parquet"
        data = pq.read_table(packed).to_pydict()
        episodes = np.asarray(data["episode_index"], dtype=np.int64)
        rows = np.flatnonzero(episodes == episode_index)[:3]
        actions = np.asarray([data["action"][row] for row in rows], dtype=np.float32)
    if actions.shape != (3, 7):
        raise RuntimeError(
            f"CALVIN episode {episode_index} needs three 7-D action priors"
        )
    return convert_action(actions)


def collect_episode(env, actions: np.ndarray, seed: int) -> dict:
    env.seed(seed)
    observation = env.reset()
    observations = [observation]
    rewards = []
    dones = []
    progress_values = []
    chunks = []
    candidates = []
    candidate_q = []

    for step_index in range(2):
        chunk = np.stack((actions[step_index], actions[step_index + 1]))
        alternatives = candidate_bank(chunk)
        base_robot = np.asarray(observation["robot_obs"], dtype=np.float64).copy()
        base_scene = np.asarray(observation["scene_obs"], dtype=np.float64).copy()

        def restore() -> None:
            env.reset(robot_obs=base_robot, scene_obs=base_scene)

        measured = measure_candidate_rollouts(
            alternatives,
            restore_state=restore,
            step_action=lambda action: env.step(native_action(action)),
            measure_progress=lambda: task_progress(env),
            measure_success=lambda: task_success(env),
        )
        candidate_q.append(measured.q_values)
        chunks.append(chunk)
        candidates.append(alternatives)

        restore()
        previous = task_progress(env)
        observation, _, _, _ = env.step(native_action(actions[step_index]))
        following = task_progress(env)
        success = task_success(env)
        rewards.append(shaped_task_reward(previous, following, success))
        dones.append(success)
        progress_values.append(following)
        observations.append(observation)

    return {
        "primary": np.stack(
            [obs["rgb_obs"]["rgb_static"] for obs in observations]
        ).astype(np.uint8),
        "wrist": np.stack(
            [obs["rgb_obs"]["rgb_gripper"] for obs in observations]
        ).astype(np.uint8),
        "state": np.stack([state_vector(obs) for obs in observations]),
        "actions": np.stack(chunks),
        "candidate_actions": np.stack(candidates),
        "candidate_q_values": np.stack(candidate_q),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "progress": np.asarray(progress_values, dtype=np.float32),
    }


def export_batch(dataset: Path, calvin_env: Path, output: Path) -> None:
    env = make_env(calvin_env)
    try:
        episodes = [
            collect_episode(env, _episode_actions(dataset, index), seed=index)
            for index in (0, 1)
        ]
    finally:
        close_env(env)

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
        dataset=np.asarray("CALVIN"),
        task=np.asarray(TASK_NAME),
        language=np.asarray(LANGUAGE),
        continuous_action_dim=np.asarray(6, dtype=np.int64),
        state_arm_dim=np.asarray(6, dtype=np.int64),
        gripper_width=np.asarray(False),
        supervision_source=np.asarray("official_pybullet_branched_rollouts"),
        reward_source=np.asarray("task_predicate_potential_delta"),
        candidate_q_source=np.asarray(
            "discounted_official_simulator_task_predicate_return"
        ),
        require_measured_candidate_q=np.asarray(True),
        candidate_q_maximum_spread=np.asarray(
            q_diagnostics["maximum_spread"], dtype=np.float32
        ),
    )
    print(json.dumps(q_diagnostics, indent=2))
    print(f"CALVIN_MEASURED_EXPORT_OK {output}")


def evaluate_action(calvin_env: Path, action_path: Path, report_path: Path) -> None:
    env = make_env(calvin_env)
    try:
        before = env.reset()
        before_progress = task_progress(env)
        submitted = native_action(np.load(action_path)[0])
        after, _, done, _ = env.step(submitted)
        after_progress = task_progress(env)
        success = task_success(env)
        finite = bool(
            np.isfinite(after["robot_obs"]).all()
            and np.isfinite(after["scene_obs"]).all()
        )
        if not finite:
            raise RuntimeError("CALVIN produced a non-finite observation")
        report = {
            "status": "ok",
            "dataset": "CALVIN",
            "environment": type(env).__name__,
            "submitted_action": submitted.tolist(),
            "reward": shaped_task_reward(before_progress, after_progress, success),
            "progress_before": before_progress,
            "progress_after": after_progress,
            "success": success,
            "done": bool(done or success),
            "robot_state_changed": bool(
                not np.allclose(before["robot_obs"], after["robot_obs"])
            ),
            "primary_shape": list(after["rgb_obs"]["rgb_static"].shape),
            "wrist_shape": list(after["rgb_obs"]["rgb_gripper"].shape),
            "finite_observation": finite,
        }
    finally:
        close_env(env)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_CALVIN_ENV_SMOKE_OK")


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_batch(args.dataset, args.calvin_env, args.output)
    else:
        evaluate_action(args.calvin_env, args.action, args.report)


if __name__ == "__main__":
    main()
