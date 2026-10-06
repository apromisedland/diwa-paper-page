"""End-to-end RoboCasa OOD environment generation and evaluation.

The variants modify the actual MuJoCo environment, not pre-rendered images:
material/texture appearance, light parameters, moving wall geometry, an unseen
kitchen layout, and target-door contact friction. Collection exports policy
observations; evaluation rebuilds every variant and executes its predicted
action in the corresponding official environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robocasa_adapter import (
    action_dict,
    inner_env,
    make_env,
    shaped_task_reward,
    state_vector,
    task_progress,
    task_success,
)


VARIANTS = (
    "baseline",
    "texture_shift",
    "lighting_shift",
    "background_motion",
    "unseen_layout",
    "contact_counterfactual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--report", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--observations", required=True, type=Path)
    evaluate.add_argument("--actions", required=True, type=Path)
    evaluate.add_argument("--report", required=True, type=Path)
    evaluate.add_argument("--metrics-output", required=True, type=Path)
    return parser.parse_args()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _geom_names(model) -> list[str]:
    return [model.geom_id2name(index) or "" for index in range(model.ngeom)]


def _target_geom_ids(env) -> np.ndarray:
    task = inner_env(env)
    prefix = task.fxtr.name.lower()
    names = _geom_names(task.sim.model)
    ids = [
        index
        for index, name in enumerate(names)
        if prefix in name.lower()
        and ("door" in name.lower() or "handle" in name.lower())
    ]
    if not ids:
        ids = [index for index, name in enumerate(names) if prefix in name.lower()]
    if not ids:
        raise RuntimeError("could not identify target cabinet contact geoms")
    return np.asarray(ids, dtype=np.int64)


def _background_geom_ids(env) -> np.ndarray:
    model = inner_env(env).sim.model
    names = _geom_names(model)
    ids = [
        index
        for index, name in enumerate(names)
        if any(marker in name.lower() for marker in ("wall", "backing"))
    ]
    if not ids:
        raise RuntimeError("could not identify background wall geoms")
    return np.asarray(ids, dtype=np.int64)


def apply_variant(env, variant: str, phase: int = 0) -> dict:
    """Apply and verify one environment-level OOD intervention."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown OOD variant {variant}")
    task = inner_env(env)
    model = task.sim.model
    result = {"variant": variant, "phase": int(phase), "changed": False}

    if variant in {"baseline", "unseen_layout"}:
        result.update(
            changed=variant == "unseen_layout",
            mechanism=(
                "layout_and_style_ids=(6,9)"
                if variant == "unseen_layout"
                else "none"
            ),
        )
    elif variant == "texture_shift":
        before = np.asarray(model.mat_rgba).copy()
        shifted = before.copy()
        shifted[:, :3] = np.clip(
            shifted[:, [1, 2, 0]] * np.asarray([1.15, 0.80, 1.05]),
            0.0,
            1.0,
        )
        model.mat_rgba[:] = shifted
        # Procedural geoms without a material need an explicit appearance shift.
        geom = np.asarray(model.geom_rgba).copy()
        geom[:, :3] = np.clip(geom[:, [2, 0, 1]], 0.0, 1.0)
        model.geom_rgba[:] = geom
        result.update(
            changed=not np.array_equal(before, shifted),
            mechanism="MuJoCo material and untextured-geom appearance remap",
            before_hash=_array_hash(before),
            after_hash=_array_hash(shifted),
        )
    elif variant == "lighting_shift":
        before = np.asarray(model.light_diffuse).copy()
        model.light_diffuse[:] = np.clip(
            before * np.asarray([0.55, 0.80, 1.25]), 0.0, 2.0
        )
        model.light_ambient[:] = np.clip(
            np.asarray(model.light_ambient) + np.asarray([0.08, 0.02, 0.12]),
            0.0,
            1.0,
        )
        model.light_pos[:, 0] += 0.20
        result.update(
            changed=not np.array_equal(before, model.light_diffuse),
            mechanism="MuJoCo light diffuse/ambient/position shift",
            before_hash=_array_hash(before),
            after_hash=_array_hash(model.light_diffuse),
        )
    elif variant == "background_motion":
        ids = _background_geom_ids(env)
        displacement = 0.04 * np.sin((phase + 1) * np.pi / 3.0)
        model.geom_pos[ids, 0] += displacement
        result.update(
            changed=bool(abs(displacement) > 0),
            mechanism="time-varying MuJoCo wall-geometry translation",
            changed_geom_count=int(len(ids)),
            displacement=float(displacement),
        )
    elif variant == "contact_counterfactual":
        ids = _target_geom_ids(env)
        before = np.asarray(model.geom_friction[ids]).copy()
        model.geom_friction[ids, 0] = np.maximum(
            model.geom_friction[ids, 0] * 0.03, 1e-4
        )
        model.geom_friction[ids, 1:] = np.minimum(
            model.geom_friction[ids, 1:], 1e-4
        )
        result.update(
            changed=not np.array_equal(before, model.geom_friction[ids]),
            mechanism="target-door contact-friction counterfactual",
            changed_geom_count=int(len(ids)),
            before_hash=_array_hash(before),
            after_hash=_array_hash(model.geom_friction[ids]),
        )

    task.sim.forward()
    if variant not in {"baseline"} and not result["changed"]:
        raise RuntimeError(f"OOD variant {variant} did not change the environment")
    return result


def make_variant_env(variant: str, seed: int):
    # OpenCabinet is not constructible in every registered kitchen.  (6, 9)
    # is an official evaluation pair with a topology distinct from training
    # pair (1, 1), and is verified by an actual reset below.
    layout = ((6, 9),) if variant == "unseen_layout" else ((1, 1),)
    env = make_env(seed=seed, layout_and_style_ids=layout)
    observation, _ = env.reset(seed=seed)
    return env, observation


def _noop_action() -> np.ndarray:
    action = np.zeros(12, dtype=np.float32)
    action[10:] = 0.0
    return action


def collect_variant(variant: str, seed: int) -> tuple[dict, dict]:
    env, _ = make_variant_env(variant, seed)
    try:
        variant_steps = []
        observations = []
        for phase in range(3):
            if phase == 0 or variant == "background_motion":
                variant_steps.append(apply_variant(env, variant, phase))
            observation, _, _, _, _ = env.step(action_dict(_noop_action()))
            observations.append(observation)
        return {
            "primary": np.stack(
                [obs["video.robot0_agentview_left"] for obs in observations]
            ).astype(np.uint8),
            "wrist": np.stack(
                [obs["video.robot0_eye_in_hand"] for obs in observations]
            ).astype(np.uint8),
            "state": np.stack([state_vector(obs) for obs in observations]),
        }, {
            "variant": variant,
            "seed": seed,
            "environment_changes": variant_steps,
        }
    finally:
        env.close()


def collect_ood(output: Path, report_path: Path) -> None:
    batches = []
    reports = []
    # Common random seed isolates each intervention from scene randomization.
    for variant in VARIANTS:
        batch, report = collect_variant(variant, seed=100)
        batches.append(batch)
        reports.append(report)
    baseline = batches[0]["primary"].astype(np.float32)
    for batch, report in zip(batches, reports):
        report["primary_pixel_mae_from_baseline"] = float(
            np.abs(batch["primary"].astype(np.float32) - baseline).mean()
        )
        report["finite_observation"] = bool(
            np.isfinite(batch["state"]).all()
            and np.isfinite(batch["primary"]).all()
            and np.isfinite(batch["wrist"]).all()
        )
    if not all(report["finite_observation"] for report in reports):
        raise RuntimeError("an OOD generator returned non-finite observations")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        primary=np.stack([batch["primary"] for batch in batches]),
        wrist=np.stack([batch["wrist"] for batch in batches]),
        state=np.stack([batch["state"] for batch in batches]),
        variants=np.asarray(VARIANTS),
        language=np.asarray("open the cabinet door"),
        action_dim=np.asarray(12, dtype=np.int64),
        continuous_action_dim=np.asarray(10, dtype=np.int64),
        state_arm_dim=np.asarray(14, dtype=np.int64),
        source=np.asarray("official_robocasa_mujoco_ood_generators"),
    )
    report = {
        "status": "ok",
        "environment": "robocasa/OpenCabinet",
        "output": str(output),
        "variants": reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_ROBOCASA_OOD_COLLECT_OK")


def _contact_count(env) -> int:
    task = inner_env(env)
    target = set(_target_geom_ids(env).tolist())
    count = 0
    for index in range(task.sim.data.ncon):
        contact = task.sim.data.contact[index]
        if int(contact.geom1) in target or int(contact.geom2) in target:
            count += 1
    return count


def evaluate_ood(
    observation_path: Path,
    action_path: Path,
    report_path: Path,
    metrics_path: Path,
) -> None:
    archive = np.load(observation_path, allow_pickle=False)
    variants = archive["variants"].tolist()
    if tuple(variants) != VARIANTS:
        raise RuntimeError("OOD observation variants do not match the evaluator")
    actions = np.load(action_path).astype(np.float32)
    if actions.shape != (len(VARIANTS), 2, 12):
        raise RuntimeError(f"unexpected OOD policy action shape {actions.shape}")

    results = []
    for variant, variant_actions in zip(VARIANTS, actions):
        env, before = make_variant_env(variant, seed=100)
        try:
            change = apply_variant(env, variant, phase=0)
            before_progress = task_progress(env)
            if variant == "background_motion":
                apply_variant(env, variant, phase=1)
            after, _, terminated, truncated, _ = env.step(
                action_dict(variant_actions[0])
            )
            after_progress = task_progress(env)
            success = task_success(env)
            results.append(
                {
                    "variant": variant,
                    "environment_change": change,
                    "reward": shaped_task_reward(
                        before_progress, after_progress, success
                    ),
                    "progress_before": before_progress,
                    "progress_after": after_progress,
                    "success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "contact_count": _contact_count(env),
                    "robot_state_changed": bool(
                        not np.allclose(state_vector(before), state_vector(after))
                    ),
                    "finite_observation": bool(
                        np.isfinite(state_vector(after)).all()
                    ),
                }
            )
        finally:
            env.close()

    baseline = results[0]
    standard_success = float(baseline["success"])
    ood_success = np.asarray([float(item["success"]) for item in results[1:]])
    standard_delta = baseline["progress_after"] - baseline["progress_before"]
    ood_delta = np.asarray(
        [item["progress_after"] - item["progress_before"] for item in results[1:]],
        dtype=np.float32,
    )
    report = {
        "status": "ok",
        "environment": "robocasa/OpenCabinet",
        "policy_action_source": str(action_path),
        "standard_success": bool(standard_success),
        "ood_success_rate": float(ood_success.mean()),
        "standard_progress_delta": float(standard_delta),
        "ood_progress_delta_mean": float(ood_delta.mean()),
        "ood_progress_retention": float(
            ood_delta.mean() / max(abs(standard_delta), 1e-6)
        ),
        "variants": results,
    }
    if not all(
        item["finite_observation"] and item["robot_state_changed"]
        for item in results
    ):
        raise RuntimeError("OOD policy evaluation failed finite/state-change checks")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metrics_path,
        variants=np.asarray(VARIANTS),
        standard_success=np.asarray([standard_success], dtype=np.float32),
        ood_success=ood_success,
        standard_progress_delta=np.asarray([standard_delta], dtype=np.float32),
        ood_progress_delta=ood_delta,
        rewards=np.asarray([item["reward"] for item in results], dtype=np.float32),
        task_progress=np.asarray(
            [item["progress_after"] for item in results], dtype=np.float32
        ),
        contact_counts=np.asarray(
            [item["contact_count"] for item in results], dtype=np.int64
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_ROBOCASA_OOD_EVAL_OK")


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect_ood(args.output, args.report)
    else:
        evaluate_ood(
            args.observations,
            args.actions,
            args.report,
            args.metrics_output,
        )


if __name__ == "__main__":
    main()
