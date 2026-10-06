"""Validate and combine simulator and offline dataset smoke reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator_path = args.outputs / "diwa_multi_dataset_smoke.json"
    offline_path = args.outputs / "diwa_offline_dataset_suite.json"
    simulator = json.loads(simulator_path.read_text())
    offline = json.loads(offline_path.read_text())
    if simulator.get("status") != "ok":
        raise RuntimeError("simulator/LIBERO suite is not successful")
    libero = simulator.get("libero", {})
    expected_ablations = {
        "full_diwa",
        "without_influence_estimator",
        "without_counterfactual_swap",
        "without_regret_geometry",
        "without_sparse_imagination",
        "uniform_topk",
    }
    ablations = libero.get("ablation_variants", {})
    if (
        libero.get("policy_status") != "ok"
        or libero.get("ablation_status") != "ok"
        or set(ablations) != expected_ablations
        or any(value != "ok" for value in ablations.values())
    ):
        raise RuntimeError("LIBERO full/ablation smoke suite is incomplete")
    if offline.get("status") != "ok" or offline.get("completed") != 13:
        raise RuntimeError("DROID/OXE suite is incomplete")

    measured_simulators = {}
    for name in ("CALVIN", "RoboCasa", "RoboTwin"):
        slug = name.lower()
        policy_path = args.outputs / f"{slug}_measured_policy.json"
        environment_path = args.outputs / f"{slug}_measured_env.json"
        policy = json.loads(policy_path.read_text())
        environment = json.loads(environment_path.read_text())
        auxiliary = policy.get("aux_losses", {})
        valid = (
            policy.get("status") == "policy_ok"
            and policy.get("measured_candidate_q_required") is True
            and policy.get("candidate_q_supervision_active") is True
            and policy.get("candidate_q_finite_ratio") == 1.0
            and policy.get("candidate_q_maximum_spread", 0.0) > 0.0
            and policy.get("candidate_q_nonconstant_fraction", 0.0) > 0.0
            and policy.get("candidate_q_source")
            == "discounted_official_simulator_task_predicate_return"
            and auxiliary.get("critic", 0.0) > 0.0
            and auxiliary.get("regret", 0.0) > 0.0
            and auxiliary.get("contrastive", 0.0) > 0.0
            and policy.get("influence_gradient_norm", 0.0) > 0.0
            and 0.0 < policy.get("eval_selected_ratio", 0.0) < 1.0
            and environment.get("status") == "ok"
            and environment.get("finite_observation") is True
            and environment.get("robot_state_changed") is True
        )
        if not valid:
            raise RuntimeError(f"invalid measured simulator result for {name}")
        measured_simulators[name] = {
            "status": "ok",
            "candidate_q_finite_ratio": policy["candidate_q_finite_ratio"],
            "candidate_q_maximum_spread": policy[
                "candidate_q_maximum_spread"
            ],
            "candidate_q_nonconstant_fraction": policy[
                "candidate_q_nonconstant_fraction"
            ],
            "critic_loss": auxiliary["critic"],
            "regret_loss": auxiliary["regret"],
            "contrastive_loss": auxiliary["contrastive"],
            "reward": environment["reward"],
            "progress_before": environment["progress_before"],
            "progress_after": environment["progress_after"],
            "policy_report": str(policy_path),
            "environment_report": str(environment_path),
        }

    named_results = {}
    for name, result in offline["datasets"].items():
        policy_path = Path(result["policy_report"])
        evaluation_path = Path(result["evaluation_report"])
        policy = json.loads(policy_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        if (
            policy.get("status") != "policy_ok"
            or evaluation.get("status") != "ok"
            or not evaluation.get("finite_action")
            or not math.isfinite(policy.get("train_total_loss", math.nan))
            or policy.get("influence_gradient_norm", 0.0) <= 0.0
            or not 0.0 < policy.get("eval_selected_ratio", 0.0) < 1.0
        ):
            raise RuntimeError(f"invalid offline smoke result for {name}")
        named_results[name] = {
            "status": "ok",
            "source_repo": result["repo"],
            "train_total_loss": policy["train_total_loss"],
            "influence_gradient_norm": policy["influence_gradient_norm"],
            "eval_selected_ratio": policy["eval_selected_ratio"],
            "finite_action": evaluation["finite_action"],
            "policy_report": str(policy_path),
            "evaluation_report": str(evaluation_path),
            "counterfactual_q": (
                "unavailable for offline recordings; critic/regret explicitly masked"
            ),
        }

    ood_collect_path = args.outputs / "robocasa_ood_collect.json"
    ood_policy_path = args.outputs / "robocasa_ood_policy.json"
    ood_environment_path = args.outputs / "robocasa_ood_env.json"
    ood_collect = json.loads(ood_collect_path.read_text())
    ood_policy = json.loads(ood_policy_path.read_text())
    ood_environment = json.loads(ood_environment_path.read_text())
    collected = ood_collect.get("variants", [])
    evaluated = ood_environment.get("variants", [])
    selected_ratios = ood_policy.get("selected_ratio", [])
    expected_ood_variants = [
        "baseline",
        "texture_shift",
        "lighting_shift",
        "background_motion",
        "unseen_layout",
        "contact_counterfactual",
    ]
    visible_shift_variants = {"texture_shift", "lighting_shift", "unseen_layout"}
    if (
        ood_collect.get("status") != "ok"
        or ood_policy.get("status") != "ok"
        or ood_environment.get("status") != "ok"
        or len(collected) != 6
        or len(evaluated) != 6
        or ood_policy.get("finite_actions") is not True
        or len(selected_ratios) != 6
        or [item.get("variant") for item in collected] != expected_ood_variants
        or [item.get("variant") for item in evaluated] != expected_ood_variants
        or not all(0.0 < ratio < 1.0 for ratio in selected_ratios)
        or not all(item.get("finite_observation") for item in collected)
        or not all(
            all(change.get("changed") for change in item["environment_changes"])
            for item in collected[1:]
        )
        or not all(
            item.get("primary_pixel_mae_from_baseline", 0.0) > 0.0
            for item in collected
            if item.get("variant") in visible_shift_variants
        )
        or not all(
            item.get("finite_observation")
            and item.get("robot_state_changed")
            and math.isfinite(item.get("reward", math.nan))
            and math.isfinite(item.get("progress_after", math.nan))
            for item in evaluated
        )
    ):
        raise RuntimeError("invalid end-to-end RoboCasa OOD result")

    report = {
        "status": "ok",
        "public_dataset_count": 17,
        "coverage": {
            "LIBERO": "full DIWA plus six ablation execution paths",
            "CALVIN": "measured candidate-Q training and official environment step",
            "RoboCasa": (
                "measured candidate-Q training, official environment step, "
                "and six end-to-end OOD variants"
            ),
            "RoboTwin": "measured candidate-Q training and official environment step",
            "DROID_and_OXE": "DROID plus all 12 configured OXE sources",
        },
        "simulator_report": str(simulator_path),
        "offline_suite_report": str(offline_path),
        "measured_simulators": measured_simulators,
        "robocasa_ood": {
            "status": "ok",
            "variant_count": 6,
            "variants": [item["variant"] for item in evaluated],
            "ood_success_rate": ood_environment["ood_success_rate"],
            "ood_progress_delta_mean": ood_environment[
                "ood_progress_delta_mean"
            ],
            "collection_report": str(ood_collect_path),
            "policy_report": str(ood_policy_path),
            "environment_report": str(ood_environment_path),
        },
        "offline_datasets": named_results,
        "excluded": {
            "real": (
                "user-supplied private loader selected by --real_dataset_names; "
                "not a named public dataset"
            )
        },
        "claim": (
            "integration training/inference coverage only; random-weight smoke "
            "runs are not benchmark success-rate reproduction"
        ),
    }
    output = args.outputs / "diwa_all_datasets_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_ALL_DATASETS_SMOKE_OK")


if __name__ == "__main__":
    main()
