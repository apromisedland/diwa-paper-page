"""Run DROID and every OXE dataset configured by DreamVLA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Support both direct execution and package-style imports used by tests/tools.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot_dataset_adapter import evaluate_action, export_batch, materialize


DATASETS = {
    "DROID": "lerobot/droid_1.0.1",
    "OXE_berkeley_autolab_ur5": "FedorX8/berkeley_autolab_ur5_lerobot",
    "OXE_jaco_play": "FedorX8/jaco_play_lerobot",
    "OXE_iamlab_cmu_pickup_insert": "FedorX8/iamlab_cmu_pickup_insert_lerobot",
    "OXE_viola": "FedorX8/viola_lerobot",
    "OXE_stanford_hydra_dataset": "FedorX8/stanford_hydra_dataset_lerobot",
    "OXE_berkeley_fanuc_manipulation": "FedorX8/berkeley_fanuc_manipulation_lerobot",
    "OXE_austin_buds_dataset": "FedorX8/austin_buds_dataset_lerobot",
    "OXE_utaustin_mutex": "FedorX8/utaustin_mutex_lerobot",
    "OXE_taco_play": "FedorX8/taco_play_lerobot",
    "OXE_austin_sailor_dataset": "FedorX8/austin_sailor_dataset_lerobot",
    "OXE_austin_sirius_dataset": "FedorX8/austin_sirius_dataset_lerobot",
    "OXE_furniture_bench_dataset": "FedorX8/furniture_bench_dataset_lerobot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="*", choices=tuple(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    selected = args.only or list(DATASETS)
    results = {}
    for dataset_name in selected:
        repo = DATASETS[dataset_name]
        slug = dataset_name.lower()
        dataset_dir = args.cache / slug
        export_path = args.outputs / f"{slug}_smoke_batch.npz"
        action_path = args.outputs / f"{slug}_eval_action.npy"
        policy_report = args.outputs / f"{slug}_policy_smoke.json"
        evaluation_report = args.outputs / f"{slug}_offline_eval_smoke.json"
        if args.resume and policy_report.is_file() and evaluation_report.is_file():
            policy = json.loads(policy_report.read_text())
            evaluation = json.loads(evaluation_report.read_text())
            if policy.get("status") == "policy_ok" and evaluation.get("status") == "ok":
                results[dataset_name] = {
                    "repo": repo,
                    "policy_report": str(policy_report),
                    "evaluation_report": str(evaluation_report),
                    "policy_status": policy["status"],
                    "evaluation_status": evaluation["status"],
                    "resumed": True,
                }
                continue

        print(f"[{dataset_name}] materializing parquet metadata", flush=True)
        materialize(repo, dataset_dir, args.endpoint)
        print(f"[{dataset_name}] exporting real two-episode batch", flush=True)
        export_batch(
            repo,
            dataset_name,
            dataset_dir,
            export_path,
            args.endpoint,
        )
        print(f"[{dataset_name}] running DIWA train update and inference", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_exported_smoke.py")),
                "--input",
                str(export_path),
                "--action-output",
                str(action_path),
                "--report",
                str(policy_report),
                "--device",
                args.device,
            ],
            cwd=project_root,
            check=True,
        )
        evaluate_action(export_path, action_path, evaluation_report)
        policy = json.loads(policy_report.read_text())
        evaluation = json.loads(evaluation_report.read_text())
        results[dataset_name] = {
            "repo": repo,
            "policy_report": str(policy_report),
            "evaluation_report": str(evaluation_report),
            "policy_status": policy["status"],
            "evaluation_status": evaluation["status"],
            "train_total_loss": policy["train_total_loss"],
            "influence_gradient_norm": policy["influence_gradient_norm"],
            "eval_selected_ratio": policy["eval_selected_ratio"],
            "finite_action": evaluation["finite_action"],
            "resumed": False,
        }
        partial = {
            "status": "running",
            "completed": len(results),
            "requested": len(selected),
            "datasets": results,
        }
        args.outputs.mkdir(parents=True, exist_ok=True)
        (args.outputs / "diwa_offline_dataset_suite.json").write_text(
            json.dumps(partial, indent=2) + "\n"
        )

    summary = {
        "status": "ok",
        "completed": len(results),
        "requested": len(selected),
        "scope": "DROID plus all 12 OXE datasets hard-coded by get_oxe_dataset",
        "datasets": results,
    }
    summary_path = args.outputs / "diwa_offline_dataset_suite.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("DIWA_ALL_OFFLINE_DATASETS_SMOKE_OK")


if __name__ == "__main__":
    main()
