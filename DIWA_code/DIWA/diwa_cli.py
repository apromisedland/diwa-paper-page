"""Train, resume, inspect, and run DIWA using precomputed causal features."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.diwa.feature_data import (
    FeatureWindowDataset,
    PairedBatchSampler,
    load_observations,
)
from models.diwa.feature_policy import FeatureDIWAPolicy, load_config
from models.diwa.optimization import make_lr_scheduler, accumulation_divisor
from utils.checkpoint_utils import (
    DIWA_METHOD_REVISION,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


ROOT = Path(__file__).resolve().parent


def to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def environment():
    import timm

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "timm": timm.__version__,
        "platform": platform.system(),
        "cuda_available": torch.cuda.is_available(),
    }


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("feature-policy checkpoint must be a mapping")
    if checkpoint.get("format_version") not in (1, 2, 3):
        raise ValueError("unsupported feature-policy checkpoint format")
    for name in ("config", "model"):
        if name not in checkpoint:
            raise ValueError(f"feature-policy checkpoint is missing {name}")
    return checkpoint


def load_policy(path, device="cpu"):
    checkpoint = read_checkpoint(path)
    model = FeatureDIWAPolicy(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def fit(
    config,
    manifest,
    output,
    device="cpu",
    *,
    resume=None,
    allow_synthetic=False,
    epochs_to_run=None,
):
    """Single-device feature training; full DDP image training is in train.py.

    Checkpoints are atomic and resume at epoch boundaries, including the
    optimizer, LR schedule, target networks, influence scale, and RNG states.
    """
    output = Path(output)
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists() and resume is None:
        raise FileExistsError(
            f"{checkpoint_path} exists; use --resume or a new output directory"
        )
    dataset = FeatureWindowDataset(manifest, config, allow_synthetic=allow_synthetic)
    t = config["training"]
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t["seed"])
    model = FeatureDIWAPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )
    batch_count = len(PairedBatchSampler(dataset, t["batch_size"], t["seed"]))
    scheduler = make_lr_scheduler(
        optimizer, batch_count, t["epochs"], t["warmup_epochs"], t["gradient_accumulation"]
    )
    epoch_start = micro_step = optimizer_step = 0
    if resume is not None:
        state = read_checkpoint(resume)
        if state.get("format_version") != 3 or state.get("method_revision") != DIWA_METHOD_REVISION:
            raise ValueError("resume checkpoint uses an earlier DIWA training method")
        required_resume_state = {
            "optimizer",
            "scheduler",
            "next_epoch",
            "micro_step",
            "optimizer_step",
            "dataset_fingerprint",
        }
        missing = sorted(required_resume_state.difference(state))
        if missing:
            raise ValueError(f"resume checkpoint is missing {missing}")
        if state["config"] != config:
            raise ValueError("resume configuration differs from the checkpoint")
        if state["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError("resume dataset differs from the checkpoint")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch_start, micro_step, optimizer_step = (
            state["next_epoch"],
            state["micro_step"],
            state["optimizer_step"],
        )
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("resume checkpoint is missing complete RNG state")
        restore_rng_state(rng_state)
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "config": config,
        "environment": environment(),
        "device": str(device),
        "dataset_fingerprint": dataset.fingerprint,
        "data_kind": dataset.manifest["data_kind"],
        "episodes": len(dataset.paths),
        "windows": len(dataset),
        "training_mode": "single_device_precomputed_features",
        "effective_batch_size": t["batch_size"] * t["gradient_accumulation"],
        "schedule_unit": "minibatch",
        "checkpoint_boundary": "epoch",
        "method_revision": DIWA_METHOD_REVISION,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    last_losses, influence_gradient, last_dense_queries = {}, 0.0, 0
    mask_gradient = 0.0
    initial_mask = model.core.counterfactual_token.detach().clone()
    epoch_end = (
        t["epochs"]
        if epochs_to_run is None
        else min(t["epochs"], epoch_start + epochs_to_run)
    )
    with (output / "metrics.jsonl").open("a" if resume else "w") as log:
        for epoch in range(epoch_start, epoch_end):
            sampler = PairedBatchSampler(dataset, t["batch_size"], t["seed"], epoch)
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=0,
                generator=torch.Generator().manual_seed(t["seed"] + epoch),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loader):
                group_size = accumulation_divisor(
                    batch_index, len(loader), t["gradient_accumulation"]
                )
                loss, losses, result = model(to_device(batch, device), micro_step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at minibatch {micro_step}"
                    )
                (loss / group_size).backward()
                update = (batch_index + 1) % t[
                    "gradient_accumulation"
                ] == 0 or batch_index + 1 == len(loader)
                if update:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), t["max_grad_norm"], error_if_nonfinite=True
                    )
                    gradient = model.core.influence_estimator[-1].weight.grad
                    influence_gradient = (
                        0.0 if gradient is None else float(gradient.norm())
                    )
                    mask_grad = model.core.counterfactual_token.grad
                    mask_gradient = 0.0 if mask_grad is None else float(mask_grad.norm())
                    optimizer.step()
                    scheduler.step()
                    model.core.update_target_networks(t["target_tau"])
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                micro_step += 1
                last_dense_queries = result.world_tokens.shape[1]
                last_losses = {
                    name: float(value.detach()) for name, value in losses.items()
                }
                record = dict(
                    epoch=epoch,
                    micro_step=micro_step,
                    optimizer_step=optimizer_step,
                    total_loss=float(loss.detach()),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    **last_losses,
                )
                log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            state = {
                "format_version": 3,
                "method_revision": DIWA_METHOD_REVISION,
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "rng_state": capture_rng_state(),
                "dataset_fingerprint": dataset.fingerprint,
                "data_kind": dataset.manifest["data_kind"],
                "environment": environment(),
            }
            atomic_torch_save(state, checkpoint_path)
            print(
                f"epoch {epoch + 1}/{t['epochs']}  loss={record['total_loss']:.6f}  updates={optimizer_step}",
                flush=True,
            )
    report = {
        "checkpoint": str(checkpoint_path),
        "data_kind": dataset.manifest["data_kind"],
        "micro_steps": micro_step,
        "optimizer_steps": optimizer_step,
        "losses": last_losses,
        "last_influence_gradient_norm": influence_gradient,
        "last_mask_gradient_norm": mask_gradient,
        "mask_parameter_max_change": float(
            (model.core.counterfactual_token.detach() - initial_mask).abs().max()
        ),
        "training_expanded_queries": last_dense_queries,
        "environment": environment(),
    }
    return report, model


def smoke(output):
    from examples.synthetic_fixture import create_fixture

    output = Path(output)
    config = load_config(ROOT / "configs/diwa_cpu.json")
    manifest, observations = create_fixture(output / "fixture", config)
    report, trained = fit(config, manifest, output / "training", allow_synthetic=True)
    inputs = load_observations(observations, config["core"]["hidden_dim"])
    noise = torch.randn(
        1, config["core"]["action_pred_steps"], config["core"]["action_dim"]
    )
    first = trained.eval().predict(inputs, initial_noise=noise)
    restored, _ = load_policy(report["checkpoint"])
    second = restored.predict(inputs, initial_noise=noise)
    difference = float((first["actions"] - second["actions"]).abs().max())
    if difference != 0:
        raise AssertionError("checkpoint reload changed deterministic CPU predictions")
    if report["last_mask_gradient_norm"] <= 0 or report["mask_parameter_max_change"] <= 0:
        raise AssertionError("the mask calibration objective did not train the mask token")
    if (
        int(second["expanded_queries"]) > 12
        or report["training_expanded_queries"] != 48
    ):
        raise AssertionError("select-before-decode query budget is incorrect")
    if report["last_influence_gradient_norm"] <= 0:
        raise AssertionError("influence scorer did not receive a training gradient")
    report.update(
        scope="synthetic_tensor_functional_test_only",
        checkpoint_reload_max_difference=difference,
        inference_expanded_queries=int(second["expanded_queries"]),
        selected_queries_per_state=second["selected_valid_mask"].sum(-1).tolist(),
        action_shape=list(second["actions"].shape),
        paper_success_rates_reproduced=False,
    )
    np.savez_compressed(
        output / "prediction.npz",
        **{name: value.cpu().numpy() for name, value in second.items()},
    )
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads", type=int, default=2, help="CPU computation threads"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser(
        "smoke", help="CPU functional check using clearly marked synthetic fixtures"
    )
    smoke_parser.add_argument("--output", type=Path, default=Path("runs/cpu_smoke"))
    for name in ("train", "validate-data"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--config", type=Path)
        if name == "train":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--device", default="cpu")
            command.add_argument("--resume", type=Path)
            command.add_argument("--seed", type=int)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ with observed context/visual features only",
    )
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "smoke":
        result = smoke(args.output)
    elif args.command in ("train", "validate-data"):
        if args.command == "train" and args.resume is not None and args.config is None:
            config = read_checkpoint(args.resume)["config"]
        else:
            config = load_config(args.config or ROOT / "configs/diwa_paper.json")
        if args.command == "train":
            if args.seed is not None:
                config["training"]["seed"] = args.seed
            result, _ = fit(
                config, args.manifest, args.output, args.device, resume=args.resume
            )
        else:
            dataset = FeatureWindowDataset(args.manifest, config)
            result = {
                "episodes": len(dataset.paths),
                "windows": len(dataset),
                "fingerprint": dataset.fingerprint,
                "status": "valid",
            }
    else:
        model, _ = load_policy(args.checkpoint, args.device)
        inputs = to_device(
            load_observations(args.input, model.core.hidden_dim), args.device
        )
        torch.manual_seed(args.seed)
        predictions = model.predict(inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            **{name: value.cpu().numpy() for name, value in predictions.items()},
        )
        result = {
            "output": str(args.output),
            "expanded_queries": int(predictions["expanded_queries"]),
            "action_shape": list(predictions["actions"].shape),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        sys.exit(f"DIWA: {error}")
