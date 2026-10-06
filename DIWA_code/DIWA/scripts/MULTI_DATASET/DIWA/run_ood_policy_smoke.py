"""Run one trained DreamVLA-DIWA checkpoint on generated OOD observations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MULTI_DATASET.DIWA.run_exported_smoke import (  # noqa: E402
    build_model,
    preprocess,
    validate_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--train-batch", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    training = np.load(args.train_batch, allow_pickle=False)
    contract = validate_batch(training)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["contract"]["action_dim"] != contract["action_dim"]:
        raise RuntimeError("checkpoint and training action contracts differ")
    model = build_model(contract, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    archive = np.load(args.observations, allow_pickle=False)
    primary_np = archive["primary"]
    wrist_np = archive["wrist"]
    state_np = archive["state"]
    variants = archive["variants"].tolist()
    if primary_np.ndim != 5 or primary_np.shape[0] != len(variants):
        raise RuntimeError("OOD primary images must be [variant,time,H,W,3]")
    if primary_np.shape[:2] != wrist_np.shape[:2] or state_np.shape[:2] != primary_np.shape[:2]:
        raise RuntimeError("OOD observation modalities are not aligned")

    import clip

    primary = preprocess(primary_np, model.image_processor).to(device)
    wrist = preprocess(wrist_np, model.image_processor).to(device)
    state = torch.from_numpy(state_np).float().to(device)
    language = str(archive["language"])
    text = clip.tokenize([language] * len(variants)).to(device)
    text = text[:, None].expand(-1, primary.shape[1], -1).contiguous()
    sequence = int(contract["sequence_length"])
    if primary.shape[1] < sequence:
        raise RuntimeError("OOD archive is shorter than the policy sequence")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            image_primary=primary[:, :sequence],
            image_wrist=wrist[:, :sequence],
            state=state[:, :sequence],
            text_token=text[:, :sequence],
            action=torch.zeros(
                len(variants),
                sequence,
                int(contract["action_dim"]),
                device=device,
            ),
            mode="test",
            diwa_current_step=sequence - 1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0

    prediction = torch.cat(
        (output["arm_action"], output["gripper_action"]), dim=-1
    )
    expected_tail = (
        int(contract["action_steps"]),
        int(contract["action_dim"]),
    )
    if (
        prediction.shape[0] != len(variants)
        or tuple(prediction.shape[-2:]) != expected_tail
    ):
        raise RuntimeError(f"unexpected OOD policy output shape {tuple(prediction.shape)}")
    # Test mode evaluates only diwa_current_step and therefore has a singleton
    # policy-time axis even when the observation context contains T > 1.
    prediction = prediction.reshape(len(variants), -1, *expected_tail)[:, -1]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("OOD policy produced non-finite actions")
    selected = output["selected_mask"].reshape(len(variants), -1)
    selected_ratio = selected.float().mean(dim=-1)
    if not torch.all((selected_ratio > 0) & (selected_ratio < 1)):
        raise RuntimeError("OOD policy did not perform sparse inference")

    actions = prediction.float().cpu().numpy()
    baseline = actions[0]
    action_shift = np.sqrt(np.mean((actions - baseline[None]) ** 2, axis=(1, 2)))
    influence = output["influence_probabilities"].reshape(len(variants), -1)
    args.actions.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.actions, actions)
    report = {
        "status": "ok",
        "evaluation_kind": "trained_policy_on_generated_ood_environments",
        "checkpoint": str(args.checkpoint),
        "variants": variants,
        "action_shape": list(actions.shape),
        "finite_actions": True,
        "action_rmse_from_baseline": action_shift.tolist(),
        "selected_ratio": selected_ratio.cpu().tolist(),
        "influence_probability_mean": influence.mean(dim=-1).cpu().tolist(),
        "latency_ms_total": latency_ms,
        "latency_ms_per_variant": latency_ms / len(variants),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_OOD_POLICY_SMOKE_OK")


if __name__ == "__main__":
    main()
