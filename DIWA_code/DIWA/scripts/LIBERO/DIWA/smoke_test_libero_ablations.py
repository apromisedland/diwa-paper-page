"""Smoke-test the five DIWA ablations on a real LIBERO environment.

This is an execution and tensor-contract test, not a statistically meaningful
ablation study. All variants share a compact random-weight model and the same
two simulator initial states. Training variants each perform a dedicated
forward/backward/optimizer step; inference variants verify their compute
selection contract. The resulting losses are not compared because the compact
model uses random weights and variants run sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import numpy as np
import torch
import torch.nn.functional as F

from scripts.LIBERO.DIWA import smoke_test_libero as base_smoke


ALL_AUXILIARY = (
    "proposal",
    "mask",
    "future",
    "influence",
    "budget",
    "critic",
    "progress",
    "regret",
    "contrastive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least two")
    return args


def collect_real_libero_batch(args: argparse.Namespace):
    libero_path = args.libero_path.expanduser().resolve()
    benchmark, offscreen_env = base_smoke.import_libero(libero_path)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task id must be in [0, {suite.n_tasks})")
    task = suite.get_task(args.task_id)
    bddl_file = (
        libero_path
        / "libero"
        / "libero"
        / "bddl_files"
        / task.problem_folder
        / task.bddl_file
    )
    init_file = (
        libero_path
        / "libero"
        / "libero"
        / "init_files"
        / task.problem_folder
        / task.init_states_file
    )
    env = offscreen_env(
        bddl_file_name=str(bddl_file),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        render_gpu_device_id=0,
    )
    sequence_length = 3
    action_steps = 2
    try:
        init_states = torch.load(init_file, map_location="cpu")
        batch = base_smoke.collect_batch(
            env,
            init_states,
            args.batch_size,
            sequence_length,
            action_steps,
        )
    finally:
        env.close()
    return task, batch, sequence_length, action_steps


def build_model(device: torch.device, action_steps: int):
    from models.dreamvla_model import DreamVLA

    model = DreamVLA(
        finetune_type="libero_finetune",
        clip_device=str(device),
        vit_checkpoint_path=None,
        allow_random_vision_encoder=True,
        sequence_length=2,
        num_resampler_query=4,
        action_pred_steps=action_steps,
        transformer_layers=2,
        hidden_dim=128,
        transformer_heads=4,
        phase="finetune",
        gripper_width=True,
        use_dit_head=False,
        attn_implementation="eager",
        use_diwa=True,
        diwa_horizon=1,
        diwa_num_slots=4,
        diwa_world_layers=1,
        diwa_fusion_layers=1,
        diwa_counterfactual_samples=2,
        diwa_eval_budget_ratio=0.5,
        diwa_adaptive_budget=True,
        diwa_minimum_budget_ratio=0.25,
        diwa_regret_candidates=4,
        diwa_require_supervision=True,
    ).to(device)
    model._init_model_type()
    return model


def prepare_inputs(
    batch,
    task,
    model,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
):
    import clip

    primary = base_smoke.preprocess_images(
        batch["primary"], model.image_processor
    ).to(device)
    wrist = base_smoke.preprocess_images(
        batch["wrist"], model.image_processor
    ).to(device)
    state = torch.from_numpy(batch["state"]).to(device)
    tokens = clip.tokenize([task.language] * batch_size).to(device)
    text = tokens[:, None].expand(-1, sequence_length, -1).contiguous()
    action_labels = torch.from_numpy(batch["actions"]).to(device)
    action_labels[..., 6] = (action_labels[..., 6] + 1.0) / 2.0
    candidate_actions = torch.from_numpy(batch["candidate_actions"]).to(
        device
    )
    candidate_actions[..., 6] = (candidate_actions[..., 6] + 1.0) / 2.0
    supervision = {
        "rewards": torch.from_numpy(batch["rewards"]).float().to(device),
        "dones": torch.from_numpy(batch["dones"]).bool().to(device),
        "progress": torch.from_numpy(batch["progress"]).float().to(device),
        "valid": torch.ones(batch_size, 2, dtype=torch.bool, device=device),
        "observation_valid": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        "candidate_actions": candidate_actions,
        "candidate_q_values": torch.from_numpy(
            batch["candidate_q_values"]
        ).float().to(device),
        "task_ids": torch.zeros(
            batch_size, dtype=torch.long, device=device
        ),
        "episode_ids": torch.arange(
            batch_size, dtype=torch.long, device=device
        ),
    }
    return {
        "primary": primary,
        "wrist": wrist,
        "state": state,
        "text": text,
        "action_labels": action_labels,
        "supervision": supervision,
    }


def action_loss(output, labels):
    return F.smooth_l1_loss(
        output["arm_action"].float(), labels[..., :6].float()
    ) + 0.01 * F.binary_cross_entropy(
        output["gripper_action"].float().clamp(1e-6, 1.0 - 1e-6),
        labels[..., 6:].float(),
    )


def train_variant(
    name,
    model,
    optimizer,
    inputs,
    *,
    included_auxiliary=ALL_AUXILIARY,
    enable_influence=True,
    enable_swaps=True,
    force_dense=False,
):
    core = model.diwa_core
    previous_influence = core.enable_influence_estimator
    previous_swaps = core.enable_counterfactual_swaps
    core.enable_influence_estimator = enable_influence
    core.enable_counterfactual_swaps = enable_swaps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        image_primary=inputs["primary"],
        image_wrist=inputs["wrist"],
        state=inputs["state"],
        text_token=inputs["text"],
        action_label=inputs["action_labels"],
        mode="train",
        diwa_budget_ratio=0.5,
        diwa_force_dense_budget=force_dense,
        diwa_teacher_forcing_ratio=0.5,
        diwa_supervision=inputs["supervision"],
    )
    primary_loss = action_loss(output, inputs["action_labels"])
    auxiliary_loss = sum(
        output["aux_losses"][key] for key in included_auxiliary
    )
    total = primary_loss + auxiliary_loss
    if not torch.isfinite(total):
        raise RuntimeError(f"{name} produced non-finite training loss")
    total.backward()
    mask_gradient = core.counterfactual_token.grad
    if "mask" in included_auxiliary and (
        mask_gradient is None or not torch.isfinite(mask_gradient).all() or mask_gradient.norm() <= 0
    ):
        raise RuntimeError(f"{name} did not train the mask token")
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} produced a non-finite gradient")
    influence_gradient = core.influence_estimator[-1].weight.grad
    influence_gradient_norm = (
        float(influence_gradient.norm())
        if influence_gradient is not None
        else 0.0
    )
    if enable_influence and influence_gradient is None:
        raise RuntimeError(f"{name} did not train the influence estimator")
    if not enable_influence and influence_gradient_norm != 0.0:
        raise RuntimeError(f"{name} unexpectedly trained influence estimator")
    swap_mask = output["counterfactual_swap_mask"]
    swap_count = int(swap_mask.sum()) if swap_mask is not None else 0
    if enable_swaps and swap_count == 0:
        raise RuntimeError(f"{name} did not execute a counterfactual swap")
    if not enable_swaps and swap_count != 0:
        raise RuntimeError(f"{name} unexpectedly executed a swap")
    optimizer.step()
    core.update_target_networks(0.01)
    core.enable_influence_estimator = previous_influence
    core.enable_counterfactual_swaps = previous_swaps
    return {
        "status": "ok",
        "kind": "train_update",
        "included_auxiliary": list(included_auxiliary),
        "excluded_auxiliary": [
            key for key in ALL_AUXILIARY if key not in included_auxiliary
        ],
        "influence_enabled": enable_influence,
        "counterfactual_swaps_enabled": enable_swaps,
        "counterfactual_swap_count": swap_count,
        "force_dense_budget": force_dense,
        "total_loss": float(total.detach()),
        "action_loss": float(primary_loss.detach()),
        "influence_gradient_norm": influence_gradient_norm,
        "mask_gradient_norm": float(mask_gradient.norm()) if mask_gradient is not None else 0.0,
        "selected_ratio": float(output["selected_ratio"]),
        "aux_losses": {
            key: float(output["aux_losses"][key].detach())
            for key in ALL_AUXILIARY
        },
    }


def eval_variant(
    name,
    model,
    inputs,
    device: torch.device,
    *,
    budget_ratio: float,
    enable_influence: bool,
    adaptive_budget: bool,
    expected_dense: bool,
):
    core = model.diwa_core
    previous_ratio = model.diwa_eval_budget_ratio
    previous_influence = core.enable_influence_estimator
    previous_adaptive = core.adaptive_budget
    model.diwa_eval_budget_ratio = budget_ratio
    core.enable_influence_estimator = enable_influence
    core.adaptive_budget = adaptive_budget
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(
            image_primary=inputs["primary"][:, :2],
            image_wrist=inputs["wrist"][:, :2],
            state=inputs["state"][:, :2],
            text_token=inputs["text"][:, :2],
            action=torch.zeros(
                inputs["state"].shape[0], 2, 7, device=device
            ),
            mode="test",
            diwa_current_step=1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    expanded = output["expanded_tokens"].detach().cpu()
    available = int(output["available_tokens"])
    is_dense = bool((expanded == available).all())
    if is_dense != expected_dense:
        raise RuntimeError(
            f"{name} dense={is_dense}, expected dense={expected_dense}"
        )
    if not torch.isfinite(output["arm_action"]).all():
        raise RuntimeError(f"{name} produced non-finite actions")
    if not enable_influence and not torch.equal(
        output["influence_logits"],
        torch.zeros_like(output["influence_logits"]),
    ):
        raise RuntimeError(f"{name} did not use uniform influence scores")
    report = {
        "status": "ok",
        "kind": "inference",
        "budget_ratio": budget_ratio,
        "influence_enabled": enable_influence,
        "adaptive_budget": adaptive_budget,
        "selected_ratio": float(output["selected_ratio"]),
        "expanded_tokens": expanded.tolist(),
        "available_tokens": available,
        "dense": is_dense,
        "latency_ms": latency_ms,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    model.diwa_eval_budget_ratio = previous_ratio
    core.enable_influence_estimator = previous_influence
    core.adaptive_budget = previous_adaptive
    return report


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    task, batch, sequence_length, action_steps = collect_real_libero_batch(args)
    model = build_model(device, action_steps)
    inputs = prepare_inputs(
        batch,
        task,
        model,
        device,
        args.batch_size,
        sequence_length,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )

    variants = {}
    variants["full_diwa"] = train_variant(
        "full_diwa", model, optimizer, inputs
    )
    variants["without_influence_estimator"] = train_variant(
        "without_influence_estimator",
        model,
        optimizer,
        inputs,
        included_auxiliary=tuple(
            key for key in ALL_AUXILIARY if key not in {"influence", "budget"}
        ),
        enable_influence=False,
        force_dense=True,
    )
    variants["without_counterfactual_swap"] = train_variant(
        "without_counterfactual_swap",
        model,
        optimizer,
        inputs,
        enable_swaps=False,
    )
    variants["without_regret_geometry"] = train_variant(
        "without_regret_geometry",
        model,
        optimizer,
        inputs,
        included_auxiliary=tuple(
            key for key in ALL_AUXILIARY if key not in {"regret", "contrastive"}
        ),
    )
    variants["without_sparse_imagination"] = eval_variant(
        "without_sparse_imagination",
        model,
        inputs,
        device,
        budget_ratio=1.0,
        enable_influence=True,
        adaptive_budget=False,
        expected_dense=True,
    )
    variants["uniform_topk"] = eval_variant(
        "uniform_topk",
        model,
        inputs,
        device,
        budget_ratio=0.5,
        enable_influence=False,
        adaptive_budget=False,
        expected_dense=False,
    )

    report = {
        "status": "ok",
        "purpose": "execution smoke; not a performance comparison",
        "dataset": "LIBERO_SPATIAL",
        "task_id": args.task_id,
        "task": task.name,
        "language": task.language,
        "device": str(device),
        "batch_size": args.batch_size,
        "candidate_q_values": batch["candidate_q_values"].tolist(),
        "variants": variants,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DIWA_LIBERO_ABLATION_SMOKE_OK")


if __name__ == "__main__":
    main()
