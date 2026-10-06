"""Train and evaluate DreamVLA-DIWA on a tiny exported simulator batch.

Simulator-specific collectors run in their official dependency environments
and write the compact NPZ contract consumed here. Keeping the policy in the
main DreamVLA environment avoids dependency conflicts between MuJoCo, SAPIEN,
PyBullet and their pinned Python versions. The resulting ``eval_action`` is
written back for a simulator-specific environment-step check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MULTI_DATASET.DIWA.measured_supervision import (  # noqa: E402
    validate_measured_candidate_q,
)
from utils.checkpoint_utils import atomic_torch_save  # noqa: E402
from utils.diwa_schema import normalize_candidate_rule_ids  # noqa: E402


EXPECTED_CANDIDATE_RULES = {
    "CALVIN": (
        "demonstration",
        "no_cartesian_motion_keep_gripper",
        "invert_gripper",
        "signed_cartesian_counter_action",
    ),
    "RoboCasa": (
        "demonstration",
        "invert_gripper_close",
        "invert_control_mode",
        "invert_continuous_action",
    ),
    "RoboTwin": (
        "demonstration",
        "invert_left_gripper",
        "invert_right_gripper",
        "invert_both_arm_actions",
    ),
}
OFFLINE_CANDIDATE_RULES = (
    "demonstration",
    "invert_gripper",
    "invert_continuous_action",
    "suppress_translation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--action-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-output", type=Path)
    return parser.parse_args()


def scalar(batch, name: str):
    value = batch[name]
    return value.item() if np.asarray(value).ndim == 0 else value.tolist()


def active_auxiliary_losses(contract):
    """The offline path trains policy/future features without invented Q labels."""
    names = ("proposal", "mask", "future", "influence", "budget")
    if contract["decision_supervision_active"]:
        names += ("critic", "progress", "regret", "contrastive")
    return names


def validate_batch(batch) -> dict[str, object]:
    required = {
        "primary",
        "wrist",
        "state",
        "actions",
        "rewards",
        "dones",
        "progress",
        "candidate_actions",
        "candidate_q_values",
        "candidate_rule_ids",
        "dataset",
        "task",
        "language",
        "continuous_action_dim",
        "state_arm_dim",
        "gripper_width",
    }
    missing = required.difference(batch.files)
    if missing:
        raise ValueError(f"export is missing fields: {sorted(missing)}")
    primary = batch["primary"]
    wrist = batch["wrist"]
    state = batch["state"]
    actions = batch["actions"]
    candidates = batch["candidate_actions"]
    candidate_q = batch["candidate_q_values"]
    candidate_rule_ids = normalize_candidate_rule_ids(
        batch["candidate_rule_ids"]
    )
    if primary.ndim != 5 or primary.shape[-1] != 3:
        raise ValueError("primary must have shape [batch,time,height,width,3]")
    if wrist.shape[:2] != primary.shape[:2] or wrist.shape[-1] != 3:
        raise ValueError("wrist camera batch/time dimensions must match primary")
    if state.ndim != 3 or state.shape[:2] != primary.shape[:2]:
        raise ValueError("state must have shape [batch,time,state_dim]")
    if actions.ndim != 4 or actions.shape[:2] != (
        primary.shape[0],
        primary.shape[1] - 1,
    ):
        raise ValueError("actions must cover every non-future observation")
    if candidates.ndim != 5 or candidates.shape[:2] != actions.shape[:2]:
        raise ValueError("candidate action batch/time dimensions are invalid")
    if candidates.shape[-2:] != actions.shape[-2:]:
        raise ValueError("candidate action chunk dimensions must match labels")
    if candidate_q.shape != candidates.shape[:3]:
        raise ValueError(
            "candidate_q_values must have shape [batch,time,candidate]"
        )
    if len(candidate_rule_ids) != candidates.shape[2]:
        raise ValueError(
            "candidate_rule_ids length must match the candidate action axis"
        )
    if primary.shape[0] < 2:
        raise ValueError("at least two episodes are required for token swaps")
    continuous = int(scalar(batch, "continuous_action_dim"))
    action_dim = actions.shape[-1]
    state_arm = int(scalar(batch, "state_arm_dim"))
    state_gripper = state.shape[-1] - state_arm
    if not 0 < continuous < action_dim:
        raise ValueError("continuous action split is outside action vector")
    if state_gripper < 1:
        raise ValueError("state must contain at least one gripper dimension")
    dataset = str(scalar(batch, "dataset"))
    measured_q_required = dataset in EXPECTED_CANDIDATE_RULES or (
        bool(scalar(batch, "require_measured_candidate_q"))
        if "require_measured_candidate_q" in batch.files
        else False
    )
    expected_rules = EXPECTED_CANDIDATE_RULES.get(dataset)
    if expected_rules is None and not measured_q_required:
        expected_rules = OFFLINE_CANDIDATE_RULES
    if expected_rules is not None and candidate_rule_ids != expected_rules:
        raise ValueError(
            "candidate_rule_ids order differs from the adapter contract: "
            f"expected {expected_rules}, received {candidate_rule_ids}"
        )
    finite_q = np.isfinite(candidate_q)
    finite_q_ratio = float(finite_q.mean())
    q_spread = np.ptp(
        np.where(finite_q, candidate_q, 0.0), axis=-1
    )
    maximum_q_spread = float(q_spread.max(initial=0.0))
    nonconstant_q_fraction = float((q_spread > 1e-7).mean())
    if measured_q_required:
        q_diagnostics = validate_measured_candidate_q(candidate_q)
        maximum_q_spread = float(q_diagnostics["maximum_spread"])
        nonconstant_q_fraction = float(
            q_diagnostics["nonconstant_fraction"]
        )
    identities = {}
    for name, source, default in (
        ("task_ids", "task_ids", [0] * primary.shape[0]),
        ("episode_ids", "episode_indices", list(range(primary.shape[0]))),
    ):
        values = np.asarray(batch[source] if source in batch.files else default)
        if values.shape != (primary.shape[0],) or values.dtype.kind not in "iu":
            raise ValueError(f"{source} must have one integer identifier per episode")
        identities[name] = values.tolist()
    languages = (
        batch["languages"].tolist() if "languages" in batch.files
        else [str(scalar(batch, "language"))] * primary.shape[0]
    )
    if len(languages) != primary.shape[0] or not all(
        isinstance(value, str) and value.strip() for value in languages
    ):
        raise ValueError("languages must have one nonempty instruction per episode")
    return {
        "batch_size": primary.shape[0],
        "sequence_length": actions.shape[1],
        "action_steps": actions.shape[2],
        "action_dim": action_dim,
        "continuous_action_dim": continuous,
        "state_arm_dim": state_arm,
        "state_gripper_dim": state_gripper,
        "gripper_width": bool(scalar(batch, "gripper_width")),
        "dataset": dataset,
        "task": str(scalar(batch, "task")),
        "language": str(scalar(batch, "language")),
        "candidate_rule_ids": list(candidate_rule_ids),
        "measured_candidate_q_required": measured_q_required,
        "training_mode": "measured_decision" if measured_q_required else "offline_imitation",
        "candidate_q_supervision_active": measured_q_required,
        "decision_supervision_active": measured_q_required,
        **identities,
        "languages": languages,
        "candidate_q_finite_ratio": finite_q_ratio,
        "candidate_q_maximum_spread": maximum_q_spread,
        "candidate_q_nonconstant_fraction": nonconstant_q_fraction,
        "candidate_q_source": (
            str(scalar(batch, "candidate_q_source"))
            if "candidate_q_source" in batch.files
            else "unspecified"
        ),
    }


def preprocess(images: np.ndarray, image_processor) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [image_processor(Image.fromarray(frame)) for frame in episode]
            )
            for episode in images
        ]
    )


def build_model(contract: dict, device: torch.device):
    from models.dreamvla_model import DreamVLA

    model = DreamVLA(
        finetune_type=contract["dataset"].lower(),
        clip_device=str(device),
        vit_checkpoint_path=None,
        allow_random_vision_encoder=True,
        sequence_length=contract["sequence_length"],
        num_resampler_query=4,
        action_pred_steps=contract["action_steps"],
        action_dim=contract["action_dim"],
        continuous_action_dim=contract["continuous_action_dim"],
        state_arm_dim=contract["state_arm_dim"],
        state_gripper_dim=contract["state_gripper_dim"],
        transformer_layers=2,
        hidden_dim=128,
        transformer_heads=4,
        phase="finetune",
        gripper_width=contract["gripper_width"],
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
        diwa_require_supervision=contract["decision_supervision_active"],
    ).to(device)
    model._init_model_type()
    return model


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch = np.load(args.input, allow_pickle=False)
    contract = validate_batch(batch)

    import clip
    device = torch.device(args.device)
    model = build_model(contract, device)

    primary = preprocess(batch["primary"], model.image_processor).to(device)
    wrist = preprocess(batch["wrist"], model.image_processor).to(device)
    state = torch.from_numpy(batch["state"]).float().to(device)
    labels = torch.from_numpy(batch["actions"]).float().to(device)
    text_tokens = clip.tokenize(
        contract["languages"]
    ).to(device)
    text_tokens = text_tokens[:, None].expand(
        -1, primary.shape[1], -1
    ).contiguous()
    supervision = {
        "decision_supervision_enabled": contract["decision_supervision_active"],
        "rewards": torch.from_numpy(batch["rewards"]).float().to(device),
        "dones": torch.from_numpy(batch["dones"]).bool().to(device),
        "progress": torch.from_numpy(batch["progress"]).float().to(device),
        "valid": torch.ones(
            contract["batch_size"],
            contract["sequence_length"],
            dtype=torch.bool,
            device=device,
        ),
        "observation_valid": torch.ones(
            contract["batch_size"],
            primary.shape[1],
            dtype=torch.bool,
            device=device,
        ),
        "candidate_actions": torch.from_numpy(
            batch["candidate_actions"]
        ).float().to(device),
        "candidate_q_values": torch.from_numpy(
            batch["candidate_q_values"]
        ).float().to(device),
        "task_ids": torch.tensor(contract["task_ids"], dtype=torch.long, device=device),
        "episode_ids": torch.tensor(contract["episode_ids"], dtype=torch.long, device=device),
    }

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )
    optimizer.zero_grad(set_to_none=True)
    trained = model(
        image_primary=primary,
        image_wrist=wrist,
        state=state,
        text_token=text_tokens,
        action_label=labels,
        mode="train",
        diwa_budget_ratio=0.5,
        diwa_teacher_forcing_ratio=0.5,
        diwa_supervision=supervision,
    )
    split = contract["continuous_action_dim"]
    action_loss = F.smooth_l1_loss(
        trained["arm_action"].float(), labels[..., :split]
    ) + 0.01 * F.binary_cross_entropy(
        trained["gripper_action"].float().clamp(1e-6, 1.0 - 1e-6),
        labels[..., split:],
    )
    optimized_auxiliary = active_auxiliary_losses(contract)
    auxiliary_loss = sum(
        trained["aux_losses"][name] for name in optimized_auxiliary
    )
    total_loss = action_loss + auxiliary_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite training loss: {total_loss}")
    total_loss.backward()
    influence_gradient = model.diwa_core.influence_estimator[-1].weight.grad
    if influence_gradient is None or not torch.isfinite(influence_gradient).all():
        raise RuntimeError("influence estimator did not receive finite gradients")
    optimizer.step()
    model.diwa_core.update_target_networks(0.01)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        evaluated = model(
            image_primary=primary[:, : contract["sequence_length"]],
            image_wrist=wrist[:, : contract["sequence_length"]],
            state=state[:, : contract["sequence_length"]],
            text_token=text_tokens[:, : contract["sequence_length"]],
            action=torch.zeros(
                contract["batch_size"],
                contract["sequence_length"],
                contract["action_dim"],
                device=device,
            ),
            mode="test",
            diwa_current_step=contract["sequence_length"] - 1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    prediction = torch.cat(
        (evaluated["arm_action"], evaluated["gripper_action"]), dim=-1
    )
    prediction = prediction.reshape(-1, contract["action_steps"], contract["action_dim"])[0]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("evaluation produced a non-finite action")
    selected_ratio = float(evaluated["selected_ratio"])
    if not 0.0 < selected_ratio < 1.0:
        raise RuntimeError(f"invalid sparse selected ratio {selected_ratio}")

    args.action_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.action_output, prediction.float().cpu().numpy())
    if args.checkpoint_output is not None:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "contract": contract,
                "source_export": str(args.input),
            },
            args.checkpoint_output,
        )
    report = {
        "status": "policy_ok",
        **contract,
        "source_export": str(args.input),
        "action_output": str(args.action_output),
        "checkpoint_output": (
            str(args.checkpoint_output)
            if args.checkpoint_output is not None
            else None
        ),
        "train_total_loss": float(total_loss.detach()),
        "train_action_loss": float(action_loss.detach()),
        "optimized_auxiliary_losses": list(optimized_auxiliary),
        "disabled_auxiliary_losses": (
            [] if contract["decision_supervision_active"]
            else ["critic", "progress", "regret", "contrastive"]
        ),
        "aux_losses": {
            name: float(trained["aux_losses"][name].detach())
            for name in optimized_auxiliary
        },
        "influence_gradient_norm": float(influence_gradient.norm()),
        "eval_latency_ms": latency_ms,
        "eval_selected_ratio": selected_ratio,
        "eval_expanded_tokens": evaluated["expanded_tokens"].cpu().tolist(),
        "eval_action_shape": list(prediction.shape),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("DIWA_EXPORTED_POLICY_SMOKE_OK")


if __name__ == "__main__":
    main()
