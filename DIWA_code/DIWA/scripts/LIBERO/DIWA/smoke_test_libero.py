"""Run a real LIBERO simulator + full DreamVLA-DIWA smoke test.

The test deliberately uses random model weights: it verifies integration,
tensor contracts, measured simulator supervision, a finite backward/update,
and genuinely sparse inference. It does not claim task performance.
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
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def import_libero(libero_path: Path):
    resolved = libero_path.expanduser().resolve()
    if not (resolved / "libero").is_dir():
        raise FileNotFoundError(f"{resolved} is not a LIBERO checkout")
    sys.path.insert(0, str(resolved))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from utils.libero_compat import apply_robosuite_mujoco3_compat

    apply_robosuite_mujoco3_compat()

    return benchmark, OffScreenRenderEnv


def simulator_state(env) -> np.ndarray:
    candidates = (env, getattr(env, "env", None))
    for candidate in candidates:
        sim = getattr(candidate, "sim", None)
        if sim is not None:
            return np.asarray(sim.get_state().flatten()).copy()
    raise RuntimeError("LIBERO environment does not expose a simulator state")


def measured_progress(env) -> float:
    for candidate in (env, getattr(env, "env", None)):
        check = getattr(candidate, "_check_success", None)
        if check is not None:
            return float(bool(check()))
    raise RuntimeError("LIBERO environment does not expose task predicates")


def policy_candidates(action_steps: int) -> np.ndarray:
    """Stable candidate rules shared across every state and episode."""
    candidates = np.zeros((4, action_steps, 7), dtype=np.float32)
    candidates[0, :, 6] = -1.0  # hold still, gripper open
    candidates[1, :, 6] = 1.0  # hold still, gripper closed
    candidates[2, :, 0] = 0.15  # positive tool-x motion
    candidates[2, :, 6] = -1.0
    candidates[3, :, 0] = -0.15  # negative tool-x motion
    candidates[3, :, 6] = -1.0
    return candidates


def measure_candidate_returns(
    env,
    state: np.ndarray,
    candidates: np.ndarray,
    discount: float = 0.99,
) -> tuple[np.ndarray, float, bool]:
    q_values = np.zeros(candidates.shape[0], dtype=np.float32)
    first_reward = 0.0
    first_done = False
    initial_progress = None
    for candidate_index, action_chunk in enumerate(candidates):
        env.set_init_state(state)
        start_progress = measured_progress(env)
        if initial_progress is None:
            initial_progress = start_progress
        discounted_return = 0.0
        final_progress = start_progress
        steps_taken = 0
        for action_index, action in enumerate(action_chunk):
            _, reward, done, _ = env.step(action)
            steps_taken = action_index + 1
            discounted_return += (discount**action_index) * float(reward)
            final_progress = measured_progress(env)
            if candidate_index == 0 and action_index == 0:
                first_reward = float(reward)
                first_done = bool(done)
            if done:
                break
        # LIBERO rewards are sparse. Predicate progress is an independently
        # measured terminal shaping term, not a frame-index proxy.
        discounted_return += (
            discount**steps_taken
        ) * max(0.0, final_progress - start_progress)
        q_values[candidate_index] = discounted_return
    return q_values, first_reward, first_done


def proprioception(obs: dict) -> np.ndarray:
    euler = Rotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            euler,
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def collect_batch(
    env,
    init_states,
    batch_size: int,
    sequence_length: int,
    action_steps: int,
) -> dict[str, np.ndarray]:
    candidates = policy_candidates(action_steps)
    result = {
        "primary": [],
        "wrist": [],
        "state": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "progress": [],
        "candidate_actions": [],
        "candidate_q_values": [],
    }
    for episode_index in range(batch_size):
        env.reset()
        obs = env.set_init_state(init_states[episode_index % len(init_states)])
        episode = {key: [] for key in ("primary", "wrist", "state")}
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_progress = []
        episode_candidate_actions = []
        episode_candidate_q = []
        for step_index in range(sequence_length):
            state = simulator_state(env)
            episode["primary"].append(
                np.asarray(obs["agentview_image"])[::-1].copy()
            )
            episode["wrist"].append(
                np.asarray(obs["robot0_eye_in_hand_image"]).copy()
            )
            episode["state"].append(proprioception(obs))
            if step_index < sequence_length - 1:
                q_values, reward, done = measure_candidate_returns(
                    env, state, candidates
                )
                env.set_init_state(state)
                obs, _, _, _ = env.step(candidates[0, 0])
                episode_actions.append(candidates[0].copy())
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_progress.append(measured_progress(env))
                episode_candidate_actions.append(candidates.copy())
                episode_candidate_q.append(q_values)
        for key in ("primary", "wrist", "state"):
            result[key].append(episode[key])
        result["actions"].append(episode_actions)
        result["rewards"].append(episode_rewards)
        result["dones"].append(episode_dones)
        result["progress"].append(episode_progress)
        result["candidate_actions"].append(episode_candidate_actions)
        result["candidate_q_values"].append(episode_candidate_q)
    return {key: np.asarray(value) for key, value in result.items()}


def preprocess_images(images: np.ndarray, image_processor) -> torch.Tensor:
    batch = []
    for episode in images:
        batch.append(
            torch.stack(
                [image_processor(Image.fromarray(frame)) for frame in episode]
            )
        )
    return torch.stack(batch)


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two for token swapping")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    benchmark, OffScreenRenderEnv = import_libero(args.libero_path)
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task id must be in [0, {task_suite.n_tasks})")
    task = task_suite.get_task(args.task_id)
    bddl_file = (
        args.libero_path
        / "libero"
        / "libero"
        / "bddl_files"
        / task.problem_folder
        / task.bddl_file
    )
    init_file = (
        args.libero_path
        / "libero"
        / "libero"
        / "init_files"
        / task.problem_folder
        / task.init_states_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        render_gpu_device_id=0,
    )
    try:
        init_states = torch.load(init_file, map_location="cpu")
        sequence_length = 3  # two policy states plus one real future state
        action_steps = 2
        batch = collect_batch(
            env,
            init_states,
            args.batch_size,
            sequence_length,
            action_steps,
        )
    finally:
        env.close()

    # Import after LIBERO has been configured so OpenGL selection is stable.
    import clip
    from models.dreamvla_model import DreamVLA

    device = torch.device(args.device)
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

    images_primary = preprocess_images(
        batch["primary"], model.image_processor
    ).to(device)
    images_wrist = preprocess_images(
        batch["wrist"], model.image_processor
    ).to(device)
    state = torch.from_numpy(batch["state"]).to(device)
    tokens = clip.tokenize([task.language] * args.batch_size).to(device)
    text = tokens[:, None].expand(-1, sequence_length, -1).contiguous()

    action_labels = torch.from_numpy(batch["actions"]).to(device)
    # Convert LIBERO -1/+1 gripper convention to policy 0/1 convention.
    action_labels[..., 6] = (action_labels[..., 6] + 1.0) / 2.0
    candidate_actions = torch.from_numpy(batch["candidate_actions"]).to(device)
    candidate_actions[..., 6] = (candidate_actions[..., 6] + 1.0) / 2.0
    supervision = {
        "rewards": torch.from_numpy(batch["rewards"]).float().to(device),
        "dones": torch.from_numpy(batch["dones"]).bool().to(device),
        "progress": torch.from_numpy(batch["progress"]).float().to(device),
        "valid": torch.ones(args.batch_size, 2, dtype=torch.bool, device=device),
        "observation_valid": torch.ones(
            args.batch_size, sequence_length, dtype=torch.bool, device=device
        ),
        "candidate_actions": candidate_actions,
        "candidate_q_values": torch.from_numpy(
            batch["candidate_q_values"]
        )
        .float()
        .to(device),
        "task_ids": torch.zeros(
            args.batch_size, dtype=torch.long, device=device
        ),
        "episode_ids": torch.arange(
            args.batch_size, dtype=torch.long, device=device
        ),
    }

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )
    optimizer.zero_grad(set_to_none=True)
    train_output = model(
        image_primary=images_primary,
        image_wrist=images_wrist,
        state=state,
        text_token=text,
        action_label=action_labels,
        mode="train",
        diwa_budget_ratio=0.5,
        diwa_teacher_forcing_ratio=0.5,
        diwa_supervision=supervision,
    )
    action_loss = F.smooth_l1_loss(
        train_output["arm_action"].float(), action_labels[..., :6].float()
    ) + 0.01 * F.binary_cross_entropy(
        train_output["gripper_action"].float().clamp(1e-6, 1 - 1e-6),
        action_labels[..., 6:].float(),
    )
    optimized_auxiliary = (
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
    auxiliary_loss = sum(
        train_output["aux_losses"][name] for name in optimized_auxiliary
    )
    total_loss = action_loss + auxiliary_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite training loss: {total_loss}")
    total_loss.backward()
    influence_gradient = (
        model.diwa_core.influence_estimator[-1].weight.grad
    )
    if influence_gradient is None or not torch.isfinite(
        influence_gradient
    ).all():
        raise RuntimeError("influence estimator did not receive finite gradients")
    mask_gradient = model.diwa_core.counterfactual_token.grad
    if mask_gradient is None or not torch.isfinite(mask_gradient).all() or mask_gradient.norm() <= 0:
        raise RuntimeError("mask token did not receive finite nonzero gradients")
    optimizer.step()
    model.diwa_core.update_target_networks(0.01)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        eval_output = model(
            image_primary=images_primary[:, :2],
            image_wrist=images_wrist[:, :2],
            state=state[:, :2],
            text_token=text[:, :2],
            action=torch.zeros(
                args.batch_size, 2, 7, device=device
            ),
            mode="test",
            diwa_current_step=1,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    selected_ratio = float(eval_output["selected_ratio"])
    expanded_tensor = eval_output["expanded_tokens"].detach().cpu()
    expanded = expanded_tensor.tolist()
    available = int(eval_output["available_tokens"])
    if not 0.0 < selected_ratio < 1.0:
        raise RuntimeError(
            f"sparse inference selected invalid ratio {selected_ratio}"
        )
    if int(expanded_tensor.max()) >= available:
        raise RuntimeError("sparse inference expanded every future token")
    if not torch.isfinite(eval_output["arm_action"]).all():
        raise RuntimeError("inference produced non-finite actions")

    report = {
        "status": "ok",
        "dataset": "LIBERO_SPATIAL",
        "task_id": args.task_id,
        "task": task.name,
        "language": task.language,
        "device": str(device),
        "batch_size": args.batch_size,
        "train_total_loss": float(total_loss.detach()),
        "train_action_loss": float(action_loss.detach()),
        "mask_gradient_norm": float(mask_gradient.norm()),
        "aux_losses": {
            name: float(train_output["aux_losses"][name].detach())
            for name in optimized_auxiliary
        },
        "influence_gradient_norm": float(influence_gradient.norm()),
        "candidate_q_values": batch["candidate_q_values"].tolist(),
        "eval_latency_ms": latency_ms,
        "eval_selected_ratio": selected_ratio,
        "eval_expanded_tokens": expanded,
        "eval_available_tokens": available,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DIWA_LIBERO_SMOKE_OK")


if __name__ == "__main__":
    main()
