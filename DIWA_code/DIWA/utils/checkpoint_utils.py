from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


DIWA_CHECKPOINT_SCHEMA_VERSION = 4
DIWA_METHOD_REVISION = "code-repair-2026-10-06-causal-pairs-mask"


CHECKPOINT_COMPATIBILITY_ARGUMENTS = (
    "sequence_length",
    "window_size",
    "num_resampler_query",
    "num_obs_token_per_image",
    "calvin_input_image_size",
    "patch_size",
    "action_pred_steps",
    "action_dim",
    "continuous_action_dim",
    "state_arm_dim",
    "state_gripper_dim",
    "atten_only_obs",
    "attn_robot_proprio_state",
    "atten_goal",
    "atten_goal_state",
    "mask_l_obs_ratio",
    "dit_type",
    "use_dit_head",
    "use_fm",
    "use_dinosiglip",
    "use_gpt2_pretrained",
    "obs_pred",
    "depth_pred",
    "use_depth_query",
    "use_dpt_head",
    "trajectory_pred",
    "use_trajectory_query",
    "track_label_patch_size",
    "no_pred_gripper_traj",
    "no_unshuffle",
    "dino_feat_pred",
    "sam_feat_pred",
    "pred_num",
    "share_query",
    "attn_implementation",
    "transformer_layers",
    "hidden_dim",
    "transformer_heads",
    "gripper_width",
    "use_diwa",
    "diwa_horizon",
    "diwa_num_slots",
    "diwa_world_layers",
    "diwa_fusion_layers",
    "diwa_dropout",
    "diwa_influence_temperature",
    "diwa_counterfactual_samples",
    "diwa_entropy_weight",
    "diwa_budget_ratio",
    "diwa_sam_feature_dim",
    "diwa_slot_iterations",
    "diwa_adaptive_budget",
    "diwa_minimum_budget_ratio",
    "diwa_budget_threshold",
    "diwa_critic_discount",
    "diwa_regret_candidates",
    "diwa_contrastive_margin",
    "diwa_influence_policy_weight",
    "diwa_influence_value_weight",
    "diwa_influence_progress_weight",
    "diwa_influence_covariance_weight",
    "diwa_influence_probes",
)


# These options affect the optimization trajectory rather than model shape.
# They are checked only when resuming training, so evaluation may still use a
# different batch size, seed, or number of workers.
RESUME_COMPATIBILITY_ARGUMENTS = (
    "seed",
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "world_size",
    "workers",
    "learning_rate",
    "weight_decay",
    "lr_scheduler",
    "warmup_epochs",
    "precision",
    "phase",
    "bf16_module",
    "max_grad_norm",
    "finetune_type",
    "calvin_dataset",
    "root_dir",
    "data_in_ceph",
    "libero_dataset_name",
    "load_libero_file",
    "dataset_info",
    "real_dataset_names",
    "real_dataset_adapter",
    "use_aug_data",
    "image_primary_size",
    "image_wrist_size",
    "primary_mode",
    "small_size",
    "min_window_size",
    "max_window_size",
    "multi_step_action",
    "future_steps",
    "rgb_pad",
    "gripper_pad",
    "traj_cons",
    "text_aug",
    "dif_ws",
    "partial_data",
    "except_lang",
    "merge_data",
    "load_track_labels",
    "track_label_path",
    "load_dino_features",
    "dino_features_path",
    "load_sam_features",
    "sam_features_path",
    "max_rel_pos",
    "max_rel_orn",
    "magic_scaling_factor_pos",
    "magic_scaling_factor_orn",
    "loss_action",
    "loss_image",
    "loss_depth",
    "loss_trajectory",
    "loss_dino_feat",
    "loss_sam_feat",
    "loss_arm_action_ratio",
    "loss_gripper_action_ratio",
    "diwa_budget_warmup_steps",
    "diwa_budget_anneal_steps",
    "diwa_teacher_forcing_steps",
    "diwa_world_pretrain_steps",
    "diwa_counterfactual_start_steps",
    "diwa_regret_start_steps",
    "diwa_loss_proposal",
    "diwa_loss_mask",
    "diwa_loss_future",
    "diwa_loss_influence",
    "diwa_loss_budget",
    "diwa_loss_critic",
    "diwa_loss_progress",
    "diwa_loss_regret",
    "diwa_loss_contrastive",
    "diwa_target_tau",
    "diwa_require_supervision",
    "diwa_supervision_path",
)


def _preview(keys: list[str]) -> str:
    text = ", ".join(keys[:8])
    return text if len(keys) <= 8 else f"{text} ... ({len(keys)} total)"


def frozen_parameter_signature(
    model: nn.Module,
    *,
    state_dict_keys: set[str] | None = None,
) -> str:
    """Hash parameters omitted from a filtered model checkpoint.

    At save time, ``state_dict_keys`` is omitted and the selection mirrors
    ``get_checkpoint``. During validation, checkpoint keys define the omitted
    set directly. This remains correct if evaluation freezes a parameter that
    was trainable, and therefore saved, during training.
    """
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if state_dict_keys is None:
            is_saved_diwa_target = "diwa_core" in name and "target_" in name
            if parameter.requires_grad or is_saved_diwa_target:
                continue
        elif name in state_dict_keys:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    digest.update(f"parameter_count={count}".encode("ascii"))
    return digest.hexdigest()


def _validate_arguments(
    checkpoint_arguments: Mapping[str, object],
    current_arguments: Mapping[str, object],
    argument_names: tuple[str, ...] = CHECKPOINT_COMPATIBILITY_ARGUMENTS,
) -> None:
    missing = [
        name
        for name in argument_names
        if name not in checkpoint_arguments or name not in current_arguments
    ]
    if missing:
        raise RuntimeError(
            "checkpoint compatibility metadata is incomplete; missing: "
            + _preview(sorted(missing))
        )
    mismatches = [
        f"{name}={checkpoint_arguments[name]!r} (checkpoint) != "
        f"{current_arguments[name]!r} (current)"
        for name in argument_names
        if checkpoint_arguments[name] != current_arguments[name]
    ]
    if mismatches:
        raise RuntimeError(
            "checkpoint configuration is incompatible: "
            + "; ".join(mismatches[:8])
            + (
                ""
                if len(mismatches) <= 8
                else f" ... ({len(mismatches)} total)"
            )
        )


def validate_resume_arguments(
    checkpoint_arguments: Mapping[str, object],
    current_arguments: Mapping[str, object],
) -> None:
    """Reject optimizer/schedule changes that make a resume non-equivalent."""
    _validate_arguments(
        checkpoint_arguments,
        current_arguments,
        RESUME_COMPATIBILITY_ARGUMENTS,
    )


def validate_model_arguments(
    checkpoint_arguments: Mapping[str, object],
    current_arguments: Mapping[str, object],
) -> None:
    """Validate model and tensor-layout arguments for any training resume."""
    _validate_arguments(checkpoint_arguments, current_arguments)


def missing_training_arguments(
    checkpoint_arguments: Mapping[str, object],
    current_arguments: Mapping[str, object],
) -> tuple[str, ...]:
    """List model/resume keys absent from either side of compatibility data."""
    required = set(CHECKPOINT_COMPATIBILITY_ARGUMENTS)
    required.update(RESUME_COMPATIBILITY_ARGUMENTS)
    return tuple(
        sorted(
            name
            for name in required
            if name not in checkpoint_arguments or name not in current_arguments
        )
    )


def _is_diwa_state(name: str) -> bool:
    return (
        "diwa_core." in name
        or name.endswith("diwa_future_position_embedding")
        or name.endswith("diwa_influence_scale")
    )


def _is_explicitly_allowed(
    name: str,
    *,
    exact_names: set[str],
    prefixes: tuple[str, ...],
) -> bool:
    return name in exact_names or any(name.startswith(prefix) for prefix in prefixes)


def validate_pretrained_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, object],
    *,
    allowed_missing_names: set[str] | None = None,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Validate transfer-learning coverage before a non-strict load.

    A baseline checkpoint may legitimately omit newly added DIWA parameters,
    frozen encoders, and heads explicitly reset by the command line. Every
    other trainable parameter must be present with the correct shape. Unknown
    source-only keys are reported to the caller but do not invalidate transfer.
    """
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("checkpoint model_state_dict must be a mapping")
    exact_names = set(allowed_missing_names or ())
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    required = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not _is_diwa_state(name)
        and not _is_explicitly_allowed(
            name,
            exact_names=exact_names,
            prefixes=allowed_missing_prefixes,
        )
    }
    missing = sorted(required - checkpoint_keys)
    if missing:
        raise RuntimeError(
            "pretrained checkpoint is missing required trainable state: "
            + _preview(missing)
        )
    incompatible = []
    for key in sorted(checkpoint_keys & model_keys):
        saved = state_dict[key]
        current = model_state[key]
        if not isinstance(saved, torch.Tensor):
            incompatible.append(f"{key} is not a tensor")
        elif saved.shape != current.shape:
            incompatible.append(
                f"{key} shape {tuple(saved.shape)} != {tuple(current.shape)}"
            )
    if incompatible:
        raise RuntimeError(
            "pretrained checkpoint tensors are incompatible: "
            + "; ".join(incompatible[:8])
            + ("" if len(incompatible) <= 8 else f" ... ({len(incompatible)} total)")
        )
    return {
        "required_parameter_count": len(required),
        "loaded_model_key_count": len(checkpoint_keys & model_keys),
        "unexpected_keys": tuple(sorted(checkpoint_keys - model_keys)),
    }


def capture_rng_state() -> dict[str, object]:
    """Capture process RNGs using only weights-only-safe checkpoint values."""
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": bit_generator,
            "keys": torch.from_numpy(keys.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_nested_tuple(item) for item in value)
    return value


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"RNG state is incomplete; missing: {sorted(missing)}")
    random.setstate(_nested_tuple(state["python"]))
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise RuntimeError("NumPy RNG state must be a mapping")
    numpy_keys = numpy_state.get("keys")
    if not isinstance(numpy_keys, torch.Tensor):
        raise RuntimeError("NumPy RNG keys must be a tensor")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_keys.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise RuntimeError("Torch CPU RNG state must be a tensor")
    torch.set_rng_state(torch_cpu.cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the checkpoint RNG state: "
                f"{len(cuda_states)} (checkpoint) != {torch.cuda.device_count()} (current)"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def gather_rng_states(*, rank: int, world_size: int) -> list[dict[str, object]]:
    """Collect one RNG snapshot per distributed worker in rank order."""
    local = {"rank": int(rank), "state": capture_rng_state()}
    if dist.is_available() and dist.is_initialized():
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
        if any(item is None for item in gathered):
            raise RuntimeError("failed to gather every worker RNG state")
        result = [item for item in gathered if item is not None]
    else:
        if world_size != 1 or rank != 0:
            raise RuntimeError("distributed RNG gathering requires an initialized process group")
        result = [local]
    ranks = [int(item["rank"]) for item in result]
    if sorted(ranks) != list(range(world_size)) or len(set(ranks)) != world_size:
        raise RuntimeError(f"checkpoint RNG ranks are invalid: {ranks}")
    return sorted(result, key=lambda item: int(item["rank"]))


def restore_rank_rng_state(
    rng_states: object,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Restore the current worker's RNG snapshot from a distributed checkpoint."""
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise RuntimeError(
            "checkpoint RNG state count does not match world size: "
            f"{len(rng_states) if isinstance(rng_states, list) else 'invalid'} "
            f"!= {world_size}"
        )
    if any(not isinstance(item, Mapping) for item in rng_states):
        raise RuntimeError("checkpoint RNG entries must be mappings")
    matching = [item for item in rng_states if int(item.get("rank", -1)) == rank]
    if len(matching) != 1 or not isinstance(matching[0].get("state"), Mapping):
        raise RuntimeError(f"checkpoint has no unique RNG state for rank {rank}")
    restore_rng_state(matching[0]["state"])


def atomic_torch_save(value: object, path: str | os.PathLike[str]) -> None:
    """Durably replace a checkpoint without exposing a partial destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_diwa_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, object],
    *,
    checkpoint_arguments: Mapping[str, object] | None = None,
    current_arguments: Mapping[str, object] | None = None,
    checkpoint_frozen_signature: str | None = None,
) -> None:
    """Reject incomplete or incompatible DIWA checkpoints before loading.

    Release checkpoints intentionally omit frozen upstream parameters, so a
    globally strict ``load_state_dict`` call is not possible. This validator
    instead requires every trainable parameter, every DIWA state entry, exact
    tensor shapes/dtypes, no unknown keys, and (when supplied) matching model
    construction arguments.
    """
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("checkpoint model_state_dict must be a mapping")
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    diwa_state = {
        key
        for key in model_keys
        if "diwa_core." in key
        or key.endswith("diwa_future_position_embedding")
        or key.endswith("diwa_influence_scale")
    }
    missing = sorted((trainable | diwa_state) - checkpoint_keys)
    if missing:
        raise RuntimeError(
            "checkpoint is missing trained DIWA/model state: " + _preview(missing)
        )
    unexpected = sorted(checkpoint_keys - model_keys)
    if unexpected:
        raise RuntimeError(
            "checkpoint contains state absent from the current model: "
            + _preview(unexpected)
        )
    incompatible = []
    for key in sorted(checkpoint_keys & model_keys):
        saved = state_dict[key]
        current = model_state[key]
        if not isinstance(saved, torch.Tensor):
            incompatible.append(f"{key} is not a tensor")
            continue
        if saved.shape != current.shape:
            incompatible.append(
                f"{key} shape {tuple(saved.shape)} != {tuple(current.shape)}"
            )
        elif saved.dtype != current.dtype:
            incompatible.append(f"{key} dtype {saved.dtype} != {current.dtype}")
    if incompatible:
        raise RuntimeError(
            "checkpoint tensors are incompatible: "
            + "; ".join(incompatible[:8])
            + (
                ""
                if len(incompatible) <= 8
                else f" ... ({len(incompatible)} total)"
            )
        )
    if (checkpoint_arguments is None) != (current_arguments is None):
        raise RuntimeError(
            "checkpoint and current compatibility arguments must be supplied together"
        )
    if checkpoint_arguments is not None:
        _validate_arguments(checkpoint_arguments, current_arguments)
    if checkpoint_frozen_signature is not None:
        if (
            not isinstance(checkpoint_frozen_signature, str)
            or len(checkpoint_frozen_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checkpoint_frozen_signature
            )
        ):
            raise RuntimeError(
                "checkpoint frozen-base signature must be a lowercase SHA-256 hex digest"
            )
        current_signature = frozen_parameter_signature(
            model,
            state_dict_keys=checkpoint_keys,
        )
        if checkpoint_frozen_signature != current_signature:
            raise RuntimeError(
                "checkpoint frozen-base signature does not match the current "
                "model; verify the vision/base checkpoint and model version"
            )
