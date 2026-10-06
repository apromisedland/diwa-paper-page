import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.dreamvla_model import DreamVLA
from models.diwa.optimization import make_lr_scheduler
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser, validate_diwa_args
from utils.checkpoint_utils import (
    DIWA_CHECKPOINT_SCHEMA_VERSION,
    DIWA_METHOD_REVISION,
    atomic_torch_save,
    frozen_parameter_signature,
    gather_rng_states,
    missing_training_arguments,
    restore_rank_rng_state,
    validate_diwa_checkpoint,
    validate_model_arguments,
    validate_pretrained_checkpoint,
    validate_resume_arguments,
)
from utils.data_utils import get_calvin_dataset, get_droid_dataset, get_libero_pretrain_dataset, get_libero_finetune_dataset, get_real_finetune_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  
from utils.model_utils import freeze_vision_backbone


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
    return total_params, trainable_params, trainable_names

@record
def main(args):
    validate_diwa_args(args, training=True)
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("dreamvla", f"dreamvla_{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}")
    print("run_name:", args.run_name)
    model = DreamVLA(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        action_dim=args.action_dim,
        continuous_action_dim=args.continuous_action_dim,
        state_arm_dim=args.state_arm_dim,
        state_gripper_dim=args.state_gripper_dim,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        
        pred_num = args.pred_num,
        depth_pred = args.depth_pred,
        use_dpt_head= args.use_dpt_head,
        use_depth_query = args.use_depth_query,
        trajectory_pred = args.trajectory_pred,
        use_trajectory_query = args.use_trajectory_query,
        track_label_patch_size=args.track_label_patch_size,

        dino_feat_pred=args.dino_feat_pred,
        sam_feat_pred=args.sam_feat_pred,

        use_dinosiglip = args.use_dinosiglip,
        use_dit_head = args.use_dit_head,
        no_pred_gripper_traj= args.no_pred_gripper_traj,
        no_unshuffle=args.no_unshuffle,
        use_gpt2_pretrained = args.use_gpt2_pretrained,
        share_query=args.share_query,
        attn_implementation= args.attn_implementation,
        dit_type = args.dit_type,
        use_fm = args.use_fm,
        use_diwa=args.use_diwa,
        diwa_horizon=args.diwa_horizon,
        diwa_num_slots=args.diwa_num_slots,
        diwa_world_layers=args.diwa_world_layers,
        diwa_fusion_layers=args.diwa_fusion_layers,
        diwa_dropout=args.diwa_dropout,
        diwa_influence_temperature=args.diwa_influence_temperature,
        diwa_counterfactual_samples=args.diwa_counterfactual_samples,
        diwa_entropy_weight=args.diwa_entropy_weight,
        diwa_eval_budget_ratio=args.diwa_budget_ratio,
        diwa_sam_feature_dim=args.diwa_sam_feature_dim,
        diwa_slot_iterations=args.diwa_slot_iterations,
        diwa_adaptive_budget=args.diwa_adaptive_budget,
        diwa_minimum_budget_ratio=args.diwa_minimum_budget_ratio,
        diwa_budget_threshold=args.diwa_budget_threshold,
        diwa_critic_discount=args.diwa_critic_discount,
        diwa_regret_candidates=args.diwa_regret_candidates,
        diwa_contrastive_margin=args.diwa_contrastive_margin,
        diwa_influence_policy_weight=args.diwa_influence_policy_weight,
        diwa_influence_value_weight=args.diwa_influence_value_weight,
        diwa_influence_progress_weight=args.diwa_influence_progress_weight,
        diwa_influence_covariance_weight=args.diwa_influence_covariance_weight,
        diwa_influence_probes=args.diwa_influence_probes,
        diwa_require_supervision=args.diwa_require_supervision,
    )
    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "droid":
        calvin_dataset = get_droid_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_pretrain":
        calvin_dataset = get_libero_pretrain_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_finetune":
        calvin_dataset = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "real":
        calvin_dataset = get_real_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "oxe":
        calvin_dataset = get_oxe_dataset(args, model.image_processor, clip, epoch=0)
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module:
            freeze_vision_backbone(
                model,
                use_dinosiglip=args.use_dinosiglip,
                convert_to_bfloat16=True,
            )
        if "image_primary_projector" in args.bf16_module:
            model.image_primary_projector.bfloat16()
            model.cls_token_primary_projector.bfloat16()
        if "image_wrist_projector" in args.bf16_module:
            model.image_wrist_projector.bfloat16()
            model.cls_token_wrist_projector.bfloat16()
        if "perceiver_resampler" in args.bf16_module:
            model.perceiver_resampler.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module and args.obs_pred:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
        if "depth_decoder" in args.bf16_module and args.depth_pred:
            model.depth_decoder.bfloat16()
            model.depth_decoder_obs_pred_projector.bfloat16()
        if "action_decoder" in args.bf16_module:
            model.action_decoder.bfloat16()
            model.action_decoder_obs_pred_projector.bfloat16()
        if "dino_decoder" in args.bf16_module and args.dino_feat_pred:
            model.dino_decoder.bfloat16()
            model.dino_decoder_obs_pred_projector.bfloat16()
        if "sam_decoder" in args.bf16_module and args.sam_feat_pred:
            model.sam_decoder.bfloat16()
            model.sam_decoder_obs_pred_projector.bfloat16()
        if "text_encoder" in args.bf16_module:
            model.clip_model.bfloat16()
    model.clip_model.requires_grad_(False)
    freeze_vision_backbone(
        model,
        use_dinosiglip=args.use_dinosiglip,
    )
    
    total_params, trainable_params, trainable_names = count_parameters(model)
    if args.rank == 0:
        print("total_params: {} M".format(total_params/1024/1024))
        print("trainable_params: {} M".format(trainable_params/1024/1024))
        print("trainable names: ", trainable_names)
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)  # TODO make sure the parameters which need to be optimized are passing
    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.use_diwa:
        lr_scheduler = make_lr_scheduler(
            optimizer, calvin_dataset.dataloader.num_batches, args.num_epochs,
            args.warmup_epochs, args.gradient_accumulation_steps, args.lr_scheduler,
        )
    elif args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    resume_from_epoch = 0
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")    
        checkpoint = torch.load(
            args.finetune_from_pretrained_ckpt,
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("pretrained checkpoint has no model_state_dict")
        state_dict = checkpoint["model_state_dict"]
        image_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "image_decoder" in k]
        image_decoder_obs_pred_projector_keys = [k for k in checkpoint["model_state_dict"].keys() if "image_decoder_obs_pred_projector" in k]
        action_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "action_decoder" in k]
        resampler_keys = [k for k in checkpoint["model_state_dict"].keys() if "perceiver_resampler" in k]
        allowed_missing_names = set()
        allowed_missing_prefixes = []
        if args.reset_action_token:
            allowed_missing_names.add("module.action_pred_token")
            state_dict.pop("module.action_pred_token", None)
        if args.reset_obs_token:
            allowed_missing_names.add("module.obs_tokens")
            state_dict.pop("module.obs_tokens", None)
        if args.reset_mask_token:
            allowed_missing_names.add("module.mask_token")
            state_dict.pop("module.mask_token", None)
        if args.reset_image_decoder:
            allowed_missing_prefixes.append("module.image_decoder")
            for k in image_decoder_keys:
                state_dict.pop(k, None)
        if args.reset_action_decoder:
            allowed_missing_prefixes.append("module.action_decoder")
            for k in action_decoder_keys:
                state_dict.pop(k, None)
        if args.share_query:
            allowed_missing_prefixes.append(
                "module.image_decoder_obs_pred_projector"
            )
            for k in image_decoder_obs_pred_projector_keys:
                state_dict.pop(k, None)
        if args.reset_resampler:
            allowed_missing_prefixes.append("module.perceiver_resampler")
            for k in resampler_keys:
                state_dict.pop(k, None)
            projector_names = {
                "module.image_primary_projector.weight",
                "module.cls_token_primary_projector.weight",
                "module.image_wrist_projector.weight",
                "module.cls_token_wrist_projector.weight",
            }
            allowed_missing_names.update(projector_names)
            for name in projector_names:
                state_dict.pop(name, None)
        position_name = "module.transformer_backbone_position_embedding"
        if position_name in state_dict:
            saved_position = state_dict[position_name]
            current_position = ddp_model.state_dict()[position_name]
            if saved_position.shape != current_position.shape:
                if (
                    saved_position.ndim == current_position.ndim == 4
                    and saved_position.shape[0] == current_position.shape[0]
                    and saved_position.shape[2:] == current_position.shape[2:]
                    and saved_position.shape[1] >= current_position.shape[1]
                ):
                    state_dict[position_name] = saved_position[
                        :, : current_position.shape[1], :, :
                    ]
        coverage = validate_pretrained_checkpoint(
            ddp_model,
            state_dict,
            allowed_missing_names=allowed_missing_names,
            allowed_missing_prefixes=tuple(allowed_missing_prefixes),
        )
        incompatible = ddp_model.load_state_dict(state_dict, strict=False)
        if args.rank == 0:
            print(
                "Loaded pretrained checkpoint:",
                coverage["loaded_model_key_count"],
                "model entries;",
                len(incompatible.missing_keys),
                "intentional/new entries initialized locally;",
                len(coverage["unexpected_keys"]),
                "source-only entries ignored.",
            )
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(
            args.resume_from_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_arguments = checkpoint.get("run_arguments")
        checkpoint_rng_states = checkpoint.get("rng_states")
        required_metadata = {
            "run_arguments": checkpoint_arguments,
            "rng_states": checkpoint_rng_states,
        }
        if args.use_diwa:
            checkpoint_frozen_signature = checkpoint.get(
                "frozen_parameter_signature"
            )
            required_metadata["frozen_parameter_signature"] = (
                checkpoint_frozen_signature
            )
            required_metadata["checkpoint_schema_version"] = checkpoint.get(
                "checkpoint_schema_version"
            )
        missing_metadata = [
            name for name, value in required_metadata.items() if value is None
        ]
        if missing_metadata and not args.diwa_allow_legacy_checkpoint:
            raise RuntimeError(
                "training checkpoint lacks required resume metadata: "
                + ", ".join(missing_metadata)
                + "; pass --diwa_allow_legacy_checkpoint only after "
                "independently verifying the legacy checkpoint"
            )
        checkpoint_schema = checkpoint.get("checkpoint_schema_version")
        if (
            args.use_diwa
            and checkpoint_schema is not None
            and checkpoint_schema != DIWA_CHECKPOINT_SCHEMA_VERSION
            and not args.diwa_allow_legacy_checkpoint
        ):
            raise RuntimeError(
                "training checkpoint schema is incompatible with the current "
                f"DIWA training method: {checkpoint_schema} != "
                f"{DIWA_CHECKPOINT_SCHEMA_VERSION}; restart training or pass "
                "--diwa_allow_legacy_checkpoint only after independently "
                "auditing the resume boundary"
            )
        validated_checkpoint_arguments = checkpoint_arguments
        if checkpoint_arguments is not None:
            argument_gaps = missing_training_arguments(
                checkpoint_arguments,
                vars(args),
            )
            if argument_gaps:
                if not args.diwa_allow_legacy_checkpoint:
                    raise RuntimeError(
                        "training checkpoint compatibility metadata is incomplete; "
                        "missing: "
                        + ", ".join(argument_gaps[:8])
                        + (
                            ""
                            if len(argument_gaps) <= 8
                            else f" ... ({len(argument_gaps)} total)"
                        )
                    )
                validated_checkpoint_arguments = None
                if args.rank == 0:
                    print(
                        "Legacy checkpoint argument validation skipped for missing "
                        f"keys: {', '.join(argument_gaps[:8])}"
                    )
            else:
                validate_model_arguments(checkpoint_arguments, vars(args))
                validate_resume_arguments(checkpoint_arguments, vars(args))
        if args.use_diwa:
            validate_diwa_checkpoint(
                ddp_model,
                checkpoint["model_state_dict"],
                checkpoint_arguments=validated_checkpoint_arguments,
                current_arguments=(
                    vars(args)
                    if validated_checkpoint_arguments is not None
                    else None
                ),
                checkpoint_frozen_signature=checkpoint_frozen_signature,
            )
        else:
            coverage = validate_pretrained_checkpoint(
                ddp_model,
                checkpoint["model_state_dict"],
            )
            if coverage["unexpected_keys"]:
                raise RuntimeError(
                    "resume checkpoint contains state absent from the model: "
                    + ", ".join(coverage["unexpected_keys"][:8])
                )
        ddp_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
        if checkpoint_rng_states is not None:
            restore_rank_rng_state(
                checkpoint_rng_states,
                rank=args.rank,
                world_size=args.world_size,
            )

    # Compute after every requested load. A parameter that was saved as
    # trainable by an older run can be frozen by the current evaluation/training
    # recipe; its loaded value must then be covered by the new signature.
    frozen_signature = frozen_parameter_signature(ddp_model)

    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if args.rank == 0:
        with open(os.path.join(ckpt_dir, "run_config.json"), "w") as stream:
            json.dump({"arguments": vars(args), "torch_version": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name(device_id),
                       "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * args.world_size,
                       "frozen_parameter_signature": frozen_signature,
                       "diwa_method_revision": DIWA_METHOD_REVISION}, stream, indent=2)
    
    ddp_model.train()
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)

        calvin_loader = calvin_dataset.dataloader
        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
        )
        should_save = (
            args.save_checkpoint
            and epoch % args.save_checkpoint_seq == 0
            and epoch > args.start_save_checkpoint
        )
        if should_save:
            rng_states = gather_rng_states(
                rank=args.rank,
                world_size=args.world_size,
            )
        if args.rank == 0 and should_save:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "run_arguments": vars(args),
                "frozen_parameter_signature": frozen_signature,
                "rng_states": rng_states,
                "checkpoint_schema_version": DIWA_CHECKPOINT_SCHEMA_VERSION,
                "diwa_method_revision": DIWA_METHOD_REVISION,
            }
            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f"Saving checkpoint to {ckpt_path}")
            atomic_torch_save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                previous_epoch = epoch - args.save_checkpoint_seq
                previous_path = os.path.join(
                    ckpt_dir, get_ckpt_name(args, previous_epoch)
                )
                if previous_epoch >= 0 and os.path.exists(previous_path):
                    os.remove(previous_path)
        if should_save and dist.is_available() and dist.is_initialized():
            dist.barrier()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
