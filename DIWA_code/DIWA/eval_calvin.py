
import os
import random
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from models.dreamvla_model import DreamVLA
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.eval_utils_calvin import eval_one_epoch_calvin_ddp
from utils.arguments_utils import get_parser, validate_diwa_args
from utils.checkpoint_utils import (
    validate_diwa_checkpoint,
    validate_pretrained_checkpoint,
)
from utils.model_utils import freeze_vision_backbone


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    validate_diwa_args(args, training=False)
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
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
        depth_pred=args.depth_pred,
        use_depth_query=args.use_depth_query,
        use_dpt_head=args.use_dpt_head,
        trajectory_pred = args.trajectory_pred,
        pred_num=args.pred_num,
        use_trajectory_query = args.use_trajectory_query,
        track_label_patch_size=args.track_label_patch_size,
        use_dinosiglip = args.use_dinosiglip,
        use_dit_head = args.use_dit_head,
        
        dino_feat_pred=args.dino_feat_pred,
        sam_feat_pred=args.sam_feat_pred,
        no_pred_gripper_traj= args.no_pred_gripper_traj,
        no_unshuffle=args.no_unshuffle,
        use_gpt2_pretrained = args.use_gpt2_pretrained,
        share_query=args.share_query,
        attn_implementation= args.attn_implementation,
        dit_type = args.dit_type,
        use_fm=args.use_fm,
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
        diwa_require_supervision=False,
        
    )
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
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
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
    model.clip_model.requires_grad_(False)
    freeze_vision_backbone(
        model,
        use_dinosiglip=args.use_dinosiglip,
    )
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(
            args.resume_from_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if args.use_diwa:
            checkpoint_arguments = checkpoint.get("run_arguments")
            checkpoint_frozen_signature = checkpoint.get(
                "frozen_parameter_signature"
            )
            missing_metadata = [
                name
                for name, value in (
                    ("run_arguments", checkpoint_arguments),
                    ("frozen_parameter_signature", checkpoint_frozen_signature),
                )
                if value is None
            ]
            if missing_metadata and not args.diwa_allow_legacy_checkpoint:
                raise RuntimeError(
                    "DIWA checkpoint lacks required compatibility metadata: "
                    + ", ".join(missing_metadata)
                    + "; pass "
                    "--diwa_allow_legacy_checkpoint only after independently "
                    "verifying its architecture and frozen base weights"
                )
            validate_diwa_checkpoint(
                ddp_model,
                checkpoint["model_state_dict"],
                checkpoint_arguments=checkpoint_arguments,
                current_arguments=(
                    vars(args) if checkpoint_arguments is not None else None
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
                    "evaluation checkpoint contains state absent from the model: "
                    + ", ".join(coverage["unexpected_keys"][:8])
                )
        ddp_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
    ddp_model.eval()
    eval_log_dir = 'evaluate'
    if args.finetune_type == "calvin":
        eval_one_epoch_calvin_ddp(
            args=args,
            model=ddp_model,
            image_processor=model.image_processor,
            tokenizer=clip,
            dataset_path=args.calvin_dataset,
            future_act_len=args.future_act_len,
            eval_log_dir=eval_log_dir,
            debug=args.visualize,
            reset=args.reset,
            diverse_inst=args.diverse_inst
        )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()
