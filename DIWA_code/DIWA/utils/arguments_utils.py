import argparse
import copy
import glob
import math
import os
import random
from collections import OrderedDict
import numpy as np
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def validate_diwa_args(args, *, training: bool) -> None:
    errors = []
    use_diwa = getattr(args, "use_diwa", False)
    if args.diwa_profile_warmup_steps < 0:
        errors.append("diwa_profile_warmup_steps must be non-negative")
    if args.diwa_profile_output and not args.diwa_profile:
        errors.append("diwa_profile_output requires diwa_profile")
    if args.diwa_profile and not use_diwa:
        errors.append("diwa_profile requires use_diwa")
    if not use_diwa:
        if errors:
            raise ValueError("invalid DIWA configuration: " + "; ".join(errors))
        return
    positive_integer_fields = (
        "sequence_length",
        "action_pred_steps",
        "action_dim",
        "continuous_action_dim",
        "state_arm_dim",
        "hidden_dim",
        "transformer_heads",
        "diwa_horizon",
        "diwa_num_slots",
        "diwa_world_layers",
        "diwa_fusion_layers",
        "diwa_sam_feature_dim",
        "diwa_slot_iterations",
    )
    for name in positive_integer_fields:
        if getattr(args, name) < 1:
            errors.append(f"{name} must be positive")
    if args.state_gripper_dim is not None and args.state_gripper_dim < 1:
        errors.append("state_gripper_dim must be positive when provided")
    if (
        not args.gripper_width
        and args.state_gripper_dim is not None
        and args.state_gripper_dim != 1
    ):
        errors.append(
            "categorical gripper state requires state_gripper_dim=1"
        )
    if not 0 < args.continuous_action_dim < args.action_dim:
        errors.append("continuous_action_dim must split the action vector")
    if (
        args.hidden_dim > 0
        and args.transformer_heads > 0
        and args.hidden_dim % args.transformer_heads
    ):
        errors.append("hidden_dim must be divisible by transformer_heads")
    required_window = args.sequence_length + max(
        args.diwa_horizon, args.action_pred_steps - 1
    )
    if args.window_size < required_window:
        errors.append(
            "window_size must cover sequence_length plus the DIWA horizon "
            "and every predicted action step"
        )
    if not 0 <= args.atten_goal < args.sequence_length:
        errors.append("atten_goal must be in [0, sequence_length)")
    if not math.isfinite(args.diwa_dropout) or not 0 <= args.diwa_dropout < 1:
        errors.append("diwa_dropout must be finite and in [0, 1)")
    if (
        not math.isfinite(args.diwa_influence_temperature)
        or args.diwa_influence_temperature <= 0
    ):
        errors.append("diwa_influence_temperature must be finite and positive")
    if args.diwa_counterfactual_samples < 0:
        errors.append("diwa_counterfactual_samples must be non-negative")
    if (
        not math.isfinite(args.diwa_entropy_weight)
        or args.diwa_entropy_weight < 0
    ):
        errors.append("diwa_entropy_weight must be finite and non-negative")
    if not 0.0 < args.diwa_budget_ratio <= 1.0:
        errors.append("diwa_budget_ratio must be in (0, 1]")
    if not 0.0 < args.diwa_minimum_budget_ratio <= args.diwa_budget_ratio:
        errors.append(
            "diwa_minimum_budget_ratio must be positive and no greater "
            "than diwa_budget_ratio"
        )
    if not 0.0 < args.diwa_budget_threshold < 1.0:
        errors.append("diwa_budget_threshold must be in (0, 1)")
    if args.diwa_influence_probes < 2:
        errors.append("diwa_influence_probes must be at least two")
    if not 0 <= args.diwa_critic_discount <= 1:
        errors.append("diwa_critic_discount must be in [0, 1]")
    if args.diwa_regret_candidates < 4:
        errors.append("diwa_regret_candidates must be at least four")
    if (
        not math.isfinite(args.diwa_contrastive_margin)
        or args.diwa_contrastive_margin < 0
    ):
        errors.append("diwa_contrastive_margin must be finite and non-negative")
    influence_weight_names = (
        "policy",
        "value",
        "progress",
        "covariance",
    )
    for name in influence_weight_names:
        value = getattr(args, f"diwa_influence_{name}_weight")
        if not math.isfinite(value) or value < 0:
            errors.append(
                f"diwa_influence_{name}_weight must be finite and non-negative"
            )
    if not training and args.resume_from_checkpoint is None:
        errors.append("DIWA evaluation requires resume_from_checkpoint")
    if training:
        for name in (
            "diwa_budget_warmup_steps",
            "diwa_budget_anneal_steps",
            "diwa_teacher_forcing_steps",
            "diwa_world_pretrain_steps",
            "diwa_counterfactual_start_steps",
            "diwa_regret_start_steps",
        ):
            if getattr(args, name) < 0:
                errors.append(f"{name} must be non-negative")
        if args.batch_size < 1:
            errors.append("batch_size must be positive")
        if args.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be positive")
        if args.num_epochs < 1:
            errors.append("num_epochs must be positive")
        if args.lr_scheduler not in ("constant", "linear", "cosine"):
            errors.append(
                "DIWA lr_scheduler must be constant, linear, or cosine"
            )
        if not 0 <= args.warmup_epochs <= args.num_epochs:
            errors.append("warmup_epochs must be in [0, num_epochs]")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            errors.append("learning_rate must be finite and positive")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            errors.append("weight_decay must be finite and non-negative")
        if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
            errors.append("max_grad_norm must be finite and positive")
        if not 0 < args.diwa_target_tau <= 1:
            errors.append("diwa_target_tau must be in (0, 1]")
        if args.diwa_require_supervision and not args.diwa_supervision_path:
            errors.append(
                "diwa_require_supervision requires diwa_supervision_path"
            )
        if args.batch_size < 2 or args.batch_size % 2:
            errors.append(
                "cross-trajectory DIWA supervision requires even batch_size >= 2"
            )
        if not (
            0
            <= args.diwa_world_pretrain_steps
            <= args.diwa_counterfactual_start_steps
            <= args.diwa_regret_start_steps
        ):
            errors.append(
                "DIWA stages must satisfy world_pretrain <= "
                "counterfactual_start <= regret_start"
            )
        loss_names = (
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
        for name in loss_names:
            value = getattr(args, f"diwa_loss_{name}")
            if not math.isfinite(value) or value < 0:
                errors.append(f"diwa_loss_{name} must be finite and non-negative")
    if errors:
        raise ValueError("invalid DIWA configuration: " + "; ".join(errors))

def get_parser(is_eval=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        default='/mnt/petrelfs/share_data/robomani/calvin_data/task_ABCD_D',
        help="path to calvin_dataset",
    )
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # hot fix for torch.distributed.launch
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32", "bf16_and_fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    
    parser.add_argument(
        "--pred_num",
        default=1,
        type=int,
        help="the number of prediction include image, depth and trajectory.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=16)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
    )
    # data
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")
    # ceph
    parser.add_argument("--data_in_ceph",default=False, action="store_true")
    # oxe
    parser.add_argument("--root_dir", type=str, default="s3://real_data")
    parser.add_argument("--image_primary_size", type=int, default=200)
    parser.add_argument("--image_wrist_size", type=int, default=84)
    parser.add_argument("--finetune_type", type=str, default="",)   
    # save checkpoint
    parser.add_argument("--start_save_checkpoint", default=-1, type=int)
    parser.add_argument("--save_checkpoint", default=False, action="store_true")
    parser.add_argument("--save_checkpoint_path", required=True, type=str)
    parser.add_argument("--save_checkpoint_seq", type=int, default=1)
    # if validate
    parser.add_argument("--validation", default=False, action="store_true")
    # bf16 module
    parser.add_argument("--bf16_module", type=str, default="")
    # model structure 
    parser.add_argument("--sequence_length", type=int, default=10)
    # for image prediction
    parser.add_argument("--future_steps", type=int, default=3)
    parser.add_argument("--num_resampler_query", type=int, default=9)
    parser.add_argument("--num_obs_token_per_image", type=int, default=9)
    parser.add_argument("--calvin_input_image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    # droid
    parser.add_argument("--primary_mode", type=str, default="image_primary")
    parser.add_argument("--small_size", type=int, default=0)
    parser.add_argument("--dataset_info", type=str, default="droid_success")
    # pretrain
    parser.add_argument("--finetune_from_pretrained_ckpt", type=str, default=None)
    # loss
    parser.add_argument("--loss_arm_action_ratio", type=float, default=1.0)
    parser.add_argument("--loss_gripper_action_ratio", type=float, default=0.01)   
    # action_pred_steps
    parser.add_argument("--action_pred_steps", type=int, default=1)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--continuous_action_dim", type=int, default=6)
    parser.add_argument("--state_arm_dim", type=int, default=6)
    parser.add_argument("--state_gripper_dim", type=int, default=None)
    parser.add_argument("--dit_type", type=str, default="DiT-B")
    # obs_pred
    parser.add_argument("--obs_pred", default=False, action="store_true")
    parser.add_argument("--atten_only_obs", default=False, action="store_true")
    parser.add_argument("--attn_robot_proprio_state", default=False, action="store_true")
    parser.add_argument("--atten_goal", default=0, type=int)
    parser.add_argument("--atten_goal_state", default=False, action="store_true")
    
    # visual encoder
    parser.add_argument("--use_dinosiglip", default=False, action="store_true")
    
    # dit_head
    parser.add_argument("--use_dit_head", default=False, action="store_true")
    parser.add_argument("--use_fm", default=False, action="store_true")

    # Decision-Influential World Abstraction
    parser.add_argument("--use_diwa", default=False, action="store_true")
    parser.add_argument("--diwa_horizon", default=3, type=int)
    parser.add_argument("--max_grad_norm", default=0.1, type=float)
    parser.add_argument("--diwa_num_slots", default=16, type=int)
    parser.add_argument("--diwa_world_layers", default=2, type=int)
    parser.add_argument("--diwa_fusion_layers", default=2, type=int)
    parser.add_argument("--diwa_dropout", default=0.0, type=float)
    parser.add_argument(
        "--diwa_influence_temperature", default=0.25, type=float
    )
    parser.add_argument(
        "--diwa_counterfactual_samples", default=4, type=int
    )
    parser.add_argument("--diwa_entropy_weight", default=0.01, type=float)
    parser.add_argument("--diwa_sam_feature_dim", default=256, type=int)
    parser.add_argument("--diwa_slot_iterations", default=3, type=int)
    parser.add_argument(
        "--diwa_adaptive_budget",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--diwa_minimum_budget_ratio", default=0.0625, type=float
    )
    parser.add_argument("--diwa_budget_threshold", default=0.5, type=float)
    parser.add_argument("--diwa_budget_ratio", default=0.25, type=float)
    parser.add_argument("--diwa_budget_warmup_steps", default=1000, type=int)
    parser.add_argument("--diwa_budget_anneal_steps", default=10000, type=int)
    parser.add_argument("--diwa_teacher_forcing_steps", default=5000, type=int)
    parser.add_argument("--diwa_world_pretrain_steps", default=1000, type=int)
    parser.add_argument(
        "--diwa_counterfactual_start_steps", default=1000, type=int
    )
    parser.add_argument("--diwa_regret_start_steps", default=2000, type=int)
    parser.add_argument("--diwa_loss_proposal", default=0.1, type=float)
    parser.add_argument("--diwa_loss_mask", default=0.01, type=float)
    parser.add_argument("--diwa_loss_future", default=0.1, type=float)
    parser.add_argument("--diwa_loss_influence", default=0.1, type=float)
    parser.add_argument("--diwa_loss_budget", default=0.001, type=float)
    parser.add_argument("--diwa_loss_critic", default=0.1, type=float)
    parser.add_argument("--diwa_loss_progress", default=0.05, type=float)
    parser.add_argument("--diwa_loss_regret", default=0.01, type=float)
    parser.add_argument("--diwa_loss_contrastive", default=0.01, type=float)
    parser.add_argument("--diwa_critic_discount", default=0.99, type=float)
    parser.add_argument("--diwa_target_tau", default=0.005, type=float)
    parser.add_argument("--diwa_regret_candidates", default=6, type=int)
    parser.add_argument("--diwa_contrastive_margin", default=0.5, type=float)
    parser.add_argument(
        "--diwa_influence_policy_weight", default=1.0, type=float
    )
    parser.add_argument(
        "--diwa_influence_value_weight", default=1.0, type=float
    )
    parser.add_argument(
        "--diwa_influence_progress_weight", default=1.0, type=float
    )
    parser.add_argument(
        "--diwa_influence_covariance_weight", default=0.1, type=float
    )
    parser.add_argument("--diwa_influence_probes", default=4, type=int)
    parser.add_argument(
        "--diwa_require_supervision", default=False, action="store_true"
    )
    parser.add_argument("--diwa_supervision_path", type=str, default=None)
    parser.add_argument(
        "--diwa_profile",
        default=False,
        action="store_true",
        help="measure synchronized end-to-end DIWA evaluation latency/memory",
    )
    parser.add_argument(
        "--diwa_allow_legacy_checkpoint",
        default=False,
        action="store_true",
        help=(
            "allow a DIWA checkpoint without run_arguments and/or frozen-base "
            "compatibility metadata; tensor coverage is still validated"
        ),
    )
    parser.add_argument(
        "--diwa_profile_warmup_steps",
        default=5,
        type=int,
        help="rank-local policy updates excluded before profiling aggregation",
    )
    parser.add_argument(
        "--diwa_profile_output",
        type=str,
        default=None,
        help="optional rank-zero JSON path for the aggregated DIWA profile",
    )
    
    
    #depth pred
    parser.add_argument("--depth_pred", default=False, action="store_true")
    parser.add_argument("--use_depth_query", default=False, action="store_true")
    parser.add_argument("--use_dpt_head", default=False, action="store_true")

    # dino & sam pred
    parser.add_argument("--dino_feat_pred", default=False, action="store_true")
    parser.add_argument("--sam_feat_pred", default=False, action="store_true")
    
    # trajectory pred
    parser.add_argument("--trajectory_pred", default=False, action="store_true")
    parser.add_argument("--use_trajectory_query", default=False, action="store_true")
    parser.add_argument("--track_label_patch_size", type=int, default=8)
    parser.add_argument("--no_pred_gripper_traj", default=False, action="store_true")
    parser.add_argument("--no_unshuffle", default=False, action="store_true")
    parser.add_argument("--flow_as_mask", default=False, action="store_true")
    parser.add_argument("--share_query", default=False, action="store_true")
    parser.add_argument("--attn_implementation", default="eager", type=str)
    # 
    parser.add_argument("--use_gpt2_pretrained", default=False, action="store_true")
    
    # action mask ratio
    parser.add_argument("--mask_l_obs_ratio", default=0.00, type=float)
    # reset during finetuning
    parser.add_argument("--reset_action_token", default=False, action="store_true")
    parser.add_argument("--reset_obs_token", default=False, action="store_true")
    parser.add_argument("--reset_mask_token", default=False, action="store_true")
    parser.add_argument("--reset_image_decoder", default=False, action="store_true")
    parser.add_argument("--reset_action_decoder", default=False, action="store_true")
    parser.add_argument("--reset_resampler", default=False, action="store_true")
    # loss
    parser.add_argument("--loss_action", default=False, action="store_true")
    parser.add_argument("--loss_image", default=False, action="store_true")
    parser.add_argument("--loss_depth", default=False, action="store_true")
    parser.add_argument("--loss_dino_feat", default=False, action="store_true")
    parser.add_argument("--loss_sam_feat", default=False, action="store_true")

    parser.add_argument("--loss_trajectory", default=False, action="store_true")
    
    # calvin
    parser.add_argument("--except_lang", default=False, action="store_true")
    parser.add_argument("--load_track_labels", default=False, action="store_true")
    parser.add_argument("--track_label_path", type=str, default=None)
    parser.add_argument("--load_dino_features", default=False, action="store_true")
    parser.add_argument("--dino_features_path", type=str, default=None)
    parser.add_argument("--load_sam_features", default=False, action="store_true")
    parser.add_argument("--sam_features_path", type=str, default=None)
    parser.add_argument("--merge_data", default=False, action="store_true")
    
    # gpt2
    parser.add_argument("--transformer_layers", default=12, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument("--transformer_heads", default=12, type=int)
    # pretrain, finetune, evaluate
    parser.add_argument('--phase', required=True, help='pretrain, finetune, evaluate')
    # libero 
    parser.add_argument("--libero_path", default="/ailab/user/tianyang/Code/LIBERO")
    parser.add_argument(
        "--libero_dataset_name",
        default="libero_10_converted",
        help="converted LIBERO dataset directory name under --root_dir",
    )
    parser.add_argument("--libero_img_size", default=128, type=int)
    parser.add_argument("--libero_eval_max_steps", default=600, type=int)
    parser.add_argument(
        "--libero_eval_episodes",
        default=20,
        type=int,
        help="number of initial states evaluated per LIBERO task",
    )
    parser.add_argument(
        "--libero_eval_task_count",
        default=10,
        type=int,
        help="number of tasks from the selected LIBERO suite to evaluate",
    )
    parser.add_argument("--gripper_width", default=False, action="store_true")
    parser.add_argument("--load_libero_file", type=str, default="h5")
    parser.add_argument("--eval_libero_ensembling", default=False, action="store_true")
    parser.add_argument("--ensembling_temp", default=0.01, type=float)
    # real
    parser.add_argument("--real_dataset_names", type=str)
    parser.add_argument(
        "--real_dataset_adapter",
        type=str,
        default=None,
        help=(
            "private real-robot dataset class as module.path:ClassName; "
            "the public release does not bundle private robot data loaders"
        ),
    )
    parser.add_argument("--use_aug_data", default=False, action="store_true")
    parser.add_argument("--real_eval_max_steps", default=600, type=int)
    # preprocess
    parser.add_argument("--max_rel_pos", type=float, default=0.02)
    parser.add_argument("--max_rel_orn", type=float, default=0.05)
    parser.add_argument("--magic_scaling_factor_pos", type=float, default=1.0)
    parser.add_argument("--magic_scaling_factor_orn", type=float, default=1.0)
    # for eval
    if is_eval:
        parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
        parser.add_argument("--future_act_len", default=-1, type=int)
        parser.add_argument(
            "--visualize",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--reset",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--diverse_inst",
            default=False,
            action="store_true"
        )
        parser.add_argument("--pad_length", type=int, default=-1)
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--vit_checkpoint_path", type=str)

    return parser

    # if args.dataloading_type == "seer":
    #     if args.phase == "pretrain":
    #         if args.finetune_type == "calvin":
    #             args.window_size = args.sequence_length + args.future_steps 
    #         else:
    #             args.window_size = args.sequence_length
    #     elif args.phase == "finetune":
    #         args.window_size = args.sequence_length + args.future_steps
