#!/bin/bash
set -euo pipefail

# Required paths.
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:?set SAVE_CHECKPOINT_PATH}"
root_dir="${ROOT_DIR:?set ROOT_DIR to the converted LIBERO data parent}"
libero_dataset_name="${LIBERO_DATASET_NAME:-libero_10_converted}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:?set VIT_CHECKPOINT_PATH}"
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:?set PRETRAINED_CHECKPOINT}"
diwa_supervision_path="${DIWA_SUPERVISION_PATH:?set DIWA_SUPERVISION_PATH}"

num_gpus="${NUM_GPUS:-8}"
master_port="${MASTER_PORT:-10221}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${num_gpus}" \
    --master_port="${master_port}" \
    train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path "${vit_checkpoint_path}" \
    --workers 16 \
    --lr_scheduler cosine \
    --num_epochs 40 \
    --seed "${SEED:-42}" \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --save_checkpoint_seq 1 \
    --finetune_type libero_finetune \
    --root_dir "${root_dir}" \
    --libero_dataset_name "${libero_dataset_name}" \
    --diwa_supervision_path "${diwa_supervision_path}" \
    --diwa_require_supervision \
    --finetune_from_pretrained_ckpt "${pretrained_checkpoint}" \
    --run_name "libero_latent_diwa_seed${SEED:-42}" \
    --save_checkpoint_path "${save_checkpoint_path}" \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase finetune \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --window_size 10 \
    --loss_action \
    --gripper_width \
    --warmup_epochs 5 \
    --use_dit_head \
    --dit_type DiT-B \
    --attn_implementation sdpa \
    --use_diwa \
    --diwa_horizon 3 \
    --diwa_num_slots 16 \
    --diwa_world_layers 2 \
    --diwa_fusion_layers 2 \
    --diwa_budget_ratio 0.25 \
    --diwa_budget_warmup_steps 1000 \
    --diwa_budget_anneal_steps 10000 \
    --diwa_teacher_forcing_steps 5000 \
    --diwa_counterfactual_samples 4 \
    --diwa_influence_probes 4 \
    --diwa_influence_temperature 0.25 \
    --diwa_minimum_budget_ratio 0.0625 \
    --diwa_budget_threshold 0.5 \
    --diwa_entropy_weight 0.01 \
    --diwa_loss_mask 0.01 \
    --diwa_slot_iterations 3 \
    --diwa_critic_discount 0.99 \
    --diwa_target_tau 0.005 \
    --diwa_regret_candidates 6 \
    --diwa_contrastive_margin 0.5 \
    --diwa_influence_covariance_weight 0.1 \
    --diwa_influence_policy_weight 1.0 \
    --diwa_influence_value_weight 1.0 \
    --diwa_influence_progress_weight 1.0 \
    --diwa_world_pretrain_steps 1000 \
    --diwa_counterfactual_start_steps 1000 \
    --diwa_regret_start_steps 2000 \
    --diwa_loss_proposal 0.1 \
    --diwa_loss_future 0.1 \
    --diwa_loss_influence 0.1 \
    --diwa_loss_budget 0.001 \
    --diwa_loss_critic 0.1 \
    --diwa_loss_progress 0.05 \
    --diwa_loss_regret 0.01 \
    --diwa_loss_contrastive 0.01 \
    --diwa_adaptive_budget
