#!/bin/bash
set -euo pipefail

checkpoint="${CHECKPOINT:?set CHECKPOINT}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:?set VIT_CHECKPOINT_PATH}"
libero_path="${LIBERO_PATH:?set LIBERO_PATH}"
num_gpus="${NUM_GPUS:-8}"
master_port="${MASTER_PORT:-10143}"
profile_args=()
if [[ "${DIWA_PROFILE:-0}" == "1" ]]; then
    profile_args+=(
        --diwa_profile
        --diwa_profile_warmup_steps "${DIWA_PROFILE_WARMUP_STEPS:-5}"
    )
    if [[ -n "${DIWA_PROFILE_OUTPUT:-}" ]]; then
        profile_args+=(--diwa_profile_output "${DIWA_PROFILE_OUTPUT}")
    fi
fi

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${num_gpus}" \
    --master_port="${master_port}" \
    eval_libero.py \
    --save_checkpoint_path evaluate \
    --vit_checkpoint_path "${vit_checkpoint_path}" \
    --libero_path "${libero_path}" \
    --seed 66 \
    --precision fp32 \
    --bf16_module vision_encoder \
    --finetune_type libero_10 \
    --num_resampler_query 16 \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase evaluate \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --window_size 10 \
    --gripper_width \
    --eval_libero_ensembling \
    --use_dit_head \
    --dit_type DiT-B \
    --attn_implementation sdpa \
    --use_diwa \
    --diwa_horizon 3 \
    --diwa_num_slots 16 \
    --diwa_world_layers 2 \
    --diwa_fusion_layers 2 \
    --diwa_budget_ratio 0.25 \
    --diwa_minimum_budget_ratio 0.0625 \
    --diwa_slot_iterations 3 \
    --libero_eval_episodes 50 \
    --diwa_adaptive_budget \
    --resume_from_checkpoint "${checkpoint}" \
    "${profile_args[@]}"
