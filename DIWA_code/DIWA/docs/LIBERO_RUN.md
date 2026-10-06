# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [huggingface](https://huggingface.co/WenyaoZhang/DreamVLA).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.
    * **libero_path** to the location of LIBERO dir.

# Data Processing
### Convert Data
Pass the official demonstration and output paths on the command line:
```bash
python utils/convert_libero_per_step.py \
  --dataset-name libero_10 \
  --src-dir /path/to/libero_10 \
  --tgt-dir /path/to/libero_10_converted \
  --num-workers 1
```
It will generate converted data in *tgt_dir*.

For DIWA, measure simulator-backed candidate returns and package them as
sidecars before training:

```bash
python data_process/collect_libero_diwa_supervision.py \
  --libero-path /path/to/LIBERO \
  --suite libero_10 \
  --dataset-dir /path/to/libero_10 \
  --output /path/to/measured_libero_10

python data_process/build_diwa_supervision.py \
  --source /path/to/measured_libero_10 \
  --output /path/to/diwa_supervision
```

Each measured episode and every generated sidecar contains
`candidate_rule_ids`. For the six-rule paper configuration the required order
is `demonstration`, `no_motion_keep_gripper`, `plus_x`, `minus_x`, `plus_y`,
`minus_y`. Packaging and loading fail if this identity vector is absent or
reordered. The collector resets the environment before restoring the recorded
state for every branch, including candidate 0, so episode timers and terminal
flags do not carry across candidate measurements.

Training windows retain the final valid policy states. When fewer than three
future observations remain, the loader pads the missing lookahead positions
with the last observation and marks them invalid for the future-prediction
loss; measured terminal reward, done and progress labels remain active.

Headless servers should use EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The collector, evaluator and smoke test set these values by default. See
`docs/LIBERO_INSTALL.md` for the supported MuJoCo/robosuite combinations.

### Dynamic Region:  
Install [co-tracker](https://github.com/facebookresearch/co-tracker.git). Note download the [checkpoints of co-tracker](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth) and put it to ```./co-tracker/checkpoints```
```bash
mv ./data_process/cotrack_extractor.py ./co-tracker/
cd co-tracker
torchrun --nproc_per_node=8 cotrack_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/cotracker_traj
```

### SAM Feature: 
Install [SAM](https://github.com/facebookresearch/segment-anything). Note download the [checkpoints of SAM](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth) and put it to ```./segment-anything/ckpts```.
```bash
cp dist_utils.py ./segment-anything/
mv ./data_info/ep_start_end_ids.npy <your_data_path>
mv ./data_process/sam_extractor.py ./segment-anything/
cd segment-anything
torchrun --nproc_per_node=8 sam_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/sam_feats
```

### DINOv2 Feature: 

Install [DINOV2](https://github.com/facebookresearch/dinov2). Note download the [checkpoints of dinov2]( https://huggingface.co/junjiexv/dinov2_vit/blob/main/dinov2_vits14_pretrain.pth) and put it to ```./dinov2/ckpts```.
```bash
cp dist_utils.py ./dinov2/
mv ./data_process/dino_extractor.py ./dinov2/
cd dinov2
torchrun --nproc_per_node=8 dino_extractor_libero.py --data_root ${tgt_dir}/episodes --save_path ${tgt_dir}/dinov2_feats
```

# Training
### Pre-train
```bash
# Pre-train DreamVLA on LIBERO-90 dataset
bash scripts/LIBERO/DreamVLA/pretrain.sh
```
You also can load the pretrained weights from 

### Fine-tune
```bash
# Fine-tune DreamVLA on LIBERO dataset
bash scripts/LIBERO/DreamVLA/finetune_long.sh
bash scripts/LIBERO/DreamVLA/finetune_object.sh
bash scripts/LIBERO/DreamVLA/finetune_spatial.sh
bash scripts/LIBERO/DreamVLA/finetune_goal.sh
```

### Train from Scratch
```bash
# Train DreamVLA on LIBERO dataset from scratch
bash scripts/LIBERO/DreamVLA/scratch_long.sh
bash scripts/LIBERO/DreamVLA/scratch_object.sh
bash scripts/LIBERO/DreamVLA/scratch_spatial.sh
bash scripts/LIBERO/DreamVLA/scratch_goal.sh
```


### Eval
You can download checkpoints from [huggingface](https://huggingface.co/WenyaoZhang/DreamVLA)
```bash
# Evaluate DreamVLA on LIBERO benchmark
bash scripts/LIBERO/DreamVLA/eval_long.sh
bash scripts/LIBERO/DreamVLA/eval_object.sh
bash scripts/LIBERO/DreamVLA/eval_spatial.sh
bash scripts/LIBERO/DreamVLA/eval_goal.sh
```

### DIWA smoke test

This command uses a real LIBERO_SPATIAL environment and two official initial
states. It measures candidate returns by restoring the simulator state,
performs a full DreamVLA-DIWA forward/backward/optimizer/EMA update, and
verifies genuinely sparse inference:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/LIBERO/DIWA/smoke_test_libero.py \
  --libero-path /path/to/LIBERO-parent \
  --task-id 0 \
  --batch-size 2 \
  --image-size 64 \
  --device cuda \
  --output outputs/diwa_libero_smoke.json
```

Success is reported only after the JSON artifact has been written and the
script prints `DIWA_LIBERO_SMOKE_OK`. Random model weights are intentional:
this is an integration smoke test, not a benchmark-success claim.

Run the paper's five ablation execution paths on the same real LIBERO task:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  scripts/LIBERO/DIWA/smoke_test_libero_ablations.py \
  --libero-path /path/to/LIBERO-parent \
  --task-id 0 \
  --batch-size 2 \
  --image-size 64 \
  --device cuda \
  --output outputs/diwa_libero_ablation_smoke.json
```

The command executes full DIWA and these ablations:

- no decision-influence estimator: uniform future weighting, no estimator
  gradient and dense selection;
- no cross-trajectory counterfactual swap: mask interventions remain active;
- no regret geometry: regret and decision-contrastive terms are excluded;
- no sparse imagination: every future token is expanded;
- uniform Top-K: a fixed sparse budget is selected from equal scores.

Every training variant performs a finite forward/backward/optimizer/target
update. Inference variants verify dense or sparse token counts. The script
prints `DIWA_LIBERO_ABLATION_SMOKE_OK` only when every contract passes. This is
an execution smoke test; random-weight losses and latencies are not an
ablation performance comparison.
