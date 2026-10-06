# Decision-Influential World Abstraction implementation

This repository implements DIWA as an opt-in DreamVLA policy path behind
`--use_diwa`. The original DreamVLA path remains unchanged when the flag is
absent.

## Implemented model

- Structured context tokens: language/state, global scene, temporally matched
  object slots and future-time embeddings.
- Object-centric encoding with Slot Attention. DreamVLA visual tokens are
  always used; current-frame SAM dense features are fused when provided.
  Precomputed forward CoTracker tracks only enter the detached EMA target
  encoder, never the online policy encoder.
- Sinkhorn matching for consistent slot identity across observations.
- An action-conditioned latent world decoder over `horizon × object` queries.
- A decision-influence estimator that runs before future expansion.
- Mask and same-task/cross-episode token-swap interventions.
- Continuous-action influence targets based on action mean and covariance
  changes. Diffusion and flow heads use common noise and timesteps so the
  measured difference is caused by the intervention rather than sampling
  noise.
- Twin Q critics, target critics and a task-progress head. Critics learn from
  both TD targets and externally measured candidate-action Q values.
- Regret-vector latent geometry and optimal-action contrastive learning using
  measured candidate Q values.
- Full-capacity, compressed and stop-gradient capacity levels for high,
  medium and very-low influence variables.
- A learned per-state compute budget with a configured maximum and minimum.
- True sparse inference: only selected future queries are passed to the world
  decoder. Unselected queries are not materialized by that decoder.
- EMA updates for target critics and the future object-target encoder.
- Strict DIWA checkpoint validation during resume and evaluation.
- Headless LIBERO EGL defaults and an in-memory robosuite 1.4/MuJoCo 3
  mass-matrix compatibility shim for environments where MuJoCo 2 wheels are
  unavailable.

## Losses

The total training loss contains the existing DreamVLA action loss and these
DIWA terms:

| Code key | Purpose |
| --- | --- |
| `proposal` | Action proposal arm regression and gripper BCE |
| `mask` | New calibration term: learned mask to detached valid world-token centroid |
| `future` | Influence-weighted, identity-aligned future object prediction |
| `influence` | Counterfactual action-distribution, Q and progress change |
| `budget` | Mean activation, adaptive allocation, ceiling and gate entropy |
| `critic` | Twin TD loss plus supervised measured candidate-Q loss |
| `progress` | Task-progress BCE |
| `regret` | Latent distance to stop-gradient regret-vector L1 distance |
| `contrastive` | Decision-equivalent/decision-different logistic contrastive loss |

`candidate_q_diagnostic`, `influence_policy`, `influence_value` and
`influence_progress` are logged diagnostics and are not added a second time to
the objective.

Training is staged. Action proposal, mask calibration and future prediction start immediately;
critic/progress begin at `--diwa_world_pretrain_steps`; influence and sparse
budget begin at `--diwa_counterfactual_start_steps`; regret and contrastive
learning begin at `--diwa_regret_start_steps`. The maximum sparse budget is
annealed after warm-up and action conditioning transitions from demonstration
actions to learned proposals.

Mask calibration has a default weight of 0.01. It learns the replacement
without relaxing stop-gradient influence targets; its exact objective and
weight were introduced in this source repair and are not attributed to
historical measured experiments. LIBERO uses same-task cross-episode paired
batch sampling per rank. The offline runner explicitly disables critic,
progress and regret objectives when candidate returns are unavailable.
See [REVIEW_FIXES.md](REVIEW_FIXES.md) for the contracts and migration details.

## Required measured supervision

The strict training script passes `--diwa_require_supervision`. Every source
episode must then provide all of the following:

| Field | Shape | Meaning |
| --- | --- | --- |
| `reward` | `[T]` | Simulator/environment reward |
| `done` | `[T]` | True episode termination |
| `progress` | `[T]` | Task-predicate progress in `[0, 1]` |
| `candidate_actions` | `[T, M, A, D]` | Aligned candidate action/policy bank |
| `candidate_q_values` | `[T, M]` | Measured discounted return of each candidate |
| `candidate_rule_ids` | `[M]` strings | Stable identities in canonical candidate order |

`A` must equal `--action_pred_steps`, and `D` is the configured action
dimension. The LIBERO loader converts its native final `-1/+1` gripper
coordinate to the policy's `0/1` convention. Cross-simulator adapters define
the equivalent boundary conversion for their native action layout. Candidate
index `m` must identify the same candidate policy or action-generation rule
across paired states within a task. The packager, converted-data loader and
sidecar loader now require the exact canonical ID vector and reject missing,
duplicate or reordered rules. Q values must come from simulator
rollouts, task predicates, human annotation or an independently audited value
pipeline. They must not be predictions from the DIWA critic being trained.

No code path synthesizes reward, progress or Q labels from frame indices.
Strict mode raises immediately when any measured field is missing or malformed.

Generate measured labels directly from official LIBERO demonstrations by
restoring every recorded simulator state and branching candidate action
chunks:

```bash
python data_process/collect_libero_diwa_supervision.py \
  --libero-path /path/to/LIBERO \
  --suite libero_spatial \
  --dataset-dir /path/to/libero_spatial \
  --output /path/to/measured_libero_spatial \
  --action-pred-steps 3 \
  --num-candidates 6
```

Candidate 0 replays the demonstrated action chunk. The remaining stable
candidate rules are no-op and signed Cartesian perturbations. Before every
candidate branch, the collector resets episode counters and then restores the
same serialized simulator state, so elapsed-step and termination state cannot
leak between candidates. Candidate returns are measured from those branches.
The optional task-predicate shaping term is measured from LIBERO success
predicates and is never inferred from episode time.

Package one measured `<episode_id>.npz` per episode into loader sidecars:

```bash
python data_process/build_diwa_supervision.py \
  --source /path/to/measured_episode_npz \
  --output /path/to/diwa_supervision
```

The resulting layout is:

```text
diwa_supervision/
  000001/
    steps/
      0000.npz
      0001.npz
```

The LIBERO conversion utility also preserves all six fields when they already
exist in the source HDF5 demo. It rejects partially populated supervision.
It is fully command-line driven:

```bash
python utils/convert_libero_per_step.py \
  --dataset-name libero_spatial \
  --src-dir /path/to/libero_spatial \
  --tgt-dir /path/to/libero_spatial_converted \
  --num-workers 1
```

## Training

```bash
export SAVE_CHECKPOINT_PATH=/path/to/output
export ROOT_DIR=/path/to/converted/libero/data
export LIBERO_DATASET_NAME=libero_10_converted
export VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth
export PRETRAINED_CHECKPOINT=/path/to/dreamvla/checkpoint.pth
export DIWA_SUPERVISION_PATH=/path/to/diwa_supervision
export NUM_GPUS=8

bash scripts/LIBERO/DIWA/train_latent_diwa.sh
```

The provided script supervises a 7-step policy sequence and exposes up to 3
future observations (`window_size=10`) for future-object targets. Regular
windows use real lookahead observations. Terminal-aligned windows repeat the
last observation only to form a fixed-shape tensor and mark unavailable
lookahead positions invalid, so future losses ignore them while terminal
reward, done and progress labels remain supervised.

## Evaluation and metrics

```bash
export CHECKPOINT=/path/to/diwa/checkpoint.pth
export VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth
export LIBERO_PATH=/path/to/LIBERO
export DIWA_PROFILE=1  # optional
export DIWA_PROFILE_WARMUP_STEPS=5
export DIWA_PROFILE_OUTPUT=/path/to/profile.json  # optional

bash scripts/LIBERO/DIWA/eval_latent_diwa.sh
```

Set `DIWA_PROFILE=1` for the provided script, or add `--diwa_profile` to a
direct evaluation command. Every rank drops its own configured warm-up
updates, then rank zero aggregates synchronized end-to-end samples and reports
mean/P95 latency, action frequency, peak allocated GPU memory, expanded-token
count and expanded-token ratio. `--diwa_profile_output` writes these metrics
and the checkpoint, device, precision and software metadata as strict JSON.

`data_process/evaluate_diwa_metrics.py` computes the paper's offline metrics
from an NPZ archive:

```bash
python data_process/evaluate_diwa_metrics.py \
  --input /path/to/diwa_evaluation.npz \
  --output /path/to/metrics.json
```

Required arrays are `influence_scores`, `measured_influence` and
`selected_mask`. Optional paired arrays enable Top-K action consistency,
regret-distance rank correlation, optimal-action retrieval, counterfactual
decision accuracy, intervention return drop, separate standard/OOD success,
task/subgoal success, latency and memory metrics. No standard-to-OOD retention
ratio is emitted because those arrays may represent different populations.
See `OPTIONAL_KEYS` in the evaluator for the exact interface. All optional
measurements are checked for finite values and valid ranges; undefined
tokens-per-success is serialized as JSON `null`.

Full DreamVLA checkpoints are checked for every trainable parameter, all DIWA
state, tensor shape/dtype, unknown keys and model-construction arguments. Each
new checkpoint also stores a SHA-256 signature of frozen parameters omitted
from `model_state_dict`; resume and evaluation recompute it and reject a
different vision/base model. Resume additionally checks optimizer, loss,
schedule, precision, batch/world-size and staged-DIWA arguments, and restores
Python, NumPy, Torch CPU and Torch CUDA RNG state separately for every rank.
Checkpoint replacement is atomic. Transfer initialization requires every
non-DIWA trainable tensor except heads explicitly reset on the command line;
an unrelated or incomplete pretrained file is rejected before loading.
The current full-checkpoint schema is version 4 and records terminal-aware
windows, causal label isolation, paired sampling and mask calibration.
Legacy checkpoints missing `run_arguments`,
per-rank RNG state, the DIWA `frozen_parameter_signature`, or the current
schema are rejected for resume by default. Structurally compatible legacy
model weights remain loadable for evaluation. The explicit
`--diwa_allow_legacy_checkpoint` escape hatch should be used only after
independently verifying the architecture, training schedule and frozen base
weights.

## Verification

```bash
pytest -q tests
python -m py_compile \
  models/diwa/*.py models/dreamvla_model.py \
  data_process/build_diwa_supervision.py \
  data_process/evaluate_diwa_metrics.py
```

The tests exercise sparse selection, adaptive budgets, object binding,
diffusion/flow common-randomness moments and exact flow integration steps,
measured-Q critic/regret losses, terminal supervision and future-valid masks,
actual DataLoader batch accounting, checkpoint completeness, supervision
packaging and evaluation metrics.

For an end-to-end GPU smoke test with real observations and measured branches
from the LIBERO simulator:

```bash
python scripts/LIBERO/DIWA/smoke_test_libero.py \
  --libero-path /path/to/LIBERO \
  --task-id 0 \
  --output /path/to/diwa_libero_smoke.json
```

This instantiates the complete DreamVLA-DIWA path, runs a finite
forward/backward/optimizer update using two LIBERO initial states, verifies
influence-estimator gradients, updates EMA targets, and then checks that
inference expands fewer than all available future tokens. It intentionally
uses random vision and policy weights, so it validates integration rather
than success rate.

The smoke report includes the measured candidate returns, every optimized
auxiliary loss, the influence-estimator gradient norm, synchronized inference
latency, selected/available token counts, and peak allocated CUDA memory. The
script prints `DIWA_LIBERO_SMOKE_OK` only after all checks pass.

The manuscript ablations have a separate real-simulator smoke entry point:

```bash
python scripts/LIBERO/DIWA/smoke_test_libero_ablations.py \
  --libero-path /path/to/LIBERO \
  --task-id 0 \
  --output /path/to/diwa_libero_ablation_smoke.json
```

It runs train updates for full DIWA, no influence estimator, no
cross-trajectory swap, and no regret geometry, plus inference for dense
imagination and uniform Top-K. The report records included/excluded losses,
influence gradient norms, actual swap counts, selected ratios, expanded token
counts, latency and peak memory. Success is marked by
`DIWA_LIBERO_ABLATION_SMOKE_OK`.

The generalized policy and action heads also have real-data/environment smoke
adapters for CALVIN, RoboCasa and RoboTwin:

```bash
python scripts/MULTI_DATASET/DIWA/run_exported_smoke.py \
  --input /path/to/<dataset>_smoke_batch.npz \
  --action-output /path/to/<dataset>_eval_action.npy \
  --report /path/to/<dataset>_policy_smoke.json \
  --device cuda
```

The dataset-specific exporter is run first in the simulator's isolated Python
environment. After the optimizer update, the same adapter loads the saved
policy action and executes it in a fresh official simulator instance. CALVIN
uses a 7-D action, RoboCasa a 12-D mobile-manipulator action, and RoboTwin a
14-D dual-arm action. See
[`scripts/MULTI_DATASET/DIWA/README.md`](../scripts/MULTI_DATASET/DIWA/README.md)
for the exact export and environment-step commands.

These checks require finite observations/actions/losses, a real optimizer
update with non-zero influence gradients, sparse inference, and observable
robot-state change after the submitted action. They are integration checks;
one update from random weights is not expected to solve the task.

CALVIN, RoboCasa and RoboTwin exports additionally set
`require_measured_candidate_q=true`. Their candidate Q labels are discounted
returns from restored or rebuilt official simulator branches using measured
task-predicate progress and success. The shared runner fails on NaNs or a
candidate bank that never changes the measured return; its report records the
finite ratio, maximum candidate spread and nonconstant-state fraction.
Offline DROID/OXE archives explicitly declare counterfactual Q unavailable
instead of substituting a critic prediction or fabricated label.

The two remaining public training families supported directly by `train.py`
are DROID and OXE. Their offline suite covers DROID plus every one of the 12
OXE component datasets configured in `get_oxe_dataset`:

```bash
python scripts/MULTI_DATASET/DIWA/run_offline_dataset_suite.py \
  --cache /path/to/offline_cache \
  --outputs /path/to/outputs \
  --device cuda --resume
```

Each component uses real public observations and actions and receives its own
training and inference report. Because DROID/OXE are recorded real-robot data,
their smoke evaluation is offline finite-action inference rather than a
simulator success rollout. The `real` loader is intentionally not included:
it names a user-supplied private dataset through `--real_dataset_names`, not a
specific downloadable public benchmark. To use that branch, pass the private
dataset class explicitly as `--real_dataset_adapter module.path:ClassName`;
otherwise the entry point stops with an actionable error instead of silently
assuming an unavailable schema.

RoboCasa also provides a complete OOD smoke pipeline. It modifies the live
MuJoCo model for texture/material, lighting, moving background and contact
friction shifts, and constructs the official unseen kitchen layout/style pair
`(6, 9)`. `robocasa_ood_adapter.py collect` gathers new observations from all
six environments; `run_ood_policy_smoke.py` runs one saved trained policy on
them; `robocasa_ood_adapter.py evaluate` reconstructs every variant, submits
the predicted action, and writes per-variant reward/progress/success/contact
metrics. Exact commands and output contracts are documented in
[`scripts/MULTI_DATASET/DIWA/README.md`](../scripts/MULTI_DATASET/DIWA/README.md).

This is a complete code implementation, not a claim that the manuscript's
reported experimental numbers have been reproduced. Reproduction still
requires the DreamVLA/vision checkpoints, converted benchmark data, measured
rollout supervision and GPU training/evaluation.
