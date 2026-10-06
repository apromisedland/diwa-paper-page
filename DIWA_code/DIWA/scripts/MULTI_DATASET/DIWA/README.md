# Cross-simulator smoke protocol

Each simulator is installed in its own environment and uses a small adapter to
export two real episodes with two supervised policy states and one future
observation. `run_exported_smoke.py` then performs one full DreamVLA-DIWA
optimizer update and sparse evaluation in the main environment. Finally, the
adapter loads the saved action and submits it to a fresh simulator instance.

An NPZ export contains RGB observations from the primary and wrist cameras,
proprioception, action chunks, measured rewards/dones/progress, candidate
actions, and candidate returns. Continuous action dimensions use `[-1, 1]`;
gripper dimensions use the policy convention `[0, 1]`. Environment adapters
are responsible for conversion to and from their native action convention.
Every export also carries a unique, ordered `candidate_rule_ids` vector. The
common runner checks this vector against the adapter contract before using
candidate returns, preventing Q values from being paired with reordered
intervention rules.

This is an integration smoke test. Random initialization and one optimizer
step do not measure task success or benchmark quality.

DROID and OXE are offline real-robot datasets, not reproducible simulators.
For those sources, the final stage is finite policy inference plus action
contract metrics rather than an environment step.

For CALVIN, RoboCasa and RoboTwin, `require_measured_candidate_q=true` is part
of the batch contract. The common runner rejects NaN candidate returns,
rejects batches with no candidate-dependent ranking, and reports the finite
ratio, maximum spread and nonconstant-state fraction. The labels come from
discounted task-predicate returns measured after restoring or rebuilding the
same simulator state for every candidate. DROID/OXE do not pretend to provide
counterfactual simulator labels: their batch marks that supervision as
unavailable. The runner explicitly uses `offline_imitation`: critic,
candidate-Q regression, progress, regret and contrastive losses are disabled,
as are Q/progress contributions to influence targets. Missing candidate Q
and progress remain NaN; predictions never substitute for those labels.
Policy imitation, proposal, future, policy influence, budget and mask
calibration stay active. Reports list the active and disabled objectives.
Each offline episode retains its own language/task identity. The simulator
datasets cannot opt out of measured-Q validation by changing the flag.

## Covered simulators

| Dataset | Source used by export | Canonical action | Official environment check |
| --- | --- | ---: | --- |
| CALVIN | two episodes from the official debug demonstrations | 7-D | one `PlayTableSimEnv.step` |
| RoboCasa | two public target-human `OpenCabinet` episodes | 12-D | one `robocasa/OpenCabinet` step |
| RoboTwin | two simulator-generated `beat_block_hammer` transitions | 14-D dual arm | one official `Base_Task.take_action` call |

The adapters preserve the native environment convention at the boundary and
place continuous coordinates before gripper/mode coordinates in the NPZ
contract. This lets the same policy smoke runner validate single-arm,
mobile-manipulator and dual-arm action spaces.

## DROID and every configured OXE source

`run_offline_dataset_suite.py` covers DROID and all 12 datasets hard-coded in
DreamVLA's `get_oxe_dataset`: Berkeley Autolab UR5, Jaco Play, CMU
Pickup-Insert, VIOLA, Stanford HYDRA, Berkeley FANUC, Austin BUDS, UT Austin
MUTEX, TACO Play, Austin SAILOR, Austin Sirius and FurnitureBench.

The suite downloads only the first packed parquet/metadata files from public
LeRobot conversions of the official releases. FFmpeg reads the required
camera frames directly from the packed public videos with HTTP range requests,
so multi-gigabyte video archives are not copied in full. Each source gets its
own real two-episode NPZ, optimizer update, sparse inference pass and offline
evaluation report.

```bash
python scripts/MULTI_DATASET/DIWA/run_offline_dataset_suite.py \
  --cache /path/to/offline_cache \
  --outputs /path/to/outputs \
  --device cuda
```

Use `--resume` to retain successful per-dataset reports after an interrupted
download or run. The aggregate report is
`diwa_offline_dataset_suite.json`; completion is marked by
`DIWA_ALL_OFFLINE_DATASETS_SMOKE_OK`.

## CALVIN

Run export and environment evaluation in an isolated official CALVIN
environment, and run the optimizer update in the main DreamVLA environment:

```bash
/path/to/calvin/python calvin_adapter.py export \
  --dataset /path/to/calvin_lerobot_debug \
  --calvin-env /path/to/calvin/calvin_env \
  --output /path/to/outputs/calvin_smoke_batch.npz

python run_exported_smoke.py \
  --input /path/to/outputs/calvin_smoke_batch.npz \
  --action-output /path/to/outputs/calvin_eval_action.npy \
  --report /path/to/outputs/calvin_policy_smoke.json \
  --device cuda

/path/to/calvin/python calvin_adapter.py evaluate \
  --calvin-env /path/to/calvin \
  --action /path/to/outputs/calvin_eval_action.npy \
  --report /path/to/outputs/calvin_env_smoke.json
```

## RoboCasa

The export expects the public LeRobot conversion of RoboCasa target-human
demonstrations. The evaluation environment must have the official RoboCasa
fixture, texture and object assets installed.

```bash
/path/to/robocasa/python robocasa_adapter.py export \
  --dataset /path/to/robocasa_target_human_unified \
  --output /path/to/outputs/robocasa_smoke_batch.npz

python run_exported_smoke.py \
  --input /path/to/outputs/robocasa_smoke_batch.npz \
  --action-output /path/to/outputs/robocasa_eval_action.npy \
  --report /path/to/outputs/robocasa_policy_smoke.json \
  --checkpoint-output /path/to/outputs/robocasa_model.pt \
  --device cuda

/path/to/robocasa/python robocasa_adapter.py evaluate \
  --action /path/to/outputs/robocasa_eval_action.npy \
  --report /path/to/outputs/robocasa_env_smoke.json
```

## RoboTwin

The selected official task needs the ALOHA-AgileX embodiment, the
`020_hammer` object, and the object metadata files `objects/same.json` and
`objects/objaverse/list.json`. cuRobo is not needed for this qpos smoke: the
adapter injects linear trajectory timing while retaining RoboTwin's official
SAPIEN scene, robot drive and `Base_Task.take_action` control loop.

```bash
/path/to/robotwin/python robotwin_adapter.py export \
  --robotwin /path/to/RoboTwin \
  --output /path/to/outputs/robotwin_smoke_batch.npz

python run_exported_smoke.py \
  --input /path/to/outputs/robotwin_smoke_batch.npz \
  --action-output /path/to/outputs/robotwin_eval_action.npy \
  --report /path/to/outputs/robotwin_policy_smoke.json \
  --device cuda

/path/to/robotwin/python robotwin_adapter.py evaluate \
  --robotwin /path/to/RoboTwin \
  --action /path/to/outputs/robotwin_eval_action.npy \
  --report /path/to/outputs/robotwin_env_smoke.json
```

## End-to-end OOD environment smoke

`robocasa_ood_adapter.py` generates six actual MuJoCo environment variants:
baseline, material/texture shift, lighting shift, moving background geometry,
official unseen layout/style pair `(6, 9)`, and target-door contact-friction
counterfactual. It does not perturb saved RGB arrays. Collection steps every
environment and saves real camera/proprioceptive observations. The same
trained RoboCasa checkpoint then predicts actions for all variants, and the
evaluator rebuilds each corresponding environment and executes its action.

```bash
/path/to/robocasa/python robocasa_ood_adapter.py collect \
  --output /path/to/outputs/robocasa_ood_observations.npz \
  --report /path/to/outputs/robocasa_ood_collect.json

python run_ood_policy_smoke.py \
  --observations /path/to/outputs/robocasa_ood_observations.npz \
  --train-batch /path/to/outputs/robocasa_smoke_batch.npz \
  --checkpoint /path/to/outputs/robocasa_model.pt \
  --actions /path/to/outputs/robocasa_ood_actions.npy \
  --report /path/to/outputs/robocasa_ood_policy.json \
  --device cuda

/path/to/robocasa/python robocasa_ood_adapter.py evaluate \
  --observations /path/to/outputs/robocasa_ood_observations.npz \
  --actions /path/to/outputs/robocasa_ood_actions.npy \
  --report /path/to/outputs/robocasa_ood_env.json \
  --metrics-output /path/to/outputs/robocasa_ood_metrics.npz
```

The collection report hashes the changed simulator parameters and measures
the rendered pixel shift. Policy evaluation requires finite actions and
sparse inference for every variant. Environment evaluation requires finite
observations and observable robot-state change, and records reward, task
progress, success and contact count per intervention.

Successful runs end with `DIWA_EXPORTED_POLICY_SMOKE_OK` and the corresponding
`DIWA_<DATASET>_ENV_SMOKE_OK` marker. The policy report additionally requires
a finite total loss, finite non-zero influence-estimator gradients, a sparse
selected ratio strictly between zero and one, and a finite predicted action.
