# Manuscript-to-code map

This package follows the current manuscript, including its running-scale influence targets and **unnormalized sum** of regret-coordinate differences. It does not replace these with fixed-temperature BCE or mean regret distance.

| Manuscript component | Implementation | Important convention |
| --- | --- | --- |
| Structured context / object identity | `object_tokens.py`: `SlotAttention`, `SinkhornTemporalMatcher`, `ObjectCentricTokenizer` | 16 slots; causal temporal alignment; EMA target tokenizer |
| Action proposal and object–time queries | `core.py`: `_predict_proposal`, `_candidate_queries`, `forward` | Three future offsets; 48 unexpanded candidates |
| Influence scores before world decoding | `core.py`: `influence_estimator`, selectors, `forward` | In eval mode only selected queries enter `world_model` |
| Adaptive budget | `core.py`: `_adaptive_budgets`, `_budget_loss` | Ceiling 0.25, floor 0.0625; ceil rounding gives 3–12 valid tokens |
| Dense training / capacity adjustment | `core.py`: `_compress_low_influence_tokens` | Selected tokens full capacity; others bottlenecked, very-low-score tokens detached |
| Latent interventions | `core.py`: `_counterfactual_features`, `_cross_trajectory_partners` | Priority plus random sampling; mask / same-task cross-episode swap |
| Mask training (new implementation choice) | `core.py`: `_mask_token_loss` | Smooth-L1 to a detached valid-token centroid; weight 0.01; historical experiment use is not asserted |
| Action-response sensitivity | `action_model.py`: `decision_moments`; `influence.py` | Shared noise and time; local response moments rather than exact generative moments |
| Influence target and loss (`eq:infloss`) | `influence.py`: `intervention_influence_loss` | EMA response scale, `1-exp(-delta/scale)`, smooth-L1 of sigmoid score |
| Future prediction | `core.py`: `_future_prediction_loss` | EMA slots, cosine error, detached influence softmax weights with temperature 0.25 |
| Critics / progress | `critic.py`: `DecisionCritic`; `core.py`: `_critic_losses` | Twin-Q TD + measured candidate regression, BCE progress, proposed next action |
| Regret (`eq:regret`, `eq:geometry`) | `core.py`: `_regret_losses` | `r=max(q)-q`; L1 **sum**; cosine latent dissimilarity; smooth-L1 |
| Best-rule contrast (`eq:contrastive`) | `core.py`: `_regret_losses` | Same argmax: softplus(distance); different: softplus(margin-distance) |
| Stage schedule | `utils/train_utils.py`; `feature_policy.py`: `schedule` | Proposal/future losses run from step 0; critic/progress start at 1,000; counterfactual influence/budget starts at 1,000; regret/contrastive starts at 2,000; target networks update after optimizer steps |
| Measured branch return | `collect_libero_diwa_supervision.py` | Restore identical simulator state, execute branch, stop at done, use actual reward/progress |
| Ranking / intervention / geometry metrics | `models/diwa/metrics.py`, `evaluate_diwa_metrics.py` | Input arrays must be collected by the evaluator; no benchmark values are embedded |

## Reference versus functional configuration

`configs/diwa_paper.json` records the core architecture, losses and optimization defaults. The standalone feature adapter adds a window stride and freezes the external encoder; its single-device effective batch differs from the 8-device DreamVLA launch. These choices are explicit and do not claim to identify every historical experiment configuration.

`configs/diwa_cpu.json` retains the 48-to-at-most-12 selection problem but uses width 32, one world/fusion layer, a small `DiT-Debug` action head and accelerated schedules. It exists to execute every objective in a local functional test. It must not be used to label paper-scale reproduction results.

Strict feature training and the reference LIBERO script require measured candidate returns. The legacy non-strict core fallback remains available for compatibility, but is disabled by `require_measured_supervision=True` and is not a substitute for measured-return regret training.

Precomputed forward CoTracker displacements are training targets only; they never enter online policy tokens. The offline export runner explicitly disables decision supervision rather than using the legacy critic-return fallback. LIBERO batches pair same-task, different-episode windows on every rank, with actual sampled epoch counts used for scheduling. See [REVIEW_FIXES.md](REVIEW_FIXES.md) for the new mask calibration objective, sampling semantics and their historical-reproduction limits.

Regret distance remains unnormalized; cosine dissimilarity is bounded by 2, so geometry fitting can have irreducible scale mismatch as discussed in the manuscript. No additional label rescaling is introduced here.

## Changes made in this delivery

- Correct TD bootstrapping to use the next proposed action chunk, matching the manuscript equation.
- Reject invalid budget ratios and incomplete / nonfinite strict supervision.
- Share the intervention objective across both policy entry points, with head-mode restoration and consistently shaped return values.
- Respect caller-provided initial noise/device in the flow sampler; preserve classifier-free guidance settings.
- Execute exactly the requested number of flow integration steps and validate diffusion/flow sampling ranges.
- Retain terminal reward, done and progress supervision; mask only unavailable future-lookahead targets.
- Record actual DataLoader batch counts and validate model/data/config tensor contracts before training.
- Make adaptive evaluation and training seeds explicit in launch scripts.
- Add feature data validation, per-rank paired batches, independent training/inference, schema-4 full checkpoints, format-3 feature checkpoints and exact CPU epoch-resume checks.

The upper-level original project and the manuscript's measured-result tables were not modified. Tests and verification reports describe this source delivery, not the provenance of historical robot trials.
