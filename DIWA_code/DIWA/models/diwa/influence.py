"""Paired intervention targets used by both DreamVLA and the feature API.

The running-scale / sigmoid smooth-L1 objective follows Eq. (5) in the
current manuscript. The temperature in DIWACore instead weights future
prediction errors; it is not the scale of intervention targets.
"""

from __future__ import annotations

import torch
import torch.distributed as distributed
from torch import Tensor
import torch.nn.functional as F


def intervention_influence_loss(
    output,
    critic,
    scale: Tensor,
    *,
    action_model=None,
    decode_actions=None,
    num_probes: int = 4,
    covariance_weight: float = 0.1,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    progress_weight: float = 1.0,
    update_scale: bool = True,
    return_responses: bool = False,
):
    """Update a detached target scale; only the scorer receives this gradient.

    ``action_model`` implements decision_moments and exposes its stochastic
    ``net``. Alternatively, ``decode_actions`` maps features to deterministic
    action means. Probe noise/time are reused across every intervened pair.
    """
    logits = output.sampled_influence_logits
    full = output.full_action_features
    counterfactual = output.counterfactual_action_features
    if logits is None or full is None or counterfactual is None:
        return output.action_features.new_zeros(()), {}
    batch_time, samples = logits.shape
    length, hidden = full.shape[-2:]
    full = full.detach()
    counterfactual = counterfactual.detach()
    proposal = output.proposal_actions.reshape(batch_time, length, -1).detach()
    action_dim = proposal.shape[-1]

    with torch.no_grad():
        if action_model is not None:
            training = action_model.net.training
            action_model.net.eval()
            try:
                mean, variance, noise, timestep = action_model.decision_moments(
                    proposal, full, num_probes=num_probes
                )

                def repeat_interventions(value):
                    return (
                        value[:, None]
                        .expand(-1, samples, *value.shape[1:])
                        .reshape(batch_time * samples, *value.shape[1:])
                    )

                cf_mean, cf_variance, _, _ = action_model.decision_moments(
                    repeat_interventions(proposal),
                    counterfactual.reshape(batch_time * samples, length, hidden),
                    num_probes=num_probes,
                    noise=repeat_interventions(noise),
                    timestep=repeat_interventions(timestep),
                )
            finally:
                action_model.net.train(training)
        else:
            if decode_actions is None:
                raise ValueError("provide an action model or deterministic decoder")
            mean = decode_actions(full)
            cf_mean = decode_actions(
                counterfactual.reshape(batch_time * samples, length, hidden)
            )
            variance, cf_variance = torch.zeros_like(mean), torch.zeros_like(cf_mean)

        cf_mean = cf_mean.reshape(batch_time, samples, length, action_dim)
        cf_variance = cf_variance.reshape_as(cf_mean)
        policy = (cf_mean - mean[:, None]).square().mean((-1, -2)).sqrt()
        policy += (
            covariance_weight
            * (cf_variance - variance[:, None]).square().mean((-1, -2)).sqrt()
        )
        state = full.mean(dim=1)
        cf_state = counterfactual.mean(dim=2).reshape(-1, hidden)
        value = torch.zeros_like(policy)
        progress = torch.zeros_like(policy)
        if value_weight:
            q = critic.minimum_q(state, mean)
            cf_q = critic.minimum_q(
                cf_state, cf_mean.reshape(-1, length, action_dim)
            ).reshape(batch_time, samples)
            value = (cf_q - q[:, None]).abs()
        if progress_weight:
            progress = (
                critic.progress(cf_state).reshape(batch_time, samples)
                - critic.progress(state)[:, None]
            ).abs()
        response = (
            policy_weight * policy + value_weight * value + progress_weight * progress
        )
        valid = output.counterfactual_valid_mask
        if valid is None:
            valid = torch.ones_like(logits, dtype=torch.bool)
        # DDP must use the same response scale on every rank. Reduce a sum
        # and a count so padded/uneven batches have the correct global mean.
        local_count = valid.sum().to(response.dtype)
        statistics = torch.stack((response.masked_fill(~valid, 0).sum(), local_count))
        world_size = 1
        if update_scale and distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)
            world_size = distributed.get_world_size()
        if update_scale and statistics[1] > 0:
            observed = (statistics[0] / statistics[1]).clamp_min(1e-6)
            scale.mul_(0.99).add_(observed.to(scale), alpha=0.01)
        target = 1.0 - torch.exp(-response / scale.clamp_min(1e-6))
    if valid.any():
        loss = F.smooth_l1_loss(torch.sigmoid(logits.float())[valid], target.float()[valid])
        loss = loss * (world_size * local_count / statistics[1].clamp_min(1))
    else:
        loss = logits.sum() * 0.0
    diagnostics = {
        "influence_policy": policy[valid].sum() / local_count.clamp_min(1),
        "influence_value": value[valid].sum() / local_count.clamp_min(1),
        "influence_progress": progress[valid].sum() / local_count.clamp_min(1),
    }
    if return_responses:
        diagnostics.update(measured_influence=response, influence_targets=target)
    return loss.to(logits.dtype), diagnostics
