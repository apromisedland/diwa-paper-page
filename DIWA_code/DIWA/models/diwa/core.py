from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as distributed
from torch import Tensor, nn
import torch.nn.functional as F

from .critic import DecisionCritic
from .object_tokens import ObjectCentricTokenizer


@dataclass
class DIWAOutput:
    """Intermediate values produced by the complete DIWA policy core."""

    action_features: Tensor
    full_action_features: Optional[Tensor]
    counterfactual_action_features: Optional[Tensor]
    counterfactual_indices: Optional[Tensor]
    counterfactual_valid_mask: Optional[Tensor]
    counterfactual_swap_mask: Optional[Tensor]
    sampled_influence_logits: Optional[Tensor]
    proposal_actions: Tensor
    influence_logits: Tensor
    influence_probabilities: Tensor
    selected_indices: Tensor
    selected_valid_mask: Tensor
    selected_mask: Tensor
    adaptive_budget_ratio: Tensor
    world_tokens: Tensor
    selected_world_tokens: Tensor
    object_tokens: Optional[Tensor]
    object_assignments: Optional[Tensor]
    aux_losses: Dict[str, Tensor]


class TopKSelector(nn.Module):
    """Select a fixed compute budget before future tokens are expanded."""

    def __init__(self, minimum_tokens: int = 1):
        super().__init__()
        if minimum_tokens < 1:
            raise ValueError("minimum_tokens must be positive")
        self.minimum_tokens = minimum_tokens

    def forward(self, logits: Tensor, budget_ratio: float) -> tuple[Tensor, Tensor]:
        if not 0.0 < budget_ratio <= 1.0:
            raise ValueError(f"budget_ratio must be in (0, 1], got {budget_ratio}")
        num_candidates = logits.shape[-1]
        k = max(self.minimum_tokens, math.ceil(num_candidates * budget_ratio))
        k = min(k, num_candidates)
        indices = torch.topk(logits, k=k, dim=-1, sorted=True).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        return indices, mask


class AdaptiveTopKSelector(nn.Module):
    """Pack a variable number of selected candidates into a padded batch."""

    def __init__(self, minimum_tokens: int = 1):
        super().__init__()
        if minimum_tokens < 1:
            raise ValueError("minimum_tokens must be positive")
        self.minimum_tokens = minimum_tokens

    def forward(
        self, logits: Tensor, budget_ratios: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if budget_ratios.shape != logits.shape[:-1]:
            raise ValueError("adaptive budget ratios must have one value per row")
        if not torch.isfinite(budget_ratios).all() or not (
            (budget_ratios > 0) & (budget_ratios <= 1)
        ).all():
            raise ValueError("adaptive budget ratios must be finite and in (0, 1]")
        num_candidates = logits.shape[-1]
        counts = torch.ceil(budget_ratios.float() * num_candidates).long()
        counts = counts.clamp(min=self.minimum_tokens, max=num_candidates)
        max_count = int(counts.max().item())
        sorted_indices = torch.topk(
            logits, k=max_count, dim=-1, sorted=True
        ).indices
        positions = torch.arange(max_count, device=logits.device)
        selected_valid = positions.unsqueeze(0) < counts.unsqueeze(1)
        selected_mask = torch.zeros_like(logits, dtype=torch.bool)
        selected_mask.scatter_(
            dim=-1,
            index=sorted_indices,
            src=selected_valid,
        )
        return sorted_indices, selected_mask, selected_valid


class ActionConditionedWorldModel(nn.Module):
    """Cross-attention decoder that expands only requested future objects."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        query_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=query_padding_mask,
        )
        if query_padding_mask is not None:
            output = output.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        return output


class PolicyFusion(nn.Module):
    """Fuse current structured state and sparse imagined objects."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        action_pred_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.action_queries = nn.Parameter(
            torch.empty(1, action_pred_steps, hidden_dim)
        )
        nn.init.normal_(self.action_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        context: Tensor,
        world_tokens: Tensor,
        world_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = context.shape[0]
        action_queries = self.action_queries.expand(batch_size, -1, -1)
        memory = torch.cat((context, world_tokens), dim=1)
        memory_padding_mask = None
        if world_padding_mask is not None:
            context_valid = torch.zeros(
                batch_size,
                context.shape[1],
                dtype=torch.bool,
                device=context.device,
            )
            memory_padding_mask = torch.cat(
                (context_valid, world_padding_mask), dim=1
            )
        return self.decoder(
            tgt=action_queries,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )


class DIWACore(nn.Module):
    """Object-centric, regret-aware and compute-adaptive DIWA policy core."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        action_pred_steps: int,
        action_dim: int = 7,
        continuous_action_dim: int = 6,
        horizon: int = 3,
        num_slots: int = 16,
        world_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.0,
        influence_temperature: float = 0.25,
        counterfactual_samples: int = 4,
        entropy_weight: float = 0.01,
        sam_feature_dim: int = 256,
        slot_iterations: int = 3,
        adaptive_budget: bool = False,
        minimum_budget_ratio: float = 0.0625,
        budget_threshold: float = 0.5,
        critic_discount: float = 0.99,
        regret_candidates: int = 6,
        contrastive_margin: float = 0.5,
        require_measured_supervision: bool = False,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        if action_pred_steps < 1:
            raise ValueError("DIWA requires at least one action prediction step")
        if action_dim < 2:
            raise ValueError("DIWA requires at least two action dimensions")
        if not 0 < continuous_action_dim < action_dim:
            raise ValueError(
                "continuous_action_dim must be in [1, action_dim - 1]"
            )
        if not 0.0 < minimum_budget_ratio <= 1.0:
            raise ValueError("minimum_budget_ratio must be in (0, 1]")
        if world_layers < 1 or fusion_layers < 1:
            raise ValueError("world_layers and fusion_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(influence_temperature) or influence_temperature <= 0:
            raise ValueError("influence_temperature must be finite and positive")
        if counterfactual_samples < 0:
            raise ValueError("counterfactual_samples must be non-negative")
        if not math.isfinite(entropy_weight) or entropy_weight < 0:
            raise ValueError("entropy_weight must be finite and non-negative")
        if sam_feature_dim < 1 or slot_iterations < 1:
            raise ValueError("sam_feature_dim and slot_iterations must be positive")
        if not 0.0 < budget_threshold < 1.0:
            raise ValueError("budget_threshold must be in (0, 1)")
        if not 0.0 <= critic_discount <= 1.0:
            raise ValueError("critic_discount must be in [0, 1]")
        if regret_candidates < 4:
            raise ValueError("regret_candidates must be at least four")
        if not math.isfinite(contrastive_margin) or contrastive_margin < 0:
            raise ValueError("contrastive_margin must be finite and non-negative")

        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_slots = num_slots
        self.num_candidates = horizon * num_slots
        self.action_pred_steps = action_pred_steps
        self.action_dim = action_dim
        self.continuous_action_dim = continuous_action_dim
        self.influence_temperature = influence_temperature
        self.counterfactual_samples = counterfactual_samples
        self.require_measured_supervision = require_measured_supervision
        self.enable_influence_estimator = True
        # Ablation entry points may disable cross-trajectory swaps while
        # retaining mask interventions and the rest of the influence target.
        self.enable_counterfactual_swaps = True
        self.entropy_weight = entropy_weight
        self.adaptive_budget = adaptive_budget
        self.minimum_budget_ratio = minimum_budget_ratio
        self.budget_threshold = budget_threshold
        self.regret_candidates = regret_candidates
        self.contrastive_margin = contrastive_margin

        self.object_tokenizer = ObjectCentricTokenizer(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            sam_feature_dim=sam_feature_dim,
            slot_iterations=slot_iterations,
        )
        self.target_object_tokenizer = deepcopy(
            self.object_tokenizer
        ).requires_grad_(False)
        self.horizon_embedding = nn.Parameter(torch.empty(horizon, hidden_dim))
        self.slot_embedding = nn.Parameter(torch.empty(num_slots, hidden_dim))
        self.counterfactual_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.counterfactual_token, std=0.02)

        self.context_norm = nn.LayerNorm(hidden_dim)
        self.proposal_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_pred_steps * action_dim),
        )
        self.action_condition_projector = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.influence_estimator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.budget_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.budget_head[-1].bias, -4.0)
        compressed_dim = max(8, hidden_dim // 4)
        self.low_capacity_transition = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, compressed_dim),
            nn.GELU(),
            nn.Linear(compressed_dim, hidden_dim),
        )
        self.selector = TopKSelector()
        self.adaptive_selector = AdaptiveTopKSelector()
        self.world_model = ActionConditionedWorldModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=world_layers,
            dropout=dropout,
        )
        self.policy_fusion = PolicyFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=fusion_layers,
            action_pred_steps=action_pred_steps,
            dropout=dropout,
        )
        self.critic = DecisionCritic(
            hidden_dim=hidden_dim,
            action_pred_steps=action_pred_steps,
            action_dim=action_dim,
            discount=critic_discount,
        )
        offsets = torch.linspace(-0.15, 0.15, regret_candidates - 4)
        self.register_buffer("regret_offsets", offsets, persistent=True)

    def tokenize_objects(
        self,
        visual_tokens: Tensor,
        **object_inputs: Optional[Tensor],
    ):
        return self.object_tokenizer(visual_tokens, **object_inputs)

    @torch.no_grad()
    def tokenize_target_objects(
        self,
        visual_tokens: Tensor,
        **object_inputs: Optional[Tensor],
    ):
        was_training = self.target_object_tokenizer.training
        self.target_object_tokenizer.eval()
        output = self.target_object_tokenizer(
            visual_tokens.detach(), **object_inputs
        )
        self.target_object_tokenizer.train(was_training)
        return output

    def _candidate_queries(
        self,
        batch_size: int,
        dtype: torch.dtype,
        current_object_tokens: Optional[Tensor],
    ) -> Tensor:
        if current_object_tokens is None:
            objects = self.slot_embedding.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            if current_object_tokens.shape != (
                batch_size,
                self.num_slots,
                self.hidden_dim,
            ):
                raise ValueError("current object token shape does not match DIWA slots")
            objects = current_object_tokens
        candidates = (
            objects[:, None, :, :]
            + self.horizon_embedding[None, :, None, :]
            + self.slot_embedding[None, None, :, :]
        )
        return candidates.reshape(
            batch_size, self.num_candidates, self.hidden_dim
        ).to(dtype=dtype)

    @staticmethod
    def _gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return torch.gather(tokens, dim=1, index=gather_indices)

    def _predict_proposal(self, context_summary: Tensor) -> Tensor:
        raw = self.proposal_head(context_summary)
        raw = raw.view(
            context_summary.shape[0], self.action_pred_steps, self.action_dim
        )
        continuous = torch.tanh(raw[..., : self.continuous_action_dim])
        discrete = torch.sigmoid(raw[..., self.continuous_action_dim :])
        return torch.cat((continuous, discrete), dim=-1)

    def _proposal_loss(
        self, proposal: Tensor, action_labels: Optional[Tensor], valid_mask: Optional[Tensor] = None
    ) -> Tensor:
        if action_labels is None:
            return proposal.new_zeros(())
        if valid_mask is not None:
            valid = valid_mask.reshape(-1).bool()
            if not valid.any():
                return proposal.sum() * 0.0
            proposal, action_labels = proposal[valid], action_labels[valid]
        continuous_loss = F.smooth_l1_loss(
            proposal[..., : self.continuous_action_dim],
            action_labels[..., : self.continuous_action_dim],
        )
        discrete_loss = F.binary_cross_entropy(
            proposal[..., self.continuous_action_dim :].clamp(1e-6, 1.0 - 1e-6),
            action_labels[..., self.continuous_action_dim :],
        )
        return continuous_loss + 0.01 * discrete_loss

    def _future_prediction_loss(
        self,
        world_tokens: Tensor,
        influence_logits: Tensor,
        future_targets: Optional[Tensor],
        future_valid_mask: Optional[Tensor],
    ) -> Tensor:
        if future_targets is None or future_valid_mask is None:
            return world_tokens.new_zeros(())
        batch_time = world_tokens.shape[0]
        targets = future_targets.reshape(
            batch_time, self.num_candidates, self.hidden_dim
        ).detach()
        valid = (
            future_valid_mask.unsqueeze(-1)
            .expand(-1, -1, -1, self.num_slots)
            .reshape(batch_time, self.num_candidates)
        )
        has_target = valid.any(dim=-1)
        if not has_target.any():
            return world_tokens.sum() * 0.0
        # Normalize over available targets before softmax. Normalizing over
        # missing targets first can underflow every valid weight to zero.
        active = valid[has_target]
        logits = influence_logits.detach().float()[has_target] / self.influence_temperature
        weights = F.softmax(logits.masked_fill(~active, -torch.inf), dim=-1)
        prediction = world_tokens[has_target].float().masked_fill(~active[..., None], 0.0)
        reference = targets[has_target].float().masked_fill(~active[..., None], 0.0)
        token_error = 1.0 - F.cosine_similarity(prediction, reference, dim=-1)
        return (weights * token_error).sum(dim=-1).mean().to(
            world_tokens.dtype
        )

    def _mask_token_loss(
        self, world_tokens: Tensor, valid_mask: Optional[Tensor]
    ) -> Tensor:
        """Learn a neutral replacement from detached reference-token moments.

        Influence responses stay stop-gradient. This separate calibration
        objective updates only the mask, avoiding scorer/mask collusion.
        The centroid is shared across ranks and excludes padded states.
        """
        if not self.training:
            return world_tokens.new_zeros(())
        with torch.no_grad():
            valid = (
                torch.ones(world_tokens.shape[0], device=world_tokens.device, dtype=torch.bool)
                if valid_mask is None else valid_mask.reshape(-1).bool()
            )
            reference = world_tokens.detach().float()
            total = reference.masked_fill(~valid[:, None, None], 0.0).sum((0, 1))
            count = valid.sum().to(total) * reference.shape[1]
            statistics = torch.cat((total, count.reshape(1)))
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)
            target = statistics[:-1] / statistics[-1].clamp_min(1)
        if statistics[-1] == 0:
            return self.counterfactual_token.sum() * 0.0
        return F.smooth_l1_loss(
            self.counterfactual_token.reshape(-1).float(), target
        ).to(world_tokens.dtype)

    def _adaptive_budgets(
        self, context_summary: Tensor, maximum_budget_ratio: float
    ) -> Tensor:
        if not 0.0 < maximum_budget_ratio <= 1.0:
            raise ValueError("maximum budget ratio must be in (0, 1]")
        if not self.adaptive_budget:
            return context_summary.new_full(
                (context_summary.shape[0],), maximum_budget_ratio
            )
        reduction = torch.sigmoid(
            self.budget_head(context_summary).squeeze(-1)
        )
        if maximum_budget_ratio < self.minimum_budget_ratio:
            raise ValueError("maximum budget ratio is below the configured minimum")
        maximum = maximum_budget_ratio
        return maximum - reduction * (
            maximum - self.minimum_budget_ratio
        )

    def _budget_loss(
        self,
        logits: Tensor,
        adaptive_ratios: Tensor,
        maximum_budget_ratio: float,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if valid_mask is not None:
            valid = valid_mask.reshape(-1).bool()
            if not valid.any():
                return logits.sum() * 0.0
            logits, adaptive_ratios = logits[valid], adaptive_ratios[valid]
        effective_maximum = max(
            maximum_budget_ratio, self.minimum_budget_ratio
        )
        probabilities = torch.sigmoid(logits.float())
        observed_ratio = probabilities.mean(dim=-1)
        # The detached target is the number of counterfactually influential
        # variables implied by calibrated gates, bounded by the compute ceiling.
        target_ratio = (
            (probabilities.detach() >= self.budget_threshold)
            .float()
            .mean(dim=-1)
            .clamp(min=self.minimum_budget_ratio, max=effective_maximum)
        )
        calibration_error = (
            observed_ratio - adaptive_ratios.float()
        ).abs().mean()
        allocation_error = (
            adaptive_ratios.float() - target_ratio
        ).abs().mean()
        ceiling_error = F.relu(
            adaptive_ratios.float() - effective_maximum
        ).mean()
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-6))
            + (1.0 - probabilities)
            * torch.log((1.0 - probabilities).clamp_min(1e-6))
        ).mean()
        return (
            calibration_error
            + allocation_error
            + ceiling_error
            + self.entropy_weight * entropy
        ).to(logits.dtype)

    def _compress_low_influence_tokens(
        self,
        world_tokens: Tensor,
        influence_probabilities: Tensor,
        selected_mask: Tensor,
    ) -> Tensor:
        """Apply the paper's full/compressed/detached capacity ladder."""
        # Unselected queries use a lower-capacity transition before policy
        # fusion; very low-scoring queries are detached from that pathway.
        compressed = self.low_capacity_transition(world_tokens)
        high = selected_mask.unsqueeze(-1)
        medium = (
            influence_probabilities
            >= self.budget_threshold * 0.5
        ).unsqueeze(-1)
        abstracted = torch.where(high, world_tokens, compressed)
        # Very low-influence variables remain available as a background
        # approximation but cannot send policy gradients into the world model.
        return torch.where(
            high | medium,
            abstracted,
            abstracted.detach(),
        )

    @staticmethod
    def _expand_metadata(
        values: Optional[Tensor], sequence_length: int, device: torch.device
    ) -> Optional[Tensor]:
        if values is None:
            return None
        values = values.to(device=device)
        if values.ndim != 1:
            raise ValueError("task and episode identifiers must be one-dimensional")
        return values[:, None].expand(-1, sequence_length).reshape(-1)

    def _cross_trajectory_partners(
        self,
        context_summary: Tensor,
        actions: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        task_ids: Optional[Tensor],
        episode_ids: Optional[Tensor],
        valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_time = context_summary.shape[0]
        if task_ids is None or episode_ids is None or batch_size < 2:
            return (
                torch.zeros(batch_time, dtype=torch.long, device=context_summary.device),
                torch.zeros(batch_time, dtype=torch.bool, device=context_summary.device),
            )
        flat_tasks = self._expand_metadata(
            task_ids, sequence_length, context_summary.device
        )
        flat_episodes = self._expand_metadata(
            episode_ids, sequence_length, context_summary.device
        )
        eligible = flat_tasks[:, None].eq(flat_tasks[None, :])
        eligible &= flat_episodes[:, None].ne(flat_episodes[None, :])
        if valid_mask is not None:
            valid = valid_mask.reshape(-1).bool()
            eligible &= valid[:, None] & valid[None, :]

        normalized = F.normalize(context_summary.detach().float(), dim=-1)
        scores = normalized @ normalized.transpose(0, 1)
        if actions is not None:
            flat_actions = actions.detach().float().flatten(1)
            action_distance = torch.cdist(flat_actions, flat_actions, p=1)
            action_distance = action_distance / action_distance.amax().clamp_min(1e-6)
            # Prefer visually similar states whose demonstrated decisions differ.
            scores = scores + 0.25 * action_distance
        scores = scores.masked_fill(~eligible, -torch.inf)
        valid = eligible.any(dim=-1)
        safe_scores = torch.where(
            valid.unsqueeze(-1), scores, torch.zeros_like(scores)
        )
        partners = safe_scores.argmax(dim=-1)
        return partners, valid

    def _counterfactual_features(
        self,
        context: Tensor,
        context_summary: Tensor,
        world_tokens: Tensor,
        influence_logits: Tensor,
        action_labels: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
        task_ids: Optional[Tensor],
        episode_ids: Optional[Tensor],
        valid_mask: Optional[Tensor] = None,
        exhaustive: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_time, num_candidates, hidden_dim = world_tokens.shape
        num_samples = num_candidates if exhaustive else min(self.counterfactual_samples, num_candidates)
        num_priority = max(1, num_samples // 2)
        priority_indices = torch.topk(
            influence_logits.detach(), k=num_priority, dim=-1
        ).indices
        num_random = num_samples - num_priority
        if num_random:
            random_scores = torch.rand_like(influence_logits)
            random_scores.scatter_(1, priority_indices, -1.0)
            random_indices = random_scores.topk(num_random, dim=-1).indices
            sampled_indices = torch.cat((priority_indices, random_indices), dim=-1)
        else:
            sampled_indices = priority_indices
        if exhaustive:
            sampled_indices = torch.arange(num_candidates, device=world_tokens.device)[None].expand(batch_time, -1)

        partners, partner_valid = self._cross_trajectory_partners(
            context_summary,
            action_labels,
            batch_size,
            sequence_length,
            task_ids,
            episode_ids,
            valid_mask,
        )
        partner_world = world_tokens.detach()[partners]
        swap_replacements = self._gather_tokens(partner_world, sampled_indices)
        mask_replacements = self.counterfactual_token.to(
            dtype=world_tokens.dtype
        ).expand(batch_time, num_samples, -1)
        use_swap = (
            torch.arange(num_samples, device=world_tokens.device)[None, :] % 2 == 1
        ) & partner_valid[:, None]
        if not self.enable_counterfactual_swaps:
            use_swap = torch.zeros_like(use_swap)
        replacements = torch.where(
            use_swap.unsqueeze(-1), swap_replacements, mask_replacements
        )

        counterfactual_world = (
            world_tokens.unsqueeze(1).expand(-1, num_samples, -1, -1).clone()
        )
        flat_counterfactual = counterfactual_world.reshape(
            batch_time * num_samples, num_candidates, hidden_dim
        )
        flat_rows = torch.arange(batch_time * num_samples, device=world_tokens.device)
        flat_counterfactual[flat_rows, sampled_indices.reshape(-1)] = (
            replacements.reshape(batch_time * num_samples, hidden_dim)
        )
        repeated_context = context.unsqueeze(1).expand(-1, num_samples, -1, -1)
        counterfactual_action_features = self.policy_fusion(
            repeated_context.reshape(
                batch_time * num_samples, context.shape[1], hidden_dim
            ),
            flat_counterfactual,
        ).view(
            batch_time,
            num_samples,
            self.action_pred_steps,
            hidden_dim,
        )
        sampled_logits = torch.gather(influence_logits, dim=1, index=sampled_indices)
        valid = torch.ones_like(sampled_logits, dtype=torch.bool)
        if valid_mask is not None:
            valid &= valid_mask.reshape(-1, 1).bool()
        return (
            counterfactual_action_features,
            sampled_indices,
            sampled_logits,
            valid,
            use_swap,
        )

    def _critic_losses(
        self,
        decision_features: Optional[Tensor],
        action_labels: Optional[Tensor],
        proposal_actions: Tensor,
        rewards: Optional[Tensor],
        dones: Optional[Tensor],
        progress_targets: Optional[Tensor],
        valid_mask: Optional[Tensor],
        candidate_actions: Optional[Tensor],
        candidate_q_values: Optional[Tensor],
        batch_size: int,
        sequence_length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        zero = proposal_actions.new_zeros(())
        required = (
            decision_features,
            action_labels,
            rewards,
            dones,
            progress_targets,
            valid_mask,
        )
        if any(item is None for item in required):
            return zero, zero, zero
        states = decision_features.mean(dim=1).view(
            batch_size, sequence_length, self.hidden_dim
        )
        actions = action_labels.view(
            batch_size, sequence_length, self.action_pred_steps, self.action_dim
        )
        proposals = proposal_actions.view_as(actions)
        next_states = torch.cat((states[:, 1:], states[:, -1:]), dim=1)
        # The TD target in the paper uses the next proposal, not the next
        # demonstrated action. The repeated final entry is masked below.
        next_actions = torch.cat((proposals[:, 1:], proposals[:, -1:]), dim=1)
        next_valid = torch.cat(
            (valid_mask[:, 1:], torch.zeros_like(valid_mask[:, -1:])), dim=1
        )
        loss_output = self.critic.losses(
            state=states.reshape(-1, self.hidden_dim),
            actions=actions.reshape(
                -1, self.action_pred_steps, self.action_dim
            ),
            rewards=rewards.reshape(-1),
            dones=dones.reshape(-1),
            progress_targets=progress_targets.reshape(-1),
            valid_mask=valid_mask.reshape(-1).bool(),
            next_state=next_states.reshape(-1, self.hidden_dim),
            next_actions=next_actions.reshape(
                -1, self.action_pred_steps, self.action_dim
            ),
            next_valid_mask=next_valid.reshape(-1).bool(),
        )
        candidate_q_loss = zero
        if (candidate_actions is None) != (candidate_q_values is None):
            raise ValueError(
                "candidate_actions and candidate_q_values must be provided together"
            )
        if candidate_actions is not None:
            expected_prefix = (batch_size, sequence_length)
            if (
                candidate_actions.ndim != 5
                or candidate_actions.shape[:2] != expected_prefix
                or candidate_actions.shape[-2:]
                != (self.action_pred_steps, self.action_dim)
            ):
                raise ValueError(
                    "candidate_actions must have shape "
                    "[batch, time, candidates, action_steps, action_dim]"
                )
            if (
                candidate_q_values.ndim != 3
                or candidate_q_values.shape[:2] != expected_prefix
                or candidate_q_values.shape[2]
                != candidate_actions.shape[2]
            ):
                raise ValueError(
                    "candidate_q_values must have shape "
                    "[batch, time, candidates]"
                )
            num_candidates = candidate_actions.shape[2]
            repeated_states = states.unsqueeze(2).expand(
                -1, -1, num_candidates, -1
            )
            predicted_q1, predicted_q2 = self.critic.q_values(
                repeated_states.reshape(-1, self.hidden_dim),
                candidate_actions.reshape(
                    -1, self.action_pred_steps, self.action_dim
                ),
            )
            measured_q = candidate_q_values.reshape(-1).float()
            candidate_valid = (
                valid_mask.unsqueeze(-1)
                .expand(-1, -1, num_candidates)
                .reshape(-1)
                .bool()
                & torch.isfinite(measured_q)
            )
            if candidate_valid.any():
                candidate_q_loss = (
                    F.smooth_l1_loss(
                        predicted_q1[candidate_valid].float(),
                        measured_q[candidate_valid],
                    )
                    + F.smooth_l1_loss(
                        predicted_q2[candidate_valid].float(),
                        measured_q[candidate_valid],
                    )
                ).to(states.dtype)
        return (
            loss_output.critic_loss + candidate_q_loss,
            loss_output.progress_loss,
            candidate_q_loss.detach(),
        )

    def _regret_losses(
        self,
        latent_tokens: Optional[Tensor],
        action_labels: Optional[Tensor],
        proposal_actions: Tensor,
        context_summary: Tensor,
        batch_size: int,
        sequence_length: int,
        task_ids: Optional[Tensor],
        episode_ids: Optional[Tensor],
        candidate_q_values: Optional[Tensor],
        valid_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        zero = proposal_actions.new_zeros(())
        if latent_tokens is None or action_labels is None:
            return zero, zero
        states = latent_tokens.mean(dim=1)
        pair_valid = valid_mask
        if candidate_q_values is not None:
            finite = torch.isfinite(candidate_q_values).all(dim=-1)
            pair_valid = finite if pair_valid is None else pair_valid.bool() & finite
        partners, partner_valid = self._cross_trajectory_partners(
            context_summary,
            action_labels,
            batch_size,
            sequence_length,
            task_ids,
            episode_ids,
            pair_valid,
        )
        if valid_mask is not None:
            flat_valid = valid_mask.reshape(-1).bool()
            partner_valid = (
                partner_valid
                & flat_valid
                & flat_valid[partners]
            )
        if not partner_valid.any():
            return zero, zero

        if candidate_q_values is not None:
            if (
                candidate_q_values.ndim != 3
                or candidate_q_values.shape[:2]
                != (batch_size, sequence_length)
            ):
                raise ValueError(
                    "candidate_q_values must have shape "
                    "[batch, time, candidates]"
                )
            q_current = candidate_q_values.reshape(
                batch_size * sequence_length, -1
            ).detach()
            q_partner = q_current[partners]
            finite = torch.isfinite(q_current).all(dim=-1)
            partner_valid = (
                partner_valid & finite & finite[partners]
            )
            if not partner_valid.any():
                return zero, zero
        else:
            demonstrated = action_labels.detach()
            proposed = proposal_actions.detach()
            partner_demonstrated = demonstrated[partners]
            partner_proposed = proposed[partners]
            candidates = [
                demonstrated,
                partner_demonstrated,
                proposed,
                partner_proposed,
            ]
            for offset in self.regret_offsets:
                perturbed = proposed.clone()
                perturbed[..., : self.continuous_action_dim] = (
                    perturbed[..., : self.continuous_action_dim] + offset
                ).clamp(-1.0, 1.0)
                candidates.append(perturbed)
            generated_actions = torch.stack(candidates, dim=1)
            generated_actions = generated_actions[:, : self.regret_candidates]
            num_candidates = generated_actions.shape[1]

            def evaluate(candidate_states: Tensor) -> Tensor:
                repeated_states = candidate_states[:, None, :].expand(
                    -1, num_candidates, -1
                )
                with torch.no_grad():
                    return self.critic.minimum_q(
                        repeated_states.reshape(-1, self.hidden_dim),
                        generated_actions.reshape(
                            -1, self.action_pred_steps, self.action_dim
                        ),
                        target=True,
                    ).view(-1, num_candidates)

            q_current = evaluate(states.detach())
            q_partner = evaluate(states.detach()[partners])
        regret_current = q_current.amax(dim=-1, keepdim=True) - q_current
        regret_partner = q_partner.amax(dim=-1, keepdim=True) - q_partner
        regret_distance = (
            regret_current - regret_partner
        ).abs().sum(dim=-1).detach()
        latent_distance = 1.0 - (
            F.normalize(states.float(), dim=-1)
            * F.normalize(states[partners].float(), dim=-1)
        ).sum(dim=-1)
        geometry = F.smooth_l1_loss(
            latent_distance[partner_valid],
            regret_distance[partner_valid].float(),
        ).to(states.dtype)

        same_optimal = q_current.argmax(dim=-1).eq(q_partner.argmax(dim=-1))
        positive = same_optimal & partner_valid
        negative = (~same_optimal) & partner_valid
        contrastive_terms = latent_distance.new_zeros(latent_distance.shape)
        # -log(sigmoid(-d)) for decision-equivalent pairs and
        # -log(sigmoid(d-m)) for decision-different pairs.
        contrastive_terms[positive] = F.softplus(latent_distance[positive])
        contrastive_terms[negative] = F.softplus(
            self.contrastive_margin - latent_distance[negative]
        )
        contrastive = contrastive_terms[partner_valid].mean().to(states.dtype)
        return geometry, contrastive

    def forward(
        self,
        context_tokens: Tensor,
        *,
        current_object_tokens: Optional[Tensor] = None,
        object_tokens: Optional[Tensor] = None,
        object_assignments: Optional[Tensor] = None,
        action_labels: Optional[Tensor] = None,
        future_targets: Optional[Tensor] = None,
        future_valid_mask: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
        dones: Optional[Tensor] = None,
        progress_targets: Optional[Tensor] = None,
        supervision_valid_mask: Optional[Tensor] = None,
        candidate_actions: Optional[Tensor] = None,
        candidate_q_values: Optional[Tensor] = None,
        task_ids: Optional[Tensor] = None,
        episode_ids: Optional[Tensor] = None,
        budget_ratio: float = 1.0,
        force_dense_budget: bool = False,
        teacher_forcing_ratio: float = 0.0,
        compute_counterfactual: bool = True,
        intervention_reference: bool = False,
        decision_supervision_enabled: bool = True,
    ) -> DIWAOutput:
        if context_tokens.ndim != 4:
            raise ValueError(
                "context_tokens must have shape [batch, time, tokens, hidden]"
            )
        batch_size, sequence_length, context_length, hidden_dim = context_tokens.shape
        if min(batch_size, sequence_length, context_length) < 1:
            raise ValueError("context batch, time and token axes must be nonempty")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")
        if not math.isfinite(float(budget_ratio)) or not 0.0 < budget_ratio <= 1.0:
            raise ValueError("budget_ratio must be finite and in (0, 1]")
        if (
            not math.isfinite(float(teacher_forcing_ratio))
            or not 0.0 <= teacher_forcing_ratio <= 1.0
        ):
            raise ValueError("teacher_forcing_ratio must be finite and in [0, 1]")

        prefix = (batch_size, sequence_length)
        if current_object_tokens is not None and current_object_tokens.shape != (
            *prefix,
            self.num_slots,
            self.hidden_dim,
        ):
            raise ValueError(
                "current_object_tokens must have shape "
                f"[batch, time, {self.num_slots}, {self.hidden_dim}]"
            )
        if action_labels is not None and action_labels.shape != (
            *prefix,
            self.action_pred_steps,
            self.action_dim,
        ):
            raise ValueError(
                "action_labels must have shape "
                f"[batch, time, {self.action_pred_steps}, {self.action_dim}]"
            )
        if (future_targets is None) != (future_valid_mask is None):
            raise ValueError(
                "future_targets and future_valid_mask must be provided together"
            )
        if future_targets is not None:
            expected_targets = (
                *prefix,
                self.horizon,
                self.num_slots,
                self.hidden_dim,
            )
            if future_targets.shape != expected_targets:
                raise ValueError(
                    f"future_targets must have shape {expected_targets}"
                )
            if future_valid_mask.shape != (*prefix, self.horizon):
                raise ValueError(
                    "future_valid_mask must have shape "
                    f"[batch, time, {self.horizon}]"
                )
        if (task_ids is None) != (episode_ids is None):
            raise ValueError("task_ids and episode_ids must be provided together")
        for name, value in (("task_ids", task_ids), ("episode_ids", episode_ids)):
            if value is not None and value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [{batch_size}]")
        for name, value in (
            ("rewards", rewards),
            ("dones", dones),
            ("progress_targets", progress_targets),
            ("supervision_valid_mask", supervision_valid_mask),
        ):
            if value is not None and value.shape != prefix:
                raise ValueError(f"{name} must have shape {prefix}")
        if (candidate_actions is None) != (candidate_q_values is None):
            raise ValueError(
                "candidate_actions and candidate_q_values must be provided together"
            )
        if candidate_actions is not None:
            if (
                candidate_actions.ndim != 5
                or candidate_actions.shape[:2] != prefix
                or candidate_actions.shape[-2:]
                != (self.action_pred_steps, self.action_dim)
            ):
                raise ValueError(
                    "candidate_actions must have shape "
                    "[batch, time, candidates, action_steps, action_dim]"
                )
            if candidate_q_values.shape != candidate_actions.shape[:3]:
                raise ValueError(
                    "candidate_q_values must have shape [batch, time, candidates]"
                )
        if self.training and self.require_measured_supervision:
            if not decision_supervision_enabled:
                raise ValueError("strict DIWA cannot disable measured decision supervision")
            required = {
                "action_labels": action_labels,
                "rewards": rewards,
                "dones": dones,
                "progress_targets": progress_targets,
                "supervision_valid_mask": supervision_valid_mask,
                "candidate_actions": candidate_actions,
                "candidate_q_values": candidate_q_values,
                "task_ids": task_ids,
                "episode_ids": episode_ids,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"strict DIWA supervision is missing {missing}")
            shapes = {
                "action_labels": (*prefix, self.action_pred_steps, self.action_dim),
                "candidate_actions": (
                    *prefix, self.regret_candidates, self.action_pred_steps, self.action_dim
                ),
                "candidate_q_values": (*prefix, self.regret_candidates),
                **{name: prefix for name in (
                    "rewards", "dones", "progress_targets", "supervision_valid_mask"
                )},
            }
            for name, shape in shapes.items():
                value = required[name]
                if value.shape != shape or not torch.isfinite(value).all():
                    raise ValueError(f"{name} must be finite with shape {shape}")
            if not ((progress_targets >= 0) & (progress_targets <= 1)).all():
                raise ValueError("progress_targets must be in [0, 1]")
            if not ((dones == 0) | (dones == 1)).all():
                raise ValueError("dones must be binary")
            if not ((supervision_valid_mask == 0) | (supervision_valid_mask == 1)).all():
                raise ValueError("supervision_valid_mask must be binary")
            tolerance = 1e-6
            for name, value in (
                ("action_labels", action_labels),
                ("candidate_actions", candidate_actions),
            ):
                continuous = value[..., : self.continuous_action_dim]
                discrete = value[..., self.continuous_action_dim :]
                if (continuous.abs() > 1.0 + tolerance).any() or (
                    (discrete < 0.0) | (discrete > 1.0)
                ).any():
                    raise ValueError(
                        f"{name} must use continuous [-1, 1] and discrete [0, 1]"
                    )
            integer_dtypes = {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
            for name in ("task_ids", "episode_ids"):
                value = required[name]
                if value.dtype not in integer_dtypes:
                    raise ValueError(f"{name} must contain integer identifiers")
            valid_episodes = supervision_valid_mask.bool().any(dim=1)
            eligible = task_ids[:, None].eq(task_ids[None, :])
            eligible &= episode_ids[:, None].ne(episode_ids[None, :])
            eligible &= valid_episodes[:, None] & valid_episodes[None, :]
            if (valid_episodes & ~eligible.any(dim=1)).any():
                raise ValueError(
                    "strict DIWA needs a valid same-task, different-episode "
                    "partner for every supervised episode in the batch"
                )
        batch_time = batch_size * sequence_length
        context = context_tokens.reshape(batch_time, context_length, hidden_dim)
        context_summary = self.context_norm(context.mean(dim=1))

        flat_objects = None
        if current_object_tokens is not None:
            flat_objects = current_object_tokens.reshape(
                batch_time, self.num_slots, hidden_dim
            )
        flat_action_labels = None
        if action_labels is not None:
            flat_action_labels = action_labels.reshape(
                batch_time, self.action_pred_steps, self.action_dim
            )
        proposal_actions = self._predict_proposal(context_summary)
        condition_actions = proposal_actions
        if (
            self.training
            and flat_action_labels is not None
            and teacher_forcing_ratio > 0
        ):
            use_teacher = (
                torch.rand(batch_time, 1, 1, device=context_tokens.device)
                < teacher_forcing_ratio
            )
            condition_actions = torch.where(
                use_teacher,
                flat_action_labels.detach().to(dtype=proposal_actions.dtype),
                proposal_actions,
            )
        proposal_condition = self.action_condition_projector(condition_actions).mean(
            dim=1
        )

        candidates = self._candidate_queries(
            batch_time, context.dtype, flat_objects
        )
        conditioned_candidates = (
            candidates
            + context_summary.unsqueeze(1)
            + proposal_condition.unsqueeze(1)
        )
        estimated_influence_logits = self.influence_estimator(
            conditioned_candidates
        ).squeeze(-1)
        influence_logits = (
            estimated_influence_logits
            if self.enable_influence_estimator
            else torch.zeros_like(estimated_influence_logits)
        )
        influence_probabilities = torch.sigmoid(influence_logits)
        selection_budget_ratio = 1.0 if force_dense_budget else budget_ratio
        adaptive_ratios = (
            context_summary.new_ones(batch_time)
            if force_dense_budget
            else self._adaptive_budgets(
                context_summary, selection_budget_ratio
            )
        )
        if self.adaptive_budget:
            selected_indices, selected_mask, selected_valid = (
                self.adaptive_selector(influence_logits, adaptive_ratios)
            )
        else:
            selected_indices, selected_mask = self.selector(
                influence_logits, selection_budget_ratio
            )
            selected_valid = torch.ones_like(selected_indices, dtype=torch.bool)

        memory = torch.cat((context, proposal_condition.unsqueeze(1)), dim=1)
        if self.training or intervention_reference:
            world_tokens = self.world_model(conditioned_candidates, memory)
            policy_world_tokens = self._compress_low_influence_tokens(
                world_tokens,
                influence_probabilities,
                selected_mask,
            )
            selected_world_tokens = self._gather_tokens(
                policy_world_tokens, selected_indices
            ).masked_fill(~selected_valid.unsqueeze(-1), 0.0)
        else:
            selected_queries = self._gather_tokens(
                conditioned_candidates, selected_indices
            )
            selected_world_tokens = self.world_model(
                selected_queries,
                memory,
                query_padding_mask=~selected_valid,
            )
            world_tokens = selected_world_tokens
            policy_world_tokens = world_tokens
        action_features = self.policy_fusion(
            context,
            selected_world_tokens,
            world_padding_mask=~selected_valid,
        )

        full_action_features = None
        counterfactual_action_features = None
        counterfactual_indices = None
        counterfactual_valid_mask = None
        counterfactual_swap_mask = None
        sampled_influence_logits = None
        if self.training or intervention_reference:
            full_action_features = self.policy_fusion(
                context, policy_world_tokens
            )
            if compute_counterfactual and (self.counterfactual_samples > 0 or intervention_reference):
                (
                    counterfactual_action_features,
                    counterfactual_indices,
                    sampled_influence_logits,
                    counterfactual_valid_mask,
                    counterfactual_swap_mask,
                ) = self._counterfactual_features(
                    context,
                    context_summary,
                    policy_world_tokens,
                    influence_logits,
                    flat_action_labels,
                    batch_size,
                    sequence_length,
                    task_ids,
                    episode_ids,
                    supervision_valid_mask,
                    exhaustive=intervention_reference,
                )

        if decision_supervision_enabled:
            critic_loss, progress_loss, candidate_q_diagnostic = self._critic_losses(
                full_action_features,
                flat_action_labels,
                proposal_actions,
                rewards,
                dones,
                progress_targets,
                supervision_valid_mask,
                candidate_actions,
                candidate_q_values,
                batch_size,
                sequence_length,
            )
            regret_loss, contrastive_loss = self._regret_losses(
                policy_world_tokens if self.training else None,
                flat_action_labels,
                proposal_actions,
                context_summary,
                batch_size,
                sequence_length,
                task_ids,
                episode_ids,
                candidate_q_values,
                supervision_valid_mask,
            )
        else:
            # Offline demonstrations lack measured counterfactual returns and
            # progress. Disable these objectives; never impute Q or use the
            # critic's own predictions as substitute supervision.
            critic_loss = progress_loss = candidate_q_diagnostic = context.new_zeros(())
            regret_loss = contrastive_loss = context.new_zeros(())
        aux_losses = {
            "proposal": self._proposal_loss(proposal_actions, flat_action_labels, supervision_valid_mask),
            "mask": self._mask_token_loss(world_tokens, supervision_valid_mask),
            "decision_supervision_active": context.new_tensor(float(
                self.training and decision_supervision_enabled and action_labels is not None
            )),
            "future": (
                self._future_prediction_loss(
                    world_tokens,
                    influence_logits,
                    future_targets,
                    future_valid_mask,
                )
                if self.training
                else world_tokens.new_zeros(())
            ),
            "budget": self._budget_loss(
                influence_logits,
                adaptive_ratios,
                selection_budget_ratio,
                supervision_valid_mask,
            ),
            "critic": critic_loss,
            "candidate_q_diagnostic": candidate_q_diagnostic,
            "progress": progress_loss,
            "regret": regret_loss,
            "contrastive": contrastive_loss,
        }
        return DIWAOutput(
            action_features=action_features.view(
                batch_size,
                sequence_length,
                self.action_pred_steps,
                hidden_dim,
            ),
            full_action_features=full_action_features,
            counterfactual_action_features=counterfactual_action_features,
            counterfactual_indices=counterfactual_indices,
            counterfactual_valid_mask=counterfactual_valid_mask,
            counterfactual_swap_mask=counterfactual_swap_mask,
            sampled_influence_logits=sampled_influence_logits,
            proposal_actions=proposal_actions.view(
                batch_size,
                sequence_length,
                self.action_pred_steps,
                self.action_dim,
            ),
            influence_logits=influence_logits.view(
                batch_size, sequence_length, self.horizon, self.num_slots
            ),
            influence_probabilities=influence_probabilities.view(
                batch_size, sequence_length, self.horizon, self.num_slots
            ),
            selected_indices=selected_indices.view(batch_size, sequence_length, -1),
            selected_valid_mask=selected_valid.view(
                batch_size, sequence_length, -1
            ),
            selected_mask=selected_mask.view(
                batch_size, sequence_length, self.horizon, self.num_slots
            ),
            adaptive_budget_ratio=adaptive_ratios.view(
                batch_size, sequence_length
            ),
            world_tokens=world_tokens,
            selected_world_tokens=selected_world_tokens,
            object_tokens=object_tokens,
            object_assignments=object_assignments,
            aux_losses=aux_losses,
        )

    @torch.no_grad()
    def update_target_networks(self, tau: float) -> None:
        self.critic.update_targets(tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        for online_parameter, target_parameter in zip(
            self.object_tokenizer.parameters(),
            self.target_object_tokenizer.parameters(),
        ):
            target_parameter.lerp_(online_parameter, tau)

    @torch.no_grad()
    def hard_sync_target_networks(self) -> None:
        self.critic.update_targets(1.0)
        self.target_object_tokenizer.load_state_dict(
            self.object_tokenizer.state_dict()
        )
