from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ObjectTokenOutput:
    """Object-centric tokens and their soft spatial assignments."""

    tokens: Tensor
    assignments: Tensor


class SlotAttention(nn.Module):
    """Iterative object binding with competition between slots."""

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int,
        num_iterations: int = 3,
        mlp_ratio: int = 2,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if hidden_dim < 1 or num_slots < 1:
            raise ValueError("hidden_dim and num_slots must be positive")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if mlp_ratio < 1:
            raise ValueError("mlp_ratio must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        self.slot_centers = nn.Parameter(torch.empty(1, num_slots, hidden_dim))
        self.slot_log_scales = nn.Parameter(torch.full((1, num_slots, hidden_dim), -2.0))
        nn.init.normal_(self.slot_centers, std=0.02)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.to_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.scale = hidden_dim**-0.5

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3:
            raise ValueError("slot-attention inputs must be [batch, tokens, hidden]")
        batch_size, _, hidden_dim = inputs.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")

        normalized_inputs = self.input_norm(inputs)
        keys = self.to_key(normalized_inputs)
        values = self.to_value(normalized_inputs)
        scales = self.slot_log_scales.exp().expand(batch_size, -1, -1)
        slots = self.slot_centers.expand(batch_size, -1, -1)
        if self.training:
            slots = slots + scales * torch.randn_like(slots)

        assignments = inputs.new_zeros(batch_size, self.num_slots, inputs.shape[1])
        for _ in range(self.num_iterations):
            previous_slots = slots
            queries = self.to_query(self.slot_norm(slots))
            logits = torch.einsum("bnd,bld->bnl", queries, keys) * self.scale
            # Each observation token competes for one object slot.
            assignments = F.softmax(logits, dim=1) + self.epsilon
            normalized_assignments = assignments / assignments.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.epsilon)
            updates = torch.einsum(
                "bnl,bld->bnd", normalized_assignments, values
            )
            slots = self.gru(
                updates.reshape(-1, hidden_dim),
                previous_slots.reshape(-1, hidden_dim),
            ).view(batch_size, self.num_slots, hidden_dim)
            slots = slots + self.mlp(self.mlp_norm(slots))

        return slots, assignments


class SinkhornTemporalMatcher(nn.Module):
    """Softly preserve object identity between adjacent observations."""

    def __init__(
        self,
        temperature: float = 0.1,
        iterations: int = 5,
        identity_bias: float = 0.05,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if identity_bias < 0:
            raise ValueError("identity_bias must be non-negative")
        self.temperature = temperature
        self.iterations = iterations
        self.identity_bias = identity_bias

    def _sinkhorn(self, logits: Tensor) -> Tensor:
        log_assignment = logits
        for _ in range(self.iterations):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-1, keepdim=True
            )
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-2, keepdim=True
            )
        return log_assignment.exp()

    def forward(
        self, tokens: Tensor, assignments: Tensor
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 4:
            raise ValueError("temporal object tokens must be [batch, time, slots, hidden]")
        aligned_tokens = [tokens[:, 0]]
        aligned_assignments = [assignments[:, 0]]
        num_slots = tokens.shape[2]
        identity = torch.eye(num_slots, device=tokens.device, dtype=tokens.dtype)

        for timestep in range(1, tokens.shape[1]):
            previous = F.normalize(aligned_tokens[-1].detach().float(), dim=-1)
            current = F.normalize(tokens[:, timestep].float(), dim=-1)
            similarities = torch.einsum("bnd,bmd->bnm", previous, current)
            similarities = similarities.to(tokens.dtype)
            similarities = similarities + self.identity_bias * identity
            permutation = self._sinkhorn(similarities / self.temperature)
            aligned_tokens.append(
                torch.einsum("bnm,bmd->bnd", permutation, tokens[:, timestep])
            )
            aligned_assignments.append(
                torch.einsum(
                    "bnm,bml->bnl", permutation, assignments[:, timestep]
                )
            )

        return torch.stack(aligned_tokens, dim=1), torch.stack(
            aligned_assignments, dim=1
        )


class ObjectCentricTokenizer(nn.Module):
    """Build tracked object tokens from VLA, SAM and CoTracker features."""

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int,
        sam_feature_dim: int = 256,
        slot_iterations: int = 3,
        track_image_size: float = 224.0,
    ):
        super().__init__()
        if hidden_dim < 1 or num_slots < 1 or sam_feature_dim < 1:
            raise ValueError(
                "hidden_dim, num_slots and sam_feature_dim must be positive"
            )
        if track_image_size <= 0:
            raise ValueError("track_image_size must be positive")
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.track_image_size = track_image_size
        self.sam_projector = nn.Sequential(
            nn.LayerNorm(sam_feature_dim),
            nn.Linear(sam_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.track_projector = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_embedding = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.source_embedding, std=0.02)
        self.slot_attention = SlotAttention(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_iterations=slot_iterations,
        )
        self.temporal_matcher = SinkhornTemporalMatcher()

    def _prepare_sam(self, features: Optional[Tensor], source: int) -> Optional[Tensor]:
        if features is None:
            return None
        if features.ndim == 5:
            features = features.flatten(2, 3)
        if features.ndim != 4:
            raise ValueError("SAM features must be [batch, time, tokens, channels]")
        return self.sam_projector(features) + self.source_embedding[source]

    def _prepare_tracks(
        self,
        tracks: Optional[Tensor],
        visibility: Optional[Tensor],
        camera_id: float,
    ) -> Optional[Tensor]:
        if tracks is None:
            return None
        if tracks.ndim != 4 or tracks.shape[-1] != 2:
            raise ValueError("tracks must be [batch, time, points, 2]")
        if visibility is None:
            visibility = torch.ones_like(tracks[..., 0])
        if visibility.ndim == 4 and visibility.shape[-1] == 1:
            visibility = visibility.squeeze(-1)
        if visibility.shape != tracks.shape[:-1]:
            raise ValueError(
                "track visibility must have shape [batch, time, points]"
            )
        coordinates = tracks.float() / self.track_image_size
        camera = torch.full_like(visibility.float(), camera_id)
        encoded = torch.cat(
            (
                coordinates,
                visibility.float().unsqueeze(-1),
                camera.unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = self.track_projector(encoded.to(dtype=tracks.dtype))
        return tokens * visibility.unsqueeze(-1).to(tokens.dtype)

    def forward(
        self,
        visual_tokens: Tensor,
        *,
        sam_primary: Optional[Tensor] = None,
        sam_wrist: Optional[Tensor] = None,
        tracks_primary: Optional[Tensor] = None,
        visibility_primary: Optional[Tensor] = None,
        tracks_wrist: Optional[Tensor] = None,
        visibility_wrist: Optional[Tensor] = None,
    ) -> ObjectTokenOutput:
        if visual_tokens.ndim != 4:
            raise ValueError(
                "visual tokens must have shape [batch, time, tokens, hidden]"
            )
        batch_size, sequence_length, _, hidden_dim = visual_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")

        sources = [visual_tokens + self.source_embedding[0]]
        sam_primary_tokens = self._prepare_sam(sam_primary, 1)
        sam_wrist_tokens = self._prepare_sam(sam_wrist, 2)
        if sam_primary_tokens is not None:
            sources.append(sam_primary_tokens)
        if sam_wrist_tokens is not None:
            sources.append(sam_wrist_tokens)
        primary_tracks = self._prepare_tracks(
            tracks_primary, visibility_primary, camera_id=0.0
        )
        wrist_tracks = self._prepare_tracks(
            tracks_wrist, visibility_wrist, camera_id=1.0
        )
        if primary_tracks is not None:
            sources.append(primary_tracks)
        if wrist_tracks is not None:
            sources.append(wrist_tracks)

        for source in sources:
            if source.shape[:2] != (batch_size, sequence_length):
                raise ValueError(
                    "all object-token sources must match the visual batch/time axes"
                )
            if source.ndim != 4 or source.shape[-1] != hidden_dim:
                raise ValueError(
                    "all object-token sources must be [batch, time, tokens, hidden]"
                )

        inputs = torch.cat(
            [source.to(dtype=visual_tokens.dtype) for source in sources], dim=2
        )
        flat_inputs = inputs.reshape(
            batch_size * sequence_length, inputs.shape[2], hidden_dim
        )
        tokens, assignments = self.slot_attention(flat_inputs)
        tokens = tokens.view(
            batch_size, sequence_length, self.num_slots, hidden_dim
        )
        assignments = assignments.view(
            batch_size,
            sequence_length,
            self.num_slots,
            inputs.shape[2],
        )
        tokens, assignments = self.temporal_matcher(tokens, assignments)
        return ObjectTokenOutput(tokens=tokens, assignments=assignments)
