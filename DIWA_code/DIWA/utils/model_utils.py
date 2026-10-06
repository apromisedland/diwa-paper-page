"""Small model-configuration helpers shared by training and evaluation."""

from __future__ import annotations

from torch import nn


def freeze_vision_backbone(
    model: nn.Module,
    *,
    use_dinosiglip: bool,
    convert_to_bfloat16: bool = False,
) -> None:
    """Freeze the configured vision backbone and optionally cast it to BF16."""
    names = (
        ("dino_featurizer", "siglip_featurizer")
        if use_dinosiglip
        else ("vision_encoder",)
    )
    missing = [name for name in names if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "model is missing the configured vision backbone module(s): "
            + ", ".join(missing)
        )
    for name in names:
        module = getattr(model, name)
        if convert_to_bfloat16:
            module.bfloat16()
        module.requires_grad_(False)
