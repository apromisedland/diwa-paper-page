"""Explicit loader hook for user-supplied private real-robot datasets."""

from __future__ import annotations

import importlib


def resolve_real_dataset_adapter(specification: str | None) -> type:
    """Resolve ``module.path:ClassName`` without assuming a private schema."""
    if not specification:
        raise RuntimeError(
            "finetune_type=real requires --real_dataset_adapter "
            "module.path:ClassName; the private real-robot loader is not "
            "part of this public release"
        )
    module_name, separator, class_name = specification.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(
            "real_dataset_adapter must use module.path:ClassName syntax"
        )
    module = importlib.import_module(module_name)
    adapter = getattr(module, class_name, None)
    if adapter is None or not isinstance(adapter, type):
        raise TypeError(
            f"{specification} does not resolve to a dataset class"
        )
    return adapter
