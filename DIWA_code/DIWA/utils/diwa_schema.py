"""Shared schema helpers for measured DIWA supervision."""

from __future__ import annotations

from collections.abc import Iterable


_CARTESIAN_AXES = ("x", "y", "z", "rx", "ry", "rz")


def stable_candidate_rule_ids(count: int) -> tuple[str, ...]:
    """Return the canonical rule identities used by the LIBERO collector.

    The first two rules replay the demonstration and suppress continuous
    motion while retaining the demonstrated gripper command. Remaining rules
    add signed Cartesian perturbations in the exact order implemented by
    ``candidate_bank``. Counts above fourteen repeat the axes at larger
    perturbation scales and receive an explicit scale suffix.
    """
    if count < 4:
        raise ValueError("DIWA candidate banks require at least four rules")
    rules = ["demonstration", "no_motion_keep_gripper"]
    for rule_index in range(count - 2):
        axis = _CARTESIAN_AXES[(rule_index // 2) % len(_CARTESIAN_AXES)]
        direction = "plus" if rule_index % 2 == 0 else "minus"
        scale = 1 + rule_index // (2 * len(_CARTESIAN_AXES))
        suffix = "" if scale == 1 else f"_x{scale}"
        rules.append(f"{direction}_{axis}{suffix}")
    return tuple(rules)


def normalize_candidate_rule_ids(values: Iterable[object]) -> tuple[str, ...]:
    """Decode an NPZ/HDF5 string vector and reject ambiguous identities."""
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError("candidate_rule_ids must be a one-dimensional vector") from exc
    rules = []
    for value in raw_values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("candidate_rule_ids must contain nonempty strings")
        rules.append(value.strip())
    if not rules or len(set(rules)) != len(rules):
        raise ValueError("candidate_rule_ids must be nonempty and unique")
    return tuple(rules)


def validate_candidate_rule_ids(
    values: Iterable[object],
    *,
    candidate_count: int,
) -> tuple[str, ...]:
    """Require the exact rule order used by the measured collector."""
    rules = normalize_candidate_rule_ids(values)
    expected = stable_candidate_rule_ids(candidate_count)
    if rules != expected:
        raise ValueError(
            "candidate_rule_ids order differs from the canonical measured "
            f"bank: expected {expected}, received {rules}"
        )
    return rules
