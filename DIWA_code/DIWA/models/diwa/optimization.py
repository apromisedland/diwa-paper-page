"""Shared optimizer-step accounting for the two DIWA training entry points."""

import math

import torch


def accumulation_divisor(batch_index, batch_count, accumulation_steps):
    if accumulation_steps < 1 or not 0 <= batch_index < batch_count:
        raise ValueError("invalid minibatch/accumulation counts")
    group_start = batch_index // accumulation_steps * accumulation_steps
    return min(accumulation_steps, batch_count - group_start)


def make_lr_scheduler(
    optimizer,
    batch_count,
    epochs,
    warmup_epochs,
    accumulation_steps,
    kind="cosine",
):
    if min(batch_count, epochs, accumulation_steps) < 1 or not 0 <= warmup_epochs <= epochs:
        raise ValueError("invalid training or warm-up duration")
    if kind not in ("constant", "linear", "cosine"):
        raise ValueError("DIWA scheduler must be constant, linear, or cosine")
    # Each epoch flushes its last partial accumulation group.
    updates_per_epoch = math.ceil(batch_count / accumulation_steps)
    total = updates_per_epoch * epochs
    warmup = updates_per_epoch * warmup_epochs

    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if kind == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 1.0 - progress if kind == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
