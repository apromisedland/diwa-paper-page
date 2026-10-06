"""Regression tests for scientific semantics and the executable workflow."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diwa_cli import fit, load_policy
from examples.synthetic_fixture import create_fixture
from models.action_model.action_model import ActionModelFM
from models.action_model.respace import space_timesteps
from models.diwa.core import AdaptiveTopKSelector
from models.diwa.feature_data import (
    FeatureWindowDataset,
    PairedBatchSampler,
    load_observations,
)
from models.diwa.feature_policy import FeatureDIWAPolicy, load_config, schedule


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    torch.set_num_threads(2)
    return load_config(ROOT / "configs/diwa_cpu.json")


def fixture_batch(tmp_path, config):
    manifest, observations = create_fixture(tmp_path, config)
    dataset = FeatureWindowDataset(manifest, config, allow_synthetic=True)
    sampler = PairedBatchSampler(dataset, 2, 42)
    return (
        next(iter(DataLoader(dataset, batch_sampler=sampler))),
        manifest,
        observations,
    )


def test_terminal_window_keeps_labels_and_masks_missing_future(tmp_path, config):
    manifest, _ = create_fixture(tmp_path / "terminal", config)
    dataset = FeatureWindowDataset(manifest, config, allow_synthetic=True)
    episode_length = config["training"]["sequence_length"] + config["core"][
        "horizon"
    ] + 2
    terminal_start = episode_length - config["training"]["sequence_length"]
    index = next(
        index
        for index, (episode, start) in enumerate(dataset.windows)
        if episode == 0 and start == terminal_start
    )
    sample = dataset[index]
    assert sample["dones"][-1]
    assert sample["observation_valid"].tolist() == (
        [True] * config["training"]["sequence_length"]
        + [False] * config["core"]["horizon"]
    )


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_adaptive_selector_rejects_invalid_budgets(value):
    with pytest.raises(ValueError, match="finite and in"):
        AdaptiveTopKSelector()(torch.zeros(1, 48), torch.tensor([value]))


def test_td_bootstrap_uses_next_proposal(tmp_path, config, monkeypatch):
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config)
    captured = {}
    original = model.core.critic.losses

    def capture(**kwargs):
        captured["next_actions"] = kwargs["next_actions"].detach()
        return original(**kwargs)

    monkeypatch.setattr(model.core.critic, "losses", capture)
    _, _, output = model(batch)
    proposals = output.proposal_actions.detach()
    expected = torch.cat((proposals[:, 1:], proposals[:, -1:]), dim=1).flatten(0, 1)
    assert torch.equal(captured["next_actions"], expected)
    assert not torch.equal(expected[:1], batch["actions"][:, 1:].flatten(0, 1)[:1])


def test_strict_core_never_falls_back_to_critic_returns(tmp_path, config):
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config)
    batch["candidate_q_values"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="candidate_q_values"):
        model(batch)
    with pytest.raises(ValueError, match="strict DIWA supervision is missing"):
        model.core(torch.randn(2, 2, 3, config["core"]["hidden_dim"]))


def test_influence_targets_are_detached_and_head_mode_restored(tmp_path, config):
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config).train()
    _, losses, _ = model(batch)
    losses["influence"].backward()
    assert model.action_model.net.training
    assert model.core.influence_estimator[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.action_model.parameters())
    assert all(p.grad is None for p in model.core.critic.parameters())
    assert all(p.grad is None for p in model.core.world_model.parameters())


@pytest.mark.parametrize("kind", ["diffusion", "flow"])
def test_inference_has_no_label_dependency_and_decodes_only_selected(
    tmp_path, config, kind
):
    config["action_head"]["kind"] = kind
    _, _, observations = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config).eval()
    batch = load_observations(observations, config["core"]["hidden_dim"])
    noise = torch.randn(1, 3, 7)
    decoded = []
    hook = model.core.world_model.register_forward_pre_hook(
        lambda module, args: decoded.append(args[0].shape[1])
    )
    first = model.predict(batch, initial_noise=noise)
    second = model.predict(
        dict(batch, candidate_q_values=torch.tensor(float("nan"))), initial_noise=noise
    )
    hook.remove()
    assert torch.equal(first["actions"], second["actions"])
    assert decoded == [12, 12]
    assert first["selected_valid_mask"].sum() <= 12
    assert torch.isfinite(first["actions"]).all()
    assert first["actions"].shape == (1, 3, 7)


def test_sampler_pairs_distinct_episodes_of_same_task(tmp_path, config):
    manifest, _ = create_fixture(tmp_path, config)
    with pytest.raises(ValueError, match="data_kind=measured"):
        FeatureWindowDataset(manifest, config)
    dataset = FeatureWindowDataset(manifest, config, allow_synthetic=True)
    for batch in PairedBatchSampler(dataset, 4, 42):
        for index in (0, 2):
            left, right = (dataset.windows[i][0] for i in batch[index : index + 2])
            assert left != right
            assert dataset.task_ids[left] == dataset.task_ids[right]


@pytest.mark.parametrize(
    "field,pattern",
    [
        ("candidate_q_values", "finite"),
        ("progress", "\\[0, 1\\]"),
        ("candidate_rule_ids", "order"),
        ("done", "binary"),
    ],
)
def test_data_rejects_invalid_supervision(tmp_path, config, field, pattern):
    manifest, _ = create_fixture(tmp_path, config)
    path = tmp_path / "synthetic_episode_0.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if field == "candidate_q_values":
        arrays[field][0, 0] = float("nan")
    elif field == "progress":
        arrays[field][0] = 1.5
    elif field == "done":
        arrays[field] = arrays[field].astype(float)
        arrays[field][0] = 0.5
    else:
        arrays[field] = arrays[field][::-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=pattern):
        FeatureWindowDataset(manifest, config, allow_synthetic=True)


def test_epoch_resume_matches_uninterrupted_training(tmp_path, config):
    # Eight minibatches per epoch with accumulation=3 exercises the short
    # final accumulation group as well as optimizer/scheduler/RNG restoration.
    config["training"]["gradient_accumulation"] = 3
    manifest, observations = create_fixture(tmp_path / "fixture", config)
    full, full_model = fit(config, manifest, tmp_path / "full", allow_synthetic=True)
    split, _ = fit(
        config, manifest, tmp_path / "split", allow_synthetic=True, epochs_to_run=1
    )
    resumed, resumed_model = fit(
        config,
        manifest,
        tmp_path / "split",
        resume=split["checkpoint"],
        allow_synthetic=True,
    )
    assert full["optimizer_steps"] == resumed["optimizer_steps"] == 6
    for key, value in full_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[key]), key
    restored, _ = load_policy(resumed["checkpoint"])
    batch = load_observations(observations, config["core"]["hidden_dim"])
    noise = torch.randn(1, 3, 7)
    assert torch.equal(
        restored.predict(batch, initial_noise=noise)["actions"],
        full_model.eval().predict(batch, initial_noise=noise)["actions"],
    )


def test_flow_sampler_uses_provided_cpu_noise():
    head = ActionModelFM(32, "DiT-Debug", 7, 2, 0, diffusion_steps=10)
    sampler = head.create_ddim(5)
    noise = torch.randn(2, 3, 7, dtype=torch.float64)
    kwargs = {"cfg_scale": 1.5}
    call_times = []

    def zero_flow(x, timestep, **_kwargs):
        call_times.append(timestep.clone())
        return torch.zeros_like(x)

    result = sampler.ddim_sample_loop(
        zero_flow,
        noise.shape,
        noise,
        model_kwargs=kwargs,
        device="cpu",
    )
    assert torch.equal(noise, result)
    assert result.dtype == torch.float64
    assert kwargs["cfg_scale"] == 1.5
    assert len(call_times) == 5
    assert torch.equal(
        torch.stack(call_times)[:, 0],
        torch.arange(5, dtype=torch.float64) / 5,
    )


def test_one_step_ddim_uses_final_diffusion_timestep():
    assert space_timesteps(10, "ddim1") == {9}
    with pytest.raises(ValueError, match="positive"):
        space_timesteps(10, "ddim0")


def test_paper_stage_boundaries():
    config = load_config(ROOT / "configs/diwa_paper.json")
    budget, dense, teacher, weights = schedule(config, 0)
    assert dense and budget == 1 and teacher == 1
    assert weights["influence"] == weights["regret"] == weights["critic"] == 0
    _, dense, _, weights = schedule(config, 1000)
    assert not dense and weights["influence"] > 0 and weights["critic"] > 0
    assert weights["regret"] == 0
    assert schedule(config, 2000)[3]["regret"] > 0
    assert schedule(config, 11000)[0] == 0.25


def test_counterfactual_branch_respects_influence_start(tmp_path, config):
    config["training"]["influence_start"] = 10
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config).train()
    _, _, before = model(batch, micro_step=0)
    _, _, after = model(batch, micro_step=10)
    assert before.counterfactual_action_features is None
    assert before.sampled_influence_logits is None
    assert after.counterfactual_action_features is not None
    assert after.sampled_influence_logits is not None


def test_future_loss_normalizes_only_over_valid_targets(config):
    model = FeatureDIWAPolicy(config)
    candidates = model.core.num_candidates
    hidden = model.core.hidden_dim
    world = torch.zeros(1, candidates, hidden)
    targets = torch.zeros(1, 1, model.core.horizon, model.core.num_slots, hidden)
    targets[..., 0] = 1.0
    valid = torch.zeros(1, 1, model.core.horizon, dtype=torch.bool)
    valid[..., 0] = True
    logits = torch.zeros(1, candidates)
    baseline = model.core._future_prediction_loss(
        world, logits, targets, valid
    )
    masked_logits = logits.clone()
    masked_logits[:, 1:] = 100.0
    changed = model.core._future_prediction_loss(
        world, masked_logits, targets, valid
    )
    assert torch.equal(baseline, changed)
