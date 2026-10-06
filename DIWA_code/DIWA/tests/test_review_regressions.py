"""Executable regressions for the five issues found in the source review.

The heavyweight image/CLIP encoders are bypassed in integration tests. Their
actual DIWA forward methods, core, losses and dataloader factory are executed
from the source AST, without optional dependencies or model downloads.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
import os
import sys

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from diwa_cli import fit
from models.diwa import DIWACore
from models.diwa.feature_policy import FeatureDIWAPolicy, load_config
from models.diwa.influence import intervention_influence_loss
from scripts.MULTI_DATASET.DIWA import run_exported_smoke
from tests.test_cross_dataset_smoke import smoke_payload
from tests.test_feature_workflow import fixture_batch
from utils.arguments_utils import get_parser, validate_diwa_args
from utils.convert_libero_per_step import DatasetConverter
from utils.dataloader_utils import (
    DistributedTaskPairBatchSampler,
    set_dataloader_epoch_metadata,
)
from utils.diwa_schema import stable_candidate_rule_ids, validate_candidate_rule_ids


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    torch.set_num_threads(2)
    return load_config(ROOT / "configs/diwa_cpu.json")


def source_definition(path, class_name, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    parent = next(n for n in tree.body if getattr(n, "name", None) == class_name) if class_name else tree
    node = next(n for n in parent.body if getattr(n, "name", None) == name)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    scope = dict(torch=torch, np=np, os=os, h5py=h5py,
                 intervention_influence_loss=intervention_influence_loss,
                 validate_candidate_rule_ids=validate_candidate_rule_ids)
    scope.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope[name]


def bind_source(obj, path, class_name, name):
    setattr(obj, name, MethodType(source_definition(path, class_name, name), obj))


def native_demo():
    n = 2
    return {
        "obs": {
            "agentview_rgb": np.zeros((n, 8, 8, 3), dtype=np.uint8),
            "eye_in_hand_rgb": np.zeros((n, 8, 8, 3), dtype=np.uint8),
            **{key: np.zeros((n, width), dtype=np.float32) for key, width in (
                ("joint_states", 7), ("ee_pos", 3), ("ee_ori", 3),
                ("ee_states", 6), ("gripper_states", 2),
            )},
        },
        "actions": np.zeros((n, 7), dtype=np.float32),
        "rewards": np.asarray([0.0, 1.0]),
        "dones": np.asarray([False, True]),
    }


def test_native_libero_outcomes_convert_and_accept_external_diwa_sidecar(tmp_path):
    demo = native_demo()
    source = tmp_path / "native.h5"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("demo_0")
        for key, value in demo.items():
            if isinstance(value, dict):
                obs = group.create_group(key)
                for name, array in value.items():
                    obs.create_dataset(name, data=array)
            else:
                group.create_dataset(key, data=value)
    converter = DatasetConverter(tmp_path, tmp_path, 0, 1, 0, 1)
    with h5py.File(source, "r") as handle:
        converter.process_episode(tmp_path / "episodes", "pick cup", handle, 0, 0)
    step = tmp_path / "episodes/000000/steps/0001/other.h5"
    sidecar = tmp_path / "diwa/000000/steps/0001.npz"
    sidecar.parent.mkdir(parents=True)
    np.savez(sidecar, reward=1.0, done=True, progress=1.0,
             candidate_actions=np.zeros((4, 2, 7), dtype=np.float32),
             candidate_q_values=np.arange(4, dtype=np.float32),
             candidate_rule_ids=np.asarray(stable_candidate_rule_ids(4)))
    loader = SimpleNamespace(load_diwa_supervision=True, require_diwa_supervision=True,
                             diwa_supervision_path=str(tmp_path / "diwa"),
                             diwa_action_dim=7, diwa_action_pred_steps=2,
                             diwa_regret_candidates=4)
    bind_source(loader, "utils/data_utils.py", "BaseLiberoDataset", "_load_diwa_step_supervision")
    with h5py.File(step, "r") as handle:
        assert handle["reward"][()] == 1.0 and handle["done"][()]
        assert "progress" not in handle and "candidate_q_values" not in handle
        supervision = loader._load_diwa_step_supervision("000000", 1, handle)
    assert supervision["valid"] and supervision["progress"] == 1.0
    np.testing.assert_array_equal(supervision["candidate_q_values"], np.arange(4))


@pytest.mark.parametrize("field", ["progress", "candidate_actions", "candidate_q_values", "candidate_rule_ids"])
def test_converter_still_rejects_partial_diwa_annotations(tmp_path, field):
    demo = native_demo()
    demo[field] = np.zeros(2)
    converter = DatasetConverter(tmp_path, tmp_path, 0, 1, 0, 1)
    with pytest.raises(ValueError, match="partial DIWA supervision"):
        converter.process_episode(tmp_path / "episodes", "pick cup", {"demo_0": demo}, 0, 0)


class EncodedDreamVLA(nn.Module):
    """Actual DIWA methods operating on causal pre-encoded visual tokens."""

    def __init__(self, **kwargs):
        super().__init__()
        self.NUM_RESAMPLER_QUERY = 1
        self.diwa_horizon = 1
        self.diwa_num_slots = 4
        self.diwa_eval_budget_ratio = 0.5
        self.diwa_require_supervision = kwargs["diwa_require_supervision"]
        self.diwa_core = DIWACore(
            hidden_dim=16, num_heads=4, action_pred_steps=2, horizon=1,
            num_slots=4, world_layers=1, fusion_layers=1, regret_candidates=4,
            counterfactual_samples=2, require_measured_supervision=self.diwa_require_supervision,
        )
        self.use_dit_head = False
        self.register_buffer("diwa_influence_scale", torch.ones(()))
        self.diwa_influence_probes = 4
        for name in ("covariance", "policy", "value", "progress"):
            setattr(self, f"diwa_influence_{name}_weight", 1.0)
        self.decoder = nn.Linear(16, 7)
        for name in ("_forward_diwa", "_build_diwa_future_targets", "_diwa_influence_loss"):
            bind_source(self, "models/dreamvla_model.py", "DreamVLA", name)

    def _init_model_type(self):
        pass

    def _forward_diwa_context(self, embeddings):
        return embeddings

    def _decode_mlp_action_features(self, features):
        raw = self.decoder(features)
        return raw[..., :6].tanh(), raw[..., 6:].sigmoid()

    def forward(self, embeddings, labels=None, supervision=None, tracks=None, mode="train"):
        return self._forward_diwa(
            embeddings, labels, mode, 0.5, False, 0.0, None,
            track_infos=tracks, supervision=supervision,
        )


def offline_batch(tmp_path):
    payload = smoke_payload(7, 6, 6, 1)
    payload.update(dataset=np.asarray("DROID"), require_measured_candidate_q=np.asarray(False),
                   candidate_q_values=np.full((2, 2, 4), np.nan, dtype=np.float32),
                   progress=np.full((2, 2), np.nan, dtype=np.float32))
    path = tmp_path / "offline.npz"
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as batch:
        contract = run_exported_smoke.validate_batch(batch)
    supervision = {name: torch.from_numpy(payload[source]) for name, source in (
        ("rewards", "rewards"), ("dones", "dones"), ("progress", "progress"),
        ("candidate_actions", "candidate_actions"), ("candidate_q_values", "candidate_q_values"),
    )}
    supervision.update(valid=torch.ones(2, 2, dtype=torch.bool),
                       task_ids=torch.tensor([0, 0]), episode_ids=torch.tensor([0, 1]),
                       decision_supervision_enabled=contract["decision_supervision_active"])
    return contract, supervision


def test_offline_nan_export_runs_real_diwa_forward_without_decision_losses(tmp_path, monkeypatch):
    contract, supervision = offline_batch(tmp_path)
    monkeypatch.setitem(sys.modules, "models.dreamvla_model", SimpleNamespace(DreamVLA=EncodedDreamVLA))
    model = run_exported_smoke.build_model(contract, torch.device("cpu")).train()
    assert not model.diwa_core.require_measured_supervision
    assert contract["training_mode"] == "offline_imitation"
    def forbidden(*args, **kwargs):
        raise AssertionError("offline labels must not reach critic or regret objectives")
    monkeypatch.setattr(model.diwa_core, "_critic_losses", forbidden)
    monkeypatch.setattr(model.diwa_core, "_regret_losses", forbidden)
    monkeypatch.setattr(model.diwa_core.critic, "minimum_q", forbidden)
    monkeypatch.setattr(model.diwa_core.critic, "progress", forbidden)
    output = model(torch.randn(2, 3, 6, 16), torch.zeros(2, 2, 2, 7), supervision)
    for name in ("critic", "progress", "regret", "contrastive", "candidate_q_diagnostic"):
        assert output["aux_losses"][name] == 0
    losses = run_exported_smoke.active_auxiliary_losses(contract)
    loss = output["arm_action"].square().mean() + sum(output["aux_losses"][name] for name in losses)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.diwa_core.influence_estimator[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.diwa_core.critic.parameters())
    assert model.diwa_core.counterfactual_token.grad.norm() > 0


@pytest.mark.parametrize("mode", ["train", "test"])
def test_future_track_labels_cannot_change_policy_outputs(tmp_path, mode):
    torch.manual_seed(17)
    _, supervision = offline_batch(tmp_path)
    model = EncodedDreamVLA(diwa_require_supervision=False)
    model.train(mode == "train")
    embeddings = torch.randn(2, 3, 6, 16)
    labels = torch.zeros(2, 2, 2, 7) if mode == "train" else None
    def predict(value):
        torch.manual_seed(91)
        tracks = {"tracks": torch.full((2, 3, 7, 2), value),
                  "track_visibility": torch.ones(2, 3, 7)}
        return model(embeddings, labels, supervision, tracks, mode)
    first, second = predict(0.0), predict(224.0)
    for key in ("arm_action", "gripper_action", "proposal_actions", "influence_logits",
                "selected_indices", "object_tokens", "decision_latents"):
        assert torch.equal(first[key], second[key]), key
    if mode == "train":
        assert first["aux_losses"]["future"] != second["aux_losses"]["future"]


def test_known_simulator_cannot_opt_out_of_measured_q(tmp_path):
    payload = smoke_payload(7, 6, 6, 1)
    payload.update(dataset=np.asarray("CALVIN"), require_measured_candidate_q=np.asarray(False),
                   candidate_rule_ids=np.asarray(run_exported_smoke.EXPECTED_CANDIDATE_RULES["CALVIN"]),
                   candidate_q_values=np.full((2, 2, 4), np.nan, dtype=np.float32))
    path = tmp_path / "calvin.npz"
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as batch, pytest.raises(ValueError, match="finite"):
        run_exported_smoke.validate_batch(batch)


@pytest.mark.parametrize("replicas,batch_size", [(1, 2), (2, 4), (5, 2)])
def test_distributed_pairs_cover_anchors_and_keep_all_ranks_balanced(replicas, batch_size):
    windows = {"a": range(0, 3), "b": range(3, 5), "c": range(5, 6), "d": range(6, 11)}
    tasks = {"a": "pick", "b": "pick", "c": "push", "d": "push"}
    identity = {index: ep for ep, values in windows.items() for index in values}
    samplers = [DistributedTaskPairBatchSampler(windows, tasks, batch_size,
                num_replicas=replicas, rank=rank, seed=7) for rank in range(replicas)]
    epochs = [list(sampler) for sampler in samplers]
    assert len({len(batches) for batches in epochs}) == 1
    anchors = set()
    for sampler, batches in zip(samplers, epochs):
        assert batches == list(sampler)
        loader = DataLoader(list(range(11)), batch_sampler=sampler)
        set_dataloader_epoch_metadata(loader, batch_size=batch_size, world_size=replicas)
        assert len(batches) == loader.num_batches == len(loader)
        assert loader.num_samples == len(batches) * batch_size * replicas
        for batch in batches:
            assert len(batch) == batch_size
            for left, right in zip(batch[::2], batch[1::2]):
                anchors.add(left)
                assert identity[left] != identity[right]
                assert tasks[identity[left]] == tasks[identity[right]]
        sampler.set_epoch(1)
        assert batches != list(sampler)
        sampler.set_epoch(0)
        assert batches == list(sampler)
    assert anchors == set(range(11))


def test_pair_sampler_rejects_unpairable_tasks_and_odd_batches():
    with pytest.raises(ValueError, match="two distinct episodes"):
        DistributedTaskPairBatchSampler({"a": range(2)}, {"a": "pick"}, 2)
    with pytest.raises(ValueError, match="even"):
        DistributedTaskPairBatchSampler({"a": range(2)}, {"a": "pick"}, 3)


def test_strict_core_rejects_missing_partners_and_decision_opt_out(tmp_path, config):
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config)
    batch["task_ids"] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="same-task, different-episode"):
        model(batch)
    with pytest.raises(ValueError, match="cannot disable"):
        model.core(torch.randn(2, 2, 4, 32), decision_supervision_enabled=False)


def test_mask_token_is_updated_without_training_target_or_scorer(tmp_path, config):
    batch, _, _ = fixture_batch(tmp_path, config)
    model = FeatureDIWAPolicy(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.core.counterfactual_token.detach().clone()
    _, losses, _ = model(batch)
    losses["mask"].backward()
    assert model.core.counterfactual_token.grad.norm() > 0
    assert all(p.grad is None for p in model.core.world_model.parameters())
    assert all(p.grad is None for p in model.core.influence_estimator.parameters())
    optimizer.step()
    assert not torch.equal(before, model.core.counterfactual_token)
    optimizer.zero_grad(set_to_none=True)
    _, losses, _ = model(batch)
    losses["influence"].backward()
    assert model.core.counterfactual_token.grad is None


def test_mask_calibration_excludes_padding_and_detaches_reference():
    model = DIWACore(hidden_dim=16, num_heads=4, num_slots=4, horizon=1, action_pred_steps=2)
    tokens = torch.randn(2, 4, 16, requires_grad=True)
    valid = torch.tensor([[True], [False]])
    loss = model._mask_token_loss(tokens, valid)
    changed = tokens.detach().clone()
    changed[1] = 1e8
    assert torch.equal(loss, model._mask_token_loss(changed, valid))
    loss.backward()
    assert tokens.grad is None
    assert model.counterfactual_token.grad.norm() > 0


def test_odd_image_training_batch_is_rejected():
    args = get_parser().parse_args(["--save_checkpoint_path", "/tmp/checkpoints",
                                   "--phase", "finetune", "--use_diwa", "--window_size", "10", "--batch_size", "3"])
    with pytest.raises(ValueError, match="even batch_size"):
        validate_diwa_args(args, training=True)


def test_old_feature_training_checkpoint_is_not_silently_resumed(tmp_path, config):
    _, manifest, _ = fixture_batch(tmp_path / "fixture", config)
    _, _ = fit(config, manifest, tmp_path / "run", allow_synthetic=True, epochs_to_run=1)
    path = tmp_path / "run/last.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["format_version"] = 2
    checkpoint.pop("method_revision")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="earlier DIWA training method"):
        fit(config, manifest, tmp_path / "run", resume=path, allow_synthetic=True)


@pytest.mark.parametrize("entry", ["get_libero_finetune_dataset", "get_libero_pretrain_dataset"])
def test_image_dataset_pair_metadata_and_loader_factory(tmp_path, entry):
    datasets = []
    for dataset_name in ("one", "two"):
        base = SimpleNamespace(dataset_path=str(tmp_path / dataset_name),
                               episode_list=["000000", "000001"],
                               num_step_per_episode=[2, 3], load_libero_file="h5",
                               language_mode="language_instruction")
        bind_source(base, "utils/data_utils.py", "BaseLiberoDataset", "load_language_instruction")
        for episode in base.episode_list:
            path = Path(base.dataset_path) / "episodes" / episode / "steps/0000/other.h5"
            path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("language_instruction", data="pick cup")
        datasets.append(base)
    class ImageDataset:
        def __init__(self, **kwargs):
            self.datasets = datasets
            assert kwargs["load_diwa_supervision"]
            assert kwargs["diwa_sequence_length"] is not None
        def __len__(self):
            return 10
        def collator(self, samples):
            return samples
    ImageDataset.paired_window_metadata = source_definition(
        "utils/data_utils.py", "DiskLiberoDataset", "paired_window_metadata")
    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch
        def set_value(self, epoch):
            self.epoch = epoch
    data_info = source_definition("utils/data_utils.py", None, "DataInfo", {
        "dataclass": dataclass, "DataLoader": DataLoader,
    })
    loader_helper = source_definition("utils/data_utils.py", None, "_libero_training_data_info", {
        "DataInfo": data_info, "DataLoader": DataLoader,
        "DistributedTaskPairBatchSampler": DistributedTaskPairBatchSampler,
        "set_dataloader_epoch_metadata": set_dataloader_epoch_metadata,
    })
    factory = source_definition("utils/data_utils.py", None, entry, {
        "functools": __import__("functools"), "SharedEpoch": Epoch,
        "preprocess_image": lambda *args, **kwargs: None,
        "preprocess_text_calvin": lambda *args, **kwargs: None,
        "DiskLiberoDataset": ImageDataset, "DataInfo": data_info,
        "DataLoader": DataLoader, "DistributedTaskPairBatchSampler": DistributedTaskPairBatchSampler,
        "set_dataloader_epoch_metadata": set_dataloader_epoch_metadata,
        "_libero_training_data_info": loader_helper,
    })
    args = get_parser().parse_args(["--save_checkpoint_path", "/tmp/checkpoints", "--phase", "finetune", "--use_diwa", "--batch_size", "2"])
    args.world_size, args.rank, args.workers = 2, 0, 1
    result = factory(args, None, None)
    assert isinstance(result.dataloader.batch_sampler, DistributedTaskPairBatchSampler)
    assert len(result.sampler.windows) == 4  # Same numeric episode IDs in different roots stay distinct.
    before = list(result.sampler)
    result.set_epoch(3)
    assert result.sampler.epoch == 3 and result.shared_epoch.epoch == 3
    assert list(result.sampler) != before
