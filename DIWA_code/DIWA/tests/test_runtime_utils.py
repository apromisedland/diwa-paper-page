import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.action_utils import (
    binarize_discrete_action_channels,
    decode_policy_actions,
    select_policy_state,
)
from utils.dataloader_utils import padded_window_count, set_dataloader_epoch_metadata
from utils.model_utils import freeze_vision_backbone
from utils.train_utils import trajectory_prediction_to_rgb


def test_action_helpers_support_non_libero_dimensions():
    actions = torch.tensor(
        [[*range(10), -1.0, 1.0]],
        dtype=torch.float32,
    )
    converted = binarize_discrete_action_channels(
        actions,
        continuous_action_dim=10,
    )
    assert converted.shape[-1] == 12
    assert converted[0, 10:].tolist() == [0.0, 1.0]

    state = torch.arange(20, dtype=torch.float32).reshape(1, 20)
    selected = select_policy_state(
        state,
        state_arm_dim=14,
        state_gripper_dim=2,
        gripper_width=True,
    )
    assert selected.shape[-1] == 16
    assert selected[0, -2:].tolist() == [18.0, 19.0]

    decoded = decode_policy_actions(
        torch.zeros(1, 10),
        torch.tensor([[0.2, 0.8]]),
    )
    assert decoded.shape[-1] == 12
    assert decoded[0, -2:].tolist() == [-1.0, 1.0]


class _SingleVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)


class _DinoSiglipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_featurizer = nn.Linear(2, 2)
        self.siglip_featurizer = nn.Linear(2, 2)


def test_freeze_vision_backbone_handles_both_configurations():
    single = _SingleVisionModel()
    freeze_vision_backbone(
        single,
        use_dinosiglip=False,
        convert_to_bfloat16=True,
    )
    assert single.vision_encoder.weight.dtype == torch.bfloat16
    assert not any(parameter.requires_grad for parameter in single.parameters())

    combined = _DinoSiglipModel()
    freeze_vision_backbone(combined, use_dinosiglip=True)
    assert not any(parameter.requires_grad for parameter in combined.parameters())


def test_trajectory_logging_reconstructs_shuffled_and_dense_flow():
    shuffled = trajectory_prediction_to_rgb(torch.zeros(196, 8))
    dense = trajectory_prediction_to_rgb(torch.zeros(784, 2))
    assert shuffled.shape == (28, 28, 3)
    assert dense.shape == (28, 28, 3)


def test_dataloader_metadata_uses_actual_batches_not_worker_rounding():
    loader = DataLoader(
        TensorDataset(torch.arange(11)),
        batch_size=3,
        drop_last=True,
        num_workers=0,
    )
    set_dataloader_epoch_metadata(loader, batch_size=3, world_size=2)
    assert loader.num_batches == len(loader) == 3
    assert loader.num_samples == 18


def test_dataloader_metadata_rejects_empty_training_epoch():
    loader = DataLoader(
        TensorDataset(torch.arange(2)),
        batch_size=3,
        drop_last=True,
        num_workers=0,
    )
    try:
        set_dataloader_epoch_metadata(loader, batch_size=3, world_size=1)
    except ValueError as error:
        assert "no complete batches" in str(error)
    else:
        raise AssertionError("empty training loader must be rejected")


def test_padded_window_count_keeps_terminal_supervision():
    assert padded_window_count(20, 7) == 14
    assert padded_window_count(3, 7) == 1
