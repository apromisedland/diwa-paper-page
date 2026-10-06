import unittest
import random

import numpy as np
import torch
from torch import nn

from utils.checkpoint_utils import (
    CHECKPOINT_COMPATIBILITY_ARGUMENTS,
    DIWA_CHECKPOINT_SCHEMA_VERSION,
    RESUME_COMPATIBILITY_ARGUMENTS,
    atomic_torch_save,
    capture_rng_state,
    frozen_parameter_signature,
    gather_rng_states,
    restore_rank_rng_state,
    restore_rng_state,
    validate_diwa_checkpoint,
    validate_pretrained_checkpoint,
    validate_resume_arguments,
)


class _SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.diwa_core = nn.Linear(2, 2)
        self.diwa_future_position_embedding = nn.Parameter(torch.zeros(1))
        self.register_buffer("diwa_influence_scale", torch.ones(()))


class _ScalarFrozenModel(_SmallModel):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)


class CheckpointValidationTest(unittest.TestCase):
    def test_current_full_training_checkpoint_schema(self):
        self.assertEqual(DIWA_CHECKPOINT_SCHEMA_VERSION, 4)

    def test_accepts_complete_diwa_state(self):
        model = _SmallModel()
        validate_diwa_checkpoint(model, model.state_dict())

    def test_rejects_baseline_checkpoint(self):
        model = _SmallModel()
        baseline = {
            key: value
            for key, value in model.state_dict().items()
            if "diwa" not in key
        }
        with self.assertRaises(RuntimeError):
            validate_diwa_checkpoint(model, baseline)

    def test_rejects_missing_trainable_backbone_state(self):
        model = _SmallModel()
        state = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("backbone.")
        }
        with self.assertRaisesRegex(RuntimeError, "backbone"):
            validate_diwa_checkpoint(model, state)

    def test_rejects_unknown_and_wrong_shape_state(self):
        model = _SmallModel()
        unknown = dict(model.state_dict())
        unknown["module.not_in_model"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "absent"):
            validate_diwa_checkpoint(model, unknown)

        wrong_shape = dict(model.state_dict())
        wrong_shape["diwa_core.weight"] = torch.ones(3, 3)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            validate_diwa_checkpoint(model, wrong_shape)

    def test_rejects_checkpoint_argument_mismatch(self):
        model = _SmallModel()
        saved = {name: 0 for name in CHECKPOINT_COMPATIBILITY_ARGUMENTS}
        current = dict(saved)
        current["diwa_horizon"] = 3
        with self.assertRaisesRegex(RuntimeError, "diwa_horizon"):
            validate_diwa_checkpoint(
                model,
                model.state_dict(),
                checkpoint_arguments=saved,
                current_arguments=current,
            )

    def test_frozen_base_signature_accepts_match_and_rejects_drift(self):
        model = _SmallModel()
        model.backbone.requires_grad_(False)
        signature = frozen_parameter_signature(model)
        filtered_state = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("backbone.")
        }
        validate_diwa_checkpoint(
            model,
            filtered_state,
            checkpoint_frozen_signature=signature,
        )
        with torch.no_grad():
            model.backbone.weight.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "frozen-base signature"):
            validate_diwa_checkpoint(
                model,
                filtered_state,
                checkpoint_frozen_signature=signature,
            )

    def test_frozen_signature_supports_scalar_parameters(self):
        signature = frozen_parameter_signature(_ScalarFrozenModel())
        self.assertEqual(len(signature), 64)

    def test_signature_uses_checkpoint_coverage_during_validation(self):
        training_model = _SmallModel()
        signature = frozen_parameter_signature(training_model)
        state = training_model.state_dict()

        evaluation_model = _SmallModel()
        evaluation_model.backbone.requires_grad_(False)
        validate_diwa_checkpoint(
            evaluation_model,
            state,
            checkpoint_frozen_signature=signature,
        )

    def test_pretrained_checkpoint_requires_non_diwa_trainable_state(self):
        model = _SmallModel()
        valid_baseline = {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("backbone.")
        }
        coverage = validate_pretrained_checkpoint(model, valid_baseline)
        self.assertEqual(coverage["required_parameter_count"], 2)

        with self.assertRaisesRegex(RuntimeError, "backbone"):
            validate_pretrained_checkpoint(model, {})

        wrong_shape = dict(valid_baseline)
        wrong_shape["backbone.weight"] = torch.zeros(1, 1)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            validate_pretrained_checkpoint(model, wrong_shape)

    def test_pretrained_checkpoint_allows_only_explicit_reset_state(self):
        model = _SmallModel()
        state = {"backbone.weight": model.state_dict()["backbone.weight"]}
        validate_pretrained_checkpoint(
            model,
            state,
            allowed_missing_names={"backbone.bias"},
        )

    def test_resume_arguments_cover_optimization_schedule(self):
        saved = {name: 0 for name in RESUME_COMPATIBILITY_ARGUMENTS}
        current = dict(saved)
        validate_resume_arguments(saved, current)
        current["diwa_loss_regret"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "diwa_loss_regret"):
            validate_resume_arguments(saved, current)

    def test_phase_is_resume_only_checkpoint_metadata(self):
        self.assertNotIn("phase", CHECKPOINT_COMPATIBILITY_ARGUMENTS)
        self.assertIn("phase", RESUME_COMPATIBILITY_ARGUMENTS)
        saved = {name: 0 for name in RESUME_COMPATIBILITY_ARGUMENTS}
        current = dict(saved, phase="evaluate")
        with self.assertRaisesRegex(RuntimeError, "phase"):
            validate_resume_arguments(saved, current)

    def test_rng_state_round_trip_and_rank_selection(self):
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

        random.seed(21)
        ranked = [{"rank": 0, "state": capture_rng_state()}]
        expected_ranked = random.random()
        random.seed(22)
        restore_rank_rng_state(ranked, rank=0, world_size=1)
        self.assertEqual(random.random(), expected_ranked)

    def test_world_one_rng_gather_and_atomic_checkpoint(self):
        states = gather_rng_states(rank=0, world_size=1)
        self.assertEqual([item["rank"] for item in states], [0])

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = f"{directory}/checkpoint.pt"
            atomic_torch_save({"rng_states": states}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["rng_states"][0]["rank"], 0)


if __name__ == "__main__":
    unittest.main()
