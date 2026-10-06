import unittest

from utils.arguments_utils import get_parser, validate_diwa_args


class DIWAArgumentsTest(unittest.TestCase):
    def _args(self, *extra):
        return get_parser().parse_args(
            [
                "--save_checkpoint_path",
                "/tmp/checkpoints",
                "--phase",
                "finetune",
                "--use_diwa",
                "--sequence_length",
                "7",
                "--diwa_horizon",
                "3",
                "--window_size",
                "10",
                *extra,
            ]
        )

    def test_valid_strict_configuration(self):
        args = self._args(
            "--batch_size",
            "2",
            "--diwa_require_supervision",
            "--diwa_supervision_path",
            "/tmp/measured",
        )
        validate_diwa_args(args, training=True)

    def test_evaluation_requires_checkpoint(self):
        args = self._args()
        with self.assertRaisesRegex(ValueError, "requires resume_from_checkpoint"):
            validate_diwa_args(args, training=False)

        args = self._args("--resume_from_checkpoint", "/tmp/model.pt")
        validate_diwa_args(args, training=False)

    def test_rejects_missing_future_window(self):
        args = self._args("--window_size", "9")
        with self.assertRaisesRegex(ValueError, "window_size"):
            validate_diwa_args(args, training=True)

    def test_rejects_invalid_profile_configuration(self):
        args = self._args("--diwa_profile_warmup_steps", "-1")
        with self.assertRaisesRegex(ValueError, "profile_warmup"):
            validate_diwa_args(args, training=False)

        args = self._args("--diwa_profile_output", "/tmp/profile.json")
        with self.assertRaisesRegex(ValueError, "requires diwa_profile"):
            validate_diwa_args(args, training=False)

        args = get_parser().parse_args(
            [
                "--save_checkpoint_path",
                "/tmp/checkpoints",
                "--phase",
                "evaluate",
                "--diwa_profile",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires use_diwa"):
            validate_diwa_args(args, training=False)

    def test_rejects_invalid_model_contracts(self):
        cases = (
            (("--action_dim", "6", "--continuous_action_dim", "6"), "split"),
            (("--hidden_dim", "31", "--transformer_heads", "4"), "divisible"),
            (("--diwa_dropout", "1"), "dropout"),
            (("--diwa_critic_discount", "1.1"), "critic_discount"),
            (("--diwa_regret_candidates", "3"), "regret_candidates"),
            (("--state_gripper_dim", "2"), "categorical gripper"),
        )
        for options, pattern in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_diwa_args(self._args(*options), training=True)

    def test_rejects_invalid_training_contracts(self):
        cases = (
            (("--gradient_accumulation_steps", "0"), "gradient_accumulation"),
            (("--learning_rate", "nan"), "learning_rate"),
            (("--diwa_target_tau", "0"), "target_tau"),
            (("--diwa_budget_warmup_steps", "-1"), "budget_warmup"),
            (("--lr_scheduler", "cosine_restart"), "lr_scheduler"),
            (("--warmup_epochs", "2", "--num_epochs", "1"), "warmup_epochs"),
        )
        for options, pattern in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_diwa_args(self._args(*options), training=True)


if __name__ == "__main__":
    unittest.main()
