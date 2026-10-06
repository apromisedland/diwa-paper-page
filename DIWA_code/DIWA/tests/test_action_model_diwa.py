import unittest

import torch

from models.action_model.action_model import ActionModel, ActionModelFM


class DecisionResponseTest(unittest.TestCase):
    def _assert_reusable_randomness(self, model):
        model.eval()
        actions = torch.randn(2, 2, 7)
        conditions = torch.randn(2, 2, 32)
        with torch.no_grad():
            first, noise, timestep = model.decision_response(actions, conditions)
            second, _, _ = model.decision_response(
                actions,
                conditions,
                noise=noise,
                timestep=timestep,
            )
        self.assertTrue(torch.equal(first, second))

        mean, variance, noise_bank, timestep_bank = model.decision_moments(
            actions, conditions, num_probes=3
        )
        repeated_mean, repeated_variance, _, _ = model.decision_moments(
            actions,
            conditions,
            num_probes=3,
            noise=noise_bank,
            timestep=timestep_bank,
        )
        self.assertEqual(mean.shape, actions.shape)
        self.assertEqual(variance.shape, actions.shape)
        self.assertTrue(torch.equal(mean, repeated_mean))
        self.assertTrue(torch.equal(variance, repeated_variance))

    def test_diffusion_response_reuses_noise_and_timestep(self):
        model = ActionModel(
            token_size=32,
            model_type="DiT-S",
            in_channels=7,
            future_action_window_size=1,
            past_action_window_size=0,
            diffusion_steps=10,
        )
        self._assert_reusable_randomness(model)

    def test_flow_response_reuses_noise_and_timestep(self):
        model = ActionModelFM(
            token_size=32,
            model_type="DiT-S",
            in_channels=7,
            future_action_window_size=1,
            past_action_window_size=0,
            diffusion_steps=10,
        )
        self._assert_reusable_randomness(model)


if __name__ == "__main__":
    unittest.main()
