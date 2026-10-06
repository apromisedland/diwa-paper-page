import unittest

import torch

from models.diwa import DIWACore, ObjectCentricTokenizer, TopKSelector
from models.diwa.core import AdaptiveTopKSelector


class TopKSelectorTest(unittest.TestCase):
    def test_selects_exact_budget(self):
        selector = TopKSelector()
        logits = torch.tensor([[0.1, 0.9, 0.2, 0.7]])
        indices, mask = selector(logits, budget_ratio=0.5)

        self.assertEqual(indices.shape, (1, 2))
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual(set(indices[0].tolist()), {1, 3})

    def test_rejects_invalid_budget(self):
        selector = TopKSelector()
        with self.assertRaises(ValueError):
            selector(torch.zeros(1, 4), budget_ratio=0.0)

    def test_adaptive_selector_packs_variable_budgets(self):
        selector = AdaptiveTopKSelector()
        logits = torch.tensor(
            [[0.1, 0.9, 0.2, 0.7], [0.8, 0.3, 0.2, 0.1]]
        )
        indices, mask, valid = selector(
            logits, torch.tensor([0.25, 0.75])
        )
        self.assertEqual(indices.shape, (2, 3))
        self.assertEqual(valid.sum(dim=-1).tolist(), [1, 3])
        self.assertEqual(mask.sum(dim=-1).tolist(), [1, 3])


class ObjectCentricTokenizerTest(unittest.TestCase):
    def test_fuses_dense_and_track_features(self):
        tokenizer = ObjectCentricTokenizer(
            hidden_dim=16,
            num_slots=3,
            sam_feature_dim=8,
            slot_iterations=2,
        )
        visual = torch.randn(2, 4, 6, 16, requires_grad=True)
        sam = torch.randn(2, 4, 5, 8)
        tracks = torch.rand(2, 4, 7, 2) * 224
        visibility = torch.ones(2, 4, 7)
        output = tokenizer(
            visual,
            sam_primary=sam,
            tracks_primary=tracks,
            visibility_primary=visibility,
        )
        self.assertEqual(output.tokens.shape, (2, 4, 3, 16))
        self.assertEqual(output.assignments.shape, (2, 4, 3, 18))
        output.tokens.square().mean().backward()
        self.assertIsNotNone(visual.grad)


class DIWACoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DIWACore(
            hidden_dim=32,
            num_heads=4,
            action_pred_steps=3,
            horizon=2,
            num_slots=4,
            world_layers=1,
            fusion_layers=1,
            counterfactual_samples=2,
        )
        self.context = torch.randn(2, 3, 6, 32)
        self.actions = torch.randn(2, 3, 3, 7)
        self.actions[..., 6:] = torch.rand_like(self.actions[..., 6:])
        self.future_targets = torch.randn(2, 3, 2, 4, 32)
        self.future_valid = torch.ones(2, 3, 2, dtype=torch.bool)

    def test_training_outputs_and_gradients(self):
        self.model.train()
        output = self.model(
            self.context,
            action_labels=self.actions,
            future_targets=self.future_targets,
            future_valid_mask=self.future_valid,
            budget_ratio=0.5,
            teacher_forcing_ratio=0.5,
        )

        self.assertEqual(output.action_features.shape, (2, 3, 3, 32))
        self.assertEqual(output.world_tokens.shape, (6, 8, 32))
        self.assertEqual(output.selected_indices.shape, (2, 3, 4))
        self.assertEqual(
            output.counterfactual_action_features.shape,
            (6, 2, 3, 32),
        )

        loss = sum(output.aux_losses.values())
        loss.backward()
        influence_gradient = self.model.influence_estimator[-1].weight.grad
        self.assertIsNotNone(influence_gradient)
        self.assertTrue(torch.isfinite(influence_gradient).all())

    def test_eval_expands_only_selected_tokens(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.context,
                budget_ratio=0.25,
                compute_counterfactual=False,
            )

        # 2 horizons * 4 slots * 25% = 2 expanded tokens.
        self.assertEqual(output.world_tokens.shape, (6, 2, 32))
        self.assertEqual(output.selected_indices.shape, (2, 3, 2))
        self.assertIsNone(output.counterfactual_action_features)

    def test_influence_ablation_uses_uniform_scores(self):
        self.model.train()
        self.model.enable_influence_estimator = False
        output = self.model(
            self.context,
            action_labels=self.actions,
            future_targets=self.future_targets,
            future_valid_mask=self.future_valid,
            budget_ratio=1.0,
            force_dense_budget=True,
        )
        self.assertTrue(
            torch.equal(
                output.influence_logits,
                torch.zeros_like(output.influence_logits),
            )
        )
        (
            output.aux_losses["proposal"]
            + output.aux_losses["future"]
        ).backward()
        self.assertIsNone(self.model.influence_estimator[-1].weight.grad)

    def test_counterfactual_swap_ablation_keeps_mask_interventions(self):
        metadata = {
            "task_ids": torch.tensor([3, 3]),
            "episode_ids": torch.tensor([10, 11]),
        }
        self.model.train()
        enabled = self.model(
            self.context,
            action_labels=self.actions,
            budget_ratio=0.5,
            **metadata,
        )
        self.assertTrue(enabled.counterfactual_swap_mask.any())

        self.model.enable_counterfactual_swaps = False
        disabled = self.model(
            self.context,
            action_labels=self.actions,
            budget_ratio=0.5,
            **metadata,
        )
        self.assertFalse(disabled.counterfactual_swap_mask.any())
        self.assertIsNotNone(disabled.counterfactual_action_features)

    def test_invalid_future_positions_do_not_break_loss(self):
        self.model.train()
        valid = torch.zeros_like(self.future_valid)
        output = self.model(
            self.context,
            action_labels=self.actions,
            future_targets=self.future_targets,
            future_valid_mask=valid,
            budget_ratio=1.0,
        )
        self.assertEqual(output.aux_losses["future"].item(), 0.0)

    def test_real_critic_regret_and_progress_losses(self):
        model = DIWACore(
            hidden_dim=32,
            num_heads=4,
            action_pred_steps=3,
            horizon=2,
            num_slots=4,
            world_layers=1,
            fusion_layers=1,
            counterfactual_samples=2,
            regret_candidates=6,
        )
        model.train()
        context = torch.randn(2, 3, 6, 32)
        objects = torch.randn(2, 3, 4, 32)
        actions = torch.rand(2, 3, 3, 7)
        actions[..., :6] = actions[..., :6] * 2 - 1
        rewards = torch.zeros(2, 3)
        rewards[:, -1] = 1
        dones = torch.zeros(2, 3, dtype=torch.bool)
        dones[:, -1] = True
        progress = torch.tensor(
            [[0.1, 0.5, 1.0], [0.2, 0.6, 1.0]]
        )
        valid = torch.ones(2, 3, dtype=torch.bool)
        candidate_actions = torch.rand(2, 3, 5, 3, 7)
        candidate_actions[..., :6] = (
            candidate_actions[..., :6] * 2 - 1
        )
        candidate_q_values = torch.tensor(
            [
                [[1.0, 0.5, 0.0, -0.5, -1.0]] * 3,
                [[0.0, 1.0, 0.5, -0.5, -1.0]] * 3,
            ]
        )
        output = model(
            context,
            current_object_tokens=objects,
            action_labels=actions,
            rewards=rewards,
            dones=dones,
            progress_targets=progress,
            supervision_valid_mask=valid,
            candidate_actions=candidate_actions,
            candidate_q_values=candidate_q_values,
            task_ids=torch.tensor([4, 4]),
            episode_ids=torch.tensor([10, 11]),
            budget_ratio=0.5,
        )
        for name in ("critic", "progress", "regret", "contrastive"):
            self.assertTrue(torch.isfinite(output.aux_losses[name]))
        self.assertGreater(output.aux_losses["critic"].item(), 0.0)
        self.assertGreater(
            output.aux_losses["candidate_q_diagnostic"].item(), 0.0
        )
        self.assertGreater(output.aux_losses["progress"].item(), 0.0)
        self.assertGreater(output.aux_losses["regret"].item(), 0.0)
        self.assertGreater(output.aux_losses["contrastive"].item(), 0.0)
        self.assertIsNotNone(output.counterfactual_valid_mask)
        (
            output.aux_losses["critic"]
            + output.aux_losses["progress"]
            + output.aux_losses["regret"]
            + output.aux_losses["contrastive"]
        ).backward()
        self.assertIsNotNone(model.critic.q1[-1].weight.grad)
        self.assertGreater(model.critic.q1[-1].weight.grad.abs().sum(), 0.0)

    def test_candidate_q_contract_is_checked(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.model(
                self.context,
                action_labels=self.actions,
                rewards=torch.zeros(2, 3),
                dones=torch.zeros(2, 3, dtype=torch.bool),
                progress_targets=torch.zeros(2, 3),
                supervision_valid_mask=torch.ones(
                    2, 3, dtype=torch.bool
                ),
                candidate_actions=torch.zeros(2, 3, 4, 3, 7),
            )

    def test_adaptive_eval_reports_actual_ratio(self):
        model = DIWACore(
            hidden_dim=32,
            num_heads=4,
            action_pred_steps=2,
            horizon=2,
            num_slots=4,
            world_layers=1,
            fusion_layers=1,
            adaptive_budget=True,
            minimum_budget_ratio=0.125,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 2, 5, 32), budget_ratio=0.75)
        actual = output.selected_mask.float().mean(dim=(-1, -2))
        expected = output.selected_valid_mask.float().sum(dim=-1) / 8
        self.assertTrue(torch.equal(actual, expected))

    def test_warmup_can_force_dense_adaptive_budget(self):
        model = DIWACore(
            hidden_dim=32,
            num_heads=4,
            action_pred_steps=2,
            horizon=2,
            num_slots=4,
            world_layers=1,
            fusion_layers=1,
            adaptive_budget=True,
        )
        model.train()
        output = model(
            torch.randn(2, 2, 5, 32),
            budget_ratio=1.0,
            force_dense_budget=True,
        )
        self.assertTrue(output.selected_mask.all())
        self.assertTrue(output.selected_valid_mask.all())
        self.assertTrue(
            torch.equal(
                output.adaptive_budget_ratio,
                torch.ones_like(output.adaptive_budget_ratio),
            )
        )

    def test_dual_arm_action_contract_trains_and_evaluates(self):
        model = DIWACore(
            hidden_dim=32,
            num_heads=4,
            action_pred_steps=2,
            action_dim=14,
            continuous_action_dim=12,
            horizon=2,
            num_slots=4,
            world_layers=1,
            fusion_layers=1,
            counterfactual_samples=2,
        )
        context = torch.randn(2, 2, 5, 32)
        actions = torch.rand(2, 2, 2, 14)
        actions[..., :12] = actions[..., :12] * 2 - 1
        candidates = torch.rand(2, 2, 4, 2, 14)
        candidates[..., :12] = candidates[..., :12] * 2 - 1
        valid = torch.ones(2, 2, dtype=torch.bool)
        model.train()
        output = model(
            context,
            action_labels=actions,
            rewards=torch.zeros(2, 2),
            dones=torch.tensor([[False, True], [False, True]]),
            progress_targets=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            supervision_valid_mask=valid,
            candidate_actions=candidates,
            candidate_q_values=torch.randn(2, 2, 4),
            task_ids=torch.tensor([0, 0]),
            episode_ids=torch.tensor([0, 1]),
            budget_ratio=0.5,
        )
        self.assertEqual(output.proposal_actions.shape, (2, 2, 2, 14))
        loss = sum(output.aux_losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.proposal_head[-1].weight.grad)

        model.eval()
        with torch.no_grad():
            evaluated = model(context, budget_ratio=0.25)
        self.assertEqual(evaluated.proposal_actions.shape, (2, 2, 2, 14))


if __name__ == "__main__":
    unittest.main()
