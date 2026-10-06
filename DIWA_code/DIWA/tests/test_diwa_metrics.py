import json
import unittest

import torch

from models.diwa.metrics import compute_diwa_metrics


class DIWAMetricsTest(unittest.TestCase):
    def test_perfect_identification_and_regret_geometry(self):
        influence = torch.tensor(
            [[0.1, 0.8, 0.2, 0.9], [0.7, 0.1, 0.6, 0.2]]
        )
        selected = influence >= 0.7
        full_actions = torch.zeros(2, 3, 7)
        sparse_actions = full_actions.clone()
        latents = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]]
        )
        q_values = torch.tensor(
            [[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]]
        )
        metrics = compute_diwa_metrics(
            influence_scores=influence,
            measured_influence=influence.clone(),
            selected_mask=selected,
            full_actions=full_actions,
            sparse_actions=sparse_actions,
            latent_states=latents,
            candidate_q_values=q_values,
            latency_ms=torch.tensor([10.0, 20.0]),
            peak_memory_mb=torch.tensor([100.0, 120.0]),
        )
        self.assertAlmostEqual(metrics["influence_spearman"], 1.0)
        self.assertAlmostEqual(metrics["influence_topk_recall"], 1.0)
        self.assertAlmostEqual(metrics["topk_action_consistency"], 1.0)
        self.assertAlmostEqual(metrics["expanded_token_ratio"], 0.375)
        self.assertAlmostEqual(metrics["peak_memory_mb"], 120.0)
        self.assertGreaterEqual(
            metrics["optimal_action_retrieval_accuracy"], 2.0 / 3.0
        )

    def test_requires_paired_optional_inputs(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            compute_diwa_metrics(
                influence_scores=torch.ones(1, 2),
                measured_influence=torch.ones(1, 2),
                selected_mask=torch.ones(1, 2, dtype=torch.bool),
                full_actions=torch.zeros(1, 1, 7),
            )

    def test_flattens_batch_time_candidate_archives(self):
        influence = torch.tensor(
            [[[0.1, 0.8, 0.2, 0.9], [0.7, 0.1, 0.6, 0.2]]]
        )
        metrics = compute_diwa_metrics(
            influence_scores=influence,
            measured_influence=influence.clone(),
            selected_mask=torch.ones_like(influence, dtype=torch.bool),
            success=torch.ones(1, 2),
        )
        self.assertEqual(metrics["influence_state_count"], 2.0)
        self.assertAlmostEqual(metrics["expanded_token_ratio"], 1.0)

    def test_zero_success_is_standard_json_null(self):
        metrics = compute_diwa_metrics(
            influence_scores=torch.ones(2, 3),
            measured_influence=torch.ones(2, 3),
            selected_mask=torch.ones(2, 3, dtype=torch.bool),
            success=torch.zeros(2),
        )
        self.assertIsNone(metrics["tokens_per_success"])
        payload = json.dumps(metrics, allow_nan=False)
        self.assertIn('"tokens_per_success": null', payload)

    def test_ood_populations_are_reported_without_retention_ratio(self):
        metrics = compute_diwa_metrics(
            influence_scores=torch.ones(2, 3),
            measured_influence=torch.ones(2, 3),
            selected_mask=torch.ones(2, 3, dtype=torch.bool),
            standard_success=torch.tensor([1, 0]),
            ood_success=torch.tensor([1, 1, 0]),
        )
        self.assertEqual(metrics["standard_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["ood_success_rate"], 2.0 / 3.0)
        self.assertNotIn("ood_performance_retention", metrics)

    def test_rejects_invalid_optional_measurements(self):
        common = {
            "influence_scores": torch.ones(1, 2),
            "measured_influence": torch.ones(1, 2),
            "selected_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_diwa_metrics(**common, success=torch.tensor([2.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_diwa_metrics(
                **common, latency_ms=torch.tensor([float("nan")])
            )


if __name__ == "__main__":
    unittest.main()
