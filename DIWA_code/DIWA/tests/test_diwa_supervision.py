import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.build_diwa_supervision import (
    package_episode,
    validate_episode,
)
from utils.diwa_schema import stable_candidate_rule_ids


class DIWASupervisionTest(unittest.TestCase):
    def _write_episode(self, path: Path, **overrides) -> None:
        arrays = {
            "reward": np.array([0.0, 1.0], dtype=np.float32),
            "done": np.array([False, True]),
            "progress": np.array([0.25, 1.0], dtype=np.float32),
            "candidate_actions": np.zeros(
                (2, 4, 3, 7), dtype=np.float32
            ),
            "candidate_q_values": np.array(
                [[1.0, 0.5, 0.0, -0.5], [0.0, 1.0, 0.5, -0.5]],
                dtype=np.float32,
            ),
            "candidate_rule_ids": np.asarray(stable_candidate_rule_ids(4)),
        }
        arrays.update(overrides)
        np.savez(path, **arrays)

    def test_packages_all_measured_fields_per_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "000001.npz"
            output = root / "sidecars"
            self._write_episode(source)
            self.assertEqual(package_episode(source, output), 2)
            with np.load(output / "000001" / "steps" / "0001.npz") as step:
                self.assertEqual(set(step.files), {
                    "reward",
                    "done",
                    "progress",
                    "candidate_actions",
                    "candidate_q_values",
                    "candidate_rule_ids",
                })
                self.assertEqual(step["candidate_actions"].shape, (4, 3, 7))
                self.assertEqual(step["candidate_q_values"].shape, (4,))
                self.assertEqual(
                    tuple(step["candidate_rule_ids"].tolist()),
                    stable_candidate_rule_ids(4),
                )

    def test_rejects_missing_or_invalid_measured_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.npz"
            np.savez(
                source,
                reward=np.zeros(2),
                done=np.zeros(2),
                progress=np.zeros(2),
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_episode(source)

            self._write_episode(
                source,
                progress=np.array([0.0, 1.1], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                validate_episode(source)

    def test_rejects_missing_or_reordered_candidate_rule_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.npz"
            self._write_episode(source)
            with np.load(source, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            arrays.pop("candidate_rule_ids")
            np.savez(source, **arrays)
            with self.assertRaisesRegex(ValueError, "candidate_rule_ids"):
                validate_episode(source)

            arrays["candidate_rule_ids"] = np.asarray(
                stable_candidate_rule_ids(4)[::-1]
            )
            np.savez(source, **arrays)
            with self.assertRaisesRegex(ValueError, "order differs"):
                validate_episode(source)


if __name__ == "__main__":
    unittest.main()
