import numpy as np

from data_process.collect_libero_diwa_supervision import (
    candidate_bank,
    expert_chunk,
    measure_rollouts,
)
from utils.diwa_schema import stable_candidate_rule_ids


def test_expert_chunk_pads_terminal_action():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk = expert_chunk(actions, step=2, length=3)
    assert chunk.shape == (3, 7)
    assert np.array_equal(chunk[0], actions[-1])
    assert np.array_equal(chunk[1], actions[-1])
    assert np.array_equal(chunk[2], actions[-1])


def test_candidate_rules_are_stable_and_bounded():
    expert = np.zeros((3, 7), dtype=np.float32)
    expert[:, 6] = -1.0
    candidates = candidate_bank(expert, count=6, perturbation=0.2)
    assert candidates.shape == (6, 3, 7)
    assert np.array_equal(candidates[0], expert)
    assert np.allclose(candidates[1, :, :6], 0.0)
    assert np.allclose(candidates[2, :, 0], 0.2)
    assert np.allclose(candidates[3, :, 0], -0.2)
    assert np.allclose(candidates[4, :, 1], 0.2)
    assert np.allclose(candidates[5, :, 1], -0.2)
    assert np.isin(candidates[..., 6], (-1.0, 1.0)).all()
    assert np.abs(candidates).max() <= 1.0
    assert stable_candidate_rule_ids(6) == (
        "demonstration",
        "no_motion_keep_gripper",
        "plus_x",
        "minus_x",
        "plus_y",
        "minus_y",
    )


class _BranchEnvironment:
    def __init__(self):
        self.reset_count = 0
        self.elapsed_steps = 0
        self.state = None

    def reset(self):
        self.reset_count += 1
        self.elapsed_steps = 0

    def set_init_state(self, state):
        self.state = np.asarray(state).copy()

    def _check_success(self):
        return False

    def step(self, action):
        self.elapsed_steps += 1
        # This assertion fails if a branch inherits elapsed time from a
        # previous candidate instead of receiving a complete environment reset.
        assert self.elapsed_steps <= 2
        return None, float(np.asarray(action)[0]), False, {}


def test_candidate_rollouts_reset_environment_state_per_branch():
    env = _BranchEnvironment()
    candidates = np.zeros((4, 2, 7), dtype=np.float32)
    candidates[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    q_values, reward, done, progress = measure_rollouts(
        env,
        state=np.arange(5, dtype=np.float32),
        candidates=candidates,
        discount=1.0,
        progress_reward=1.0,
    )
    assert env.reset_count == candidates.shape[0]
    assert np.allclose(q_values, np.arange(4, dtype=np.float32) * 2)
    assert reward == 0.0
    assert done is False
    assert progress == 0.0
