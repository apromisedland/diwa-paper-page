import numpy as np
import pytest

from scripts.MULTI_DATASET.DIWA.robocasa_adapter import canonical_action as robocasa_canonical
from scripts.MULTI_DATASET.DIWA.robotwin_adapter import (
    _pose_position,
    canonical_action as robotwin_canonical,
    native_action as robotwin_native,
)
from scripts.MULTI_DATASET.DIWA.lerobot_dataset_adapter import (
    canonical_action as lerobot_canonical,
    canonical_state as lerobot_state,
    select_camera_keys,
)
from scripts.MULTI_DATASET.DIWA.run_exported_smoke import validate_batch
from scripts.MULTI_DATASET.DIWA.robocasa_ood_adapter import (
    VARIANTS,
    apply_variant,
)
from scripts.MULTI_DATASET.DIWA.measured_supervision import (
    measure_candidate_rollouts,
    validate_measured_candidate_q,
)
from scripts.MULTI_DATASET.DIWA.run_offline_dataset_suite import DATASETS


def smoke_payload(action_dim, continuous_dim, state_arm_dim, state_gripper_dim):
    state_dim = state_arm_dim + state_gripper_dim
    return {
        "primary": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        "state": np.zeros((2, 3, state_dim), dtype=np.float32),
        "actions": np.zeros((2, 2, 2, action_dim), dtype=np.float32),
        "rewards": np.zeros((2, 2), dtype=np.float32),
        "dones": np.zeros((2, 2), dtype=bool),
        "progress": np.zeros((2, 2), dtype=np.float32),
        "candidate_actions": np.zeros(
            (2, 2, 4, 2, action_dim), dtype=np.float32
        ),
        "candidate_q_values": np.zeros((2, 2, 4), dtype=np.float32),
        "candidate_rule_ids": np.asarray(
            (
                "demonstration",
                "invert_gripper",
                "invert_continuous_action",
                "suppress_translation",
            )
        ),
        "dataset": np.asarray("smoke"),
        "task": np.asarray("task"),
        "language": np.asarray("perform the task"),
        "continuous_action_dim": np.asarray(continuous_dim),
        "state_arm_dim": np.asarray(state_arm_dim),
        "gripper_width": np.asarray(state_gripper_dim > 1),
    }


@pytest.mark.parametrize(
    "action_dim,continuous_dim,state_arm_dim,state_gripper_dim",
    [(7, 6, 6, 1), (12, 10, 14, 2), (14, 12, 12, 2)],
)
def test_export_contract_supports_all_simulator_dimensions(
    tmp_path, action_dim, continuous_dim, state_arm_dim, state_gripper_dim
):
    path = tmp_path / "batch.npz"
    np.savez_compressed(
        path,
        **smoke_payload(
            action_dim, continuous_dim, state_arm_dim, state_gripper_dim
        ),
    )
    with np.load(path, allow_pickle=False) as batch:
        contract = validate_batch(batch)
    assert contract["action_dim"] == action_dim
    assert contract["continuous_action_dim"] == continuous_dim
    assert contract["state_gripper_dim"] == state_gripper_dim


def test_export_contract_rejects_invalid_action_split(tmp_path):
    payload = smoke_payload(7, 7, 6, 1)
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as batch:
        with pytest.raises(ValueError, match="continuous action split"):
            validate_batch(batch)


def test_export_contract_rejects_missing_or_reordered_candidate_rules(tmp_path):
    payload = smoke_payload(7, 6, 6, 1)
    payload.pop("candidate_rule_ids")
    missing_path = tmp_path / "missing_rules.npz"
    np.savez_compressed(missing_path, **payload)
    with np.load(missing_path, allow_pickle=False) as batch:
        with pytest.raises(ValueError, match="candidate_rule_ids"):
            validate_batch(batch)

    payload = smoke_payload(7, 6, 6, 1)
    payload["candidate_rule_ids"] = payload["candidate_rule_ids"][::-1]
    reordered_path = tmp_path / "reordered_rules.npz"
    np.savez_compressed(reordered_path, **payload)
    with np.load(reordered_path, allow_pickle=False) as batch:
        with pytest.raises(ValueError, match="order differs"):
            validate_batch(batch)


def test_robocasa_action_places_binary_controls_last():
    native = np.arange(12, dtype=np.float32) - 6
    converted = robocasa_canonical(native)
    assert converted.shape == (12,)
    assert converted[-2:].tolist() == [0.0, 1.0]


def test_robotwin_action_mapping_round_trip():
    native = np.linspace(0.0, 1.0, 14, dtype=np.float32)
    np.testing.assert_allclose(robotwin_native(robotwin_canonical(native)), native)


def test_robotwin_pose_supports_native_lists_and_sapien_objects():
    class Pose:
        p = [1.0, 2.0, 3.0]

    np.testing.assert_allclose(_pose_position(Pose()), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        _pose_position([4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]),
        [4.0, 5.0, 6.0],
    )


def test_lerobot_oxe_contract_and_camera_selection():
    action = lerobot_canonical(
        np.asarray([2.0, -2.0, 0.0, 0.1, 0.2, 0.3, -1.0])
    )
    np.testing.assert_allclose(
        action, [1.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.0]
    )
    assert lerobot_state(np.arange(8, dtype=np.float32)).tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
    ]
    info = {
        "features": {
            "observation.images.eye_in_hand_rgb": {"dtype": "video"},
            "observation.images.agentview_rgb": {"dtype": "video"},
            "action": {"dtype": "float32"},
        }
    }
    assert select_camera_keys(info) == (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )


def test_offline_suite_covers_droid_and_all_twelve_oxe_datasets():
    assert len(DATASETS) == 13
    assert DATASETS["DROID"] == "lerobot/droid_1.0.1"
    assert sum(name.startswith("OXE_") for name in DATASETS) == 12


def test_candidate_q_is_measured_from_restored_branches():
    state = {"value": 0.0}
    candidates = np.asarray(
        [
            [[0.2], [0.2]],
            [[-0.1], [-0.1]],
            [[0.4], [0.0]],
        ],
        dtype=np.float32,
    )

    def restore():
        state["value"] = 0.0

    def step(action):
        state["value"] = float(np.clip(state["value"] + action[0], 0.0, 1.0))

    result = measure_candidate_rollouts(
        candidates,
        restore_state=restore,
        step_action=step,
        measure_progress=lambda: state["value"],
        measure_success=lambda: state["value"] >= 0.75,
    )
    assert np.isfinite(result.q_values).all()
    assert result.q_values[0] > result.q_values[1]
    diagnostics = validate_measured_candidate_q(result.q_values[None])
    assert diagnostics["maximum_spread"] > 0


def test_strict_export_contract_rejects_nan_candidate_q(tmp_path):
    payload = smoke_payload(7, 6, 6, 1)
    payload["candidate_q_values"][:] = np.nan
    payload["require_measured_candidate_q"] = np.asarray(True)
    path = tmp_path / "masked_q.npz"
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as batch:
        with pytest.raises(ValueError, match="must all be finite"):
            validate_batch(batch)


def test_ood_generators_change_environment_parameters():
    class Model:
        ngeom = 4
        mat_rgba = np.ones((3, 4), dtype=np.float64)
        geom_rgba = np.asarray(
            [[1.0, 0.5, 0.2, 1.0], [0.2, 1.0, 0.5, 1.0]] * 2
        )
        light_diffuse = np.ones((1, 3), dtype=np.float64)
        light_ambient = np.zeros((1, 3), dtype=np.float64)
        light_pos = np.zeros((1, 3), dtype=np.float64)
        geom_pos = np.zeros((4, 3), dtype=np.float64)
        geom_friction = np.ones((4, 3), dtype=np.float64)

        def geom_id2name(self, index):
            return (
                "wall_room",
                "cab_target_door",
                "cab_target_handle",
                "robot_geom",
            )[index]

    class Sim:
        model = Model()

        def forward(self):
            pass

    class Fixture:
        name = "cab_target"

    class Task:
        sim = Sim()
        fxtr = Fixture()

    class Unwrapped:
        env = Task()

    class Env:
        unwrapped = Unwrapped()

    env = Env()
    for variant in (
        "texture_shift",
        "lighting_shift",
        "background_motion",
        "contact_counterfactual",
    ):
        result = apply_variant(env, variant)
        assert result["changed"]
    assert VARIANTS == (
        "baseline",
        "texture_shift",
        "lighting_shift",
        "background_motion",
        "unseen_layout",
        "contact_counterfactual",
    )
