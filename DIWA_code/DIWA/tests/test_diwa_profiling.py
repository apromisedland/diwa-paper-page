import json

import pytest

from utils.diwa_profiling import summarize_profiles, write_profile_report


def _profile(latency, memory, tokens, ratio):
    return {
        "latency_ms": latency,
        "peak_memory_mb": memory,
        "expanded_tokens": tokens,
        "selected_ratio": ratio,
    }


def test_profiles_drop_rank_local_warmup_and_aggregate(tmp_path):
    profiles = [
        _profile([100, 10, 20], [80, 81, 82], [12, 10, 11], [0.25, 0.2, 0.22]),
        _profile([200, 30, 40], [90, 91, 92], [12, 9, 8], [0.25, 0.18, 0.16]),
    ]
    summary = summarize_profiles(profiles, warmup_steps=1)
    assert summary["rank_count"] == 2
    assert summary["profile_sample_count"] == 4
    assert summary["latency_ms_mean"] == 25.0
    assert summary["peak_memory_mb"] == 92.0
    assert summary["expanded_tokens_mean"] == 9.5

    output = tmp_path / "profile.json"
    write_profile_report(output, summary, metadata={"device": "fixture"})
    payload = json.loads(output.read_text())
    assert payload["metrics"]["latency_ms_mean"] == 25.0
    assert payload["metadata"]["device"] == "fixture"


def test_profiles_reject_nonfinite_or_misaligned_samples():
    with pytest.raises(ValueError, match="finite"):
        summarize_profiles(
            [_profile([float("nan")], [1], [1], [0.1])]
        )
    with pytest.raises(ValueError, match="same number"):
        summarize_profiles([_profile([1, 2], [1], [1], [0.1])])
