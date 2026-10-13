"""T6/T9 derived checks plus opt-in real-video integration.

The synthetic tests prove sampling, motion components, threshold math and
missing-data status rules. The opt-in test runs the actual three downloaded
videos with MediaPipe and the committed Keras model when
``SIGNBRIDGE_RUN_REAL_VIDEO=1``; its purpose is evidence collection, not a
hardcoded expectation that cross-signer accuracy must be perfect.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_capture import evaluate, _missing_variant, MISSING_CONDITIONS
from service.segmentation import calibrate_motion_thresholds, motion_components, resample_fixed_rate


def frame(hand_offset=0.0, face_offset=0.0):
    value = np.zeros((110, 3), dtype=np.float32)
    value[:21, 0] = np.linspace(0.3, 0.4, 21) + hand_offset
    value[:21, 1] = np.linspace(0.3, 0.4, 21)
    value[42:, 0] = 0.5 + face_offset
    value[42:, 1] = 0.5
    return value


def test_fixed_rate_sampling_25_to_15_is_monotonic_and_deterministic():
    source = np.stack([frame(float(i)) for i in range(25)])
    sampled = resample_fixed_rate(source, source_fps=25.0, target_fps=15)
    assert sampled.shape[0] == 16
    np.testing.assert_array_equal(sampled, resample_fixed_rate(source, source_fps=25.0, target_fps=15))
    assert np.all(np.diff(sampled[:, 0, 0]) >= 0)
    np.testing.assert_array_equal(sampled[0], source[0])
    np.testing.assert_array_equal(sampled[-1], source[-1])


def test_motion_report_separates_hands_face_and_weighted_overall():
    first = frame()
    hand = frame(hand_offset=.2)
    face = frame(face_offset=.2)
    hand_values = motion_components(first, hand)
    face_values = motion_components(first, face)
    assert hand_values["hand"] > 0
    assert hand_values["face"] == 0
    assert face_values["hand"] == 0
    assert face_values["face"] > 0
    assert face_values["overall"] == pytest.approx(face_values["face"] * .35)


def test_adaptive_calibration_uses_robust_noise_and_fixed_fallback():
    assert calibrate_motion_thresholds([.01, .02]) == (.04, .08)
    pause, active = calibrate_motion_thresholds([.01, .011, .009, .01, .012, .01, .2])
    assert .04 <= pause <= .08
    assert active > pause
    assert active <= .12


@pytest.mark.parametrize("condition", MISSING_CONDITIONS)
def test_missing_data_corruptions_are_finite_and_shape_preserving(condition):
    raw = np.stack([frame(i * .01) for i in range(20)])
    corrupted = _missing_variant(raw, condition)
    assert corrupted.shape == raw.shape
    assert np.isfinite(corrupted).all()
    if condition == "all_landmarks_absent":
        assert not corrupted.any()
    if condition == "missing_hand_0":
        assert not corrupted[:, :21].any() and corrupted[:, 21:].any()


@pytest.mark.real_video
@pytest.mark.integration
def test_real_three_video_pipeline_opt_in(tmp_path):
    if os.getenv("SIGNBRIDGE_RUN_REAL_VIDEO") != "1":
        pytest.skip("set SIGNBRIDGE_RUN_REAL_VIDEO=1 for MediaPipe/Keras video evidence")
    root = Path(__file__).resolve().parents[2]
    video_dir = root / "tests" / "Videos"
    face_model = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
    if not video_dir.is_dir() or not face_model.is_file():
        pytest.skip("real video fixtures or face_landmarker.task are unavailable")
    report = evaluate(video_dir, face_model, target_fps=15)
    assert report["fixture_count"] == 3
    assert report["pair_count"] == 6
    assert all(item["coverage"]["finite"] for item in report["single_videos"])
    assert all(item["missing_data"]["all_absent_rejected"] for item in report["single_videos"])
    assert all(len(item["expected_sequence"]) == 2 for item in report["pair_tests"])
    output = tmp_path / "capture_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
