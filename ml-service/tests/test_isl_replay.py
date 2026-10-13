"""B10 incremental SYNTHETIC replay tests.

INTEGRATION_WARNING: correct known outputs here come from a test double, never
from recognizing real signs. Person A supplies the annotated capture set.
"""

from scripts.evaluate_segmentation import boundary_metrics, evaluate_recording
from service.isl_inference import ISLInferenceEngine
from test_isl_inference import ModelDouble
from test_sign_segmentation import CONFIG, recording


def test_recording_to_ordered_predictions():
    model = ModelDouble()
    report = evaluate_recording(recording(3), config=CONFIG, predictor=ISLInferenceEngine(model))
    assert report["segment_count"] == 3
    assert [p["keyword"] for p in report["predictions"]] == ["sign0"] * 3
    assert model.calls[0].shape == (3, 60, 660)
    assert not report["integration_ready"]


def test_boundary_scoring_counts_false_splits_and_misses():
    detected = [{"start_frame": 1, "end_frame": 10}, {"start_frame": 2, "end_frame": 11},
                {"start_frame": 90, "end_frame": 100}]
    expected = [{"start_frame": 0, "end_frame": 10}, {"start_frame": 40, "end_frame": 50}]
    result = boundary_metrics(detected, expected)
    assert result["true_positive_boundaries"] == 1
    assert result["extra_segments"] == 2
    assert result["missed_segments"] == 1


def test_rejected_segment_never_sent_to_model():
    from dataclasses import replace
    model = ModelDouble()
    report = evaluate_recording(recording(1), config=replace(CONFIG, min_segment_frames=50),
                                predictor=ISLInferenceEngine(model))
    assert report["predictions"][0]["status"] == "partial"
    assert not model.calls
