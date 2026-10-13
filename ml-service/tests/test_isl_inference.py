"""B4–B6 SYNTHETIC policy/batching tests (the model double is test-only).

INTEGRATION_WARNING: replace no runtime component with this double. Real-model
mechanics are tested separately; Person A's golden parity remains required.
"""

import numpy as np
import pytest

from service.isl_inference import ISLInferenceEngine, SegmentRequest
from service.isl_quality import QualityPolicy


class ModelDouble:
    version = "TEST-ONLY"
    labels = [f"sign{i}" for i in range(50)]

    def __init__(self, first=0.9, second=0.08):
        self.first, self.second = first, second
        self.calls = []

    def predict_probabilities(self, data):
        self.calls.append(data.copy())
        p = np.full((len(data), 50), (1 - self.first - self.second) / 48, dtype=np.float32)
        p[:, 0], p[:, 1] = self.first, self.second
        return p


def landmarks():
    return np.random.default_rng(31).random((12, 110, 3), dtype=np.float32)


def test_mixed_batch_preserves_slots_and_runs_only_valid_inputs_once():
    model = ModelDouble()
    engine = ISLInferenceEngine(model)
    results = engine.predict_segments([
        SegmentRequest(3, landmarks()), SegmentRequest(1, []),
        SegmentRequest(2, np.zeros((12, 110, 3))), SegmentRequest(0, landmarks()),
    ])
    assert [r.segment_index for r in results] == [0, 1, 2, 3]
    assert [r.status for r in results] == ["recognized", "invalid", "insufficient_data", "recognized"]
    assert model.calls[0].shape == (2, 60, 660)
    assert len(model.calls) == 1
    assert results[1].keyword is None
    assert results[2].confidence is None
    assert not results[0].as_dict()["integration_ready"]


def test_top_one_display_does_not_bypass_margin_rejection():
    engine = ISLInferenceEngine(ModelDouble(first=0.51, second=0.49))
    for top_k in [1, 3]:
        result = engine.predict_segment(SegmentRequest(0, landmarks()), top_k=top_k)
        assert result.status == "uncertain"
        assert result.keyword is None
        assert len(result.top_k) == top_k


def test_face_only_is_not_a_sign_and_one_hand_without_face_is_allowed():
    model = ModelDouble()
    engine = ISLInferenceEngine(model)
    face = landmarks()
    face[:, :42] = 0
    hand = landmarks()
    hand[:, 21:] = 0
    results = engine.predict_segments([SegmentRequest(0, face), SegmentRequest(1, hand)])
    assert results[0].status == "insufficient_data"
    assert results[1].status == "recognized"
    assert model.calls[0].shape[0] == 1


def test_all_rejected_does_not_infer():
    model = ModelDouble()
    result = ISLInferenceEngine(model).predict_segment(SegmentRequest(0, np.zeros((4, 110, 3))))
    assert result.status == "insufficient_data"
    assert not model.calls


def test_visible_but_perfectly_static_buffer_cannot_become_a_word():
    model = ModelDouble()
    result = ISLInferenceEngine(model).predict_segment(SegmentRequest(0, np.full((12, 110, 3), .5)))
    assert result.status == "insufficient_data"
    assert not model.calls


@pytest.mark.parametrize("kwargs", [
    {"min_confidence": 1.1}, {"min_margin": -0.1}, {"min_coverage": float("nan")},
    {"min_valid_frames": 0}, {"min_valid_frames": True},
])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        QualityPolicy(**kwargs)


def test_request_bounds():
    engine = ISLInferenceEngine(ModelDouble())
    for requests in [[], [SegmentRequest(i, []) for i in range(11)],
                     [SegmentRequest(0, []), SegmentRequest(0, [])], [SegmentRequest(True, [])]]:
        with pytest.raises(ValueError):
            engine.predict_segments(requests)


def test_low_coverage_and_low_confidence():
    sparse = np.full((12, 110, 3), np.nan, dtype=np.float32)
    sparse[:, 0] = 0.5
    engine = ISLInferenceEngine(ModelDouble(first=0.4, second=0.3))
    results = engine.predict_segments([SegmentRequest(0, sparse), SegmentRequest(1, landmarks())])
    assert [r.status for r in results] == ["tracking_lost", "uncertain"]
