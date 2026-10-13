"""B9 SYNTHETIC state-machine tests, not ISL boundary-accuracy evidence.

INTEGRATION_WARNING: Person A/frontend must add annotated real captures before
tuning/publishing thresholds. Port semantics, not just these synthetic shapes.
"""

from dataclasses import replace

import numpy as np
import pytest

from service.segmentation import SegmentConfig, SignSegmenter, segment_recording


CONFIG = SegmentConfig(calibration_frames=3, pause_frames=4, post_roll_frames=2,
                       min_segment_frames=4, adaptive_thresholds=False)


def pose(offset=0.0):
    frame = np.zeros((110, 3), dtype=np.float32)
    frame[:21, 0] = np.linspace(0.3, 0.4, 21) + offset
    frame[:21, 1] = np.linspace(0.3, 0.4, 21)
    frame[42:, :2] = 0.5
    return frame


def recording(words=2):
    frames = [pose() for _ in range(4)]
    offset = 0.0
    for _ in range(words):
        for i in range(1, 7):
            frames.append(pose(offset + i * 0.03))
        offset += 0.18
        frames.extend(pose(offset) for _ in range(6))
    return frames


def test_two_signs_have_ordered_nonoverlapping_boundaries_and_fixed_frames():
    segments = segment_recording(recording(), config=CONFIG)
    assert len(segments) == 2
    assert all(s.status == "ready" for s in segments)
    assert segments[0].end_frame < segments[1].start_frame
    for segment in segments:
        assert segment.frame_count == segment.end_frame - segment.start_frame + 1


def test_short_internal_hold_does_not_split_and_stop_finalizes():
    frames = [pose() for _ in range(4)]
    frames += [pose(i * .03) for i in range(1, 5)]
    frames += [pose(.12)] * 2
    frames += [pose(.12 + i * .03) for i in range(1, 5)]
    result = segment_recording(frames, config=CONFIG)
    assert len(result) == 1 and result[0].status == "ready"


def test_missing_hands_not_a_pause_even_with_face_visible():
    frames = recording(1)[:9]
    missing = pose()
    missing[:42] = 0
    frames += [missing.copy() for _ in range(6)]
    frames += [pose(.2 + i * .03) for i in range(4)]
    result = segment_recording(frames, config=CONFIG)
    assert len(result) == 1
    assert result[0].status == "tracking_lost"


def test_tracking_loss_followed_by_confirmed_hold_can_close_boundary():
    frames = [pose() for _ in range(4)] + [pose(i * .03) for i in range(1, 7)]
    missing = pose()
    missing[:42] = 0
    frames.extend([missing.copy() for _ in range(5)])
    frames.extend([pose(.18) for _ in range(5)])
    result = segment_recording(frames, config=CONFIG)
    assert result and result[0].status == "ready"


def test_stop_converts_only_terminal_tracking_loss_to_ready():
    frames = [pose() for _ in range(4)] + [pose(i * .03) for i in range(1, 7)]
    missing = pose()
    missing[:42] = 0
    frames.extend([missing.copy() for _ in range(5)])
    result = segment_recording(frames, config=CONFIG)
    assert result and result[0].status == "ready"


def test_overflow_is_one_rejected_segment_not_multiple_words():
    frames = [pose() for _ in range(4)] + [pose(i * .03) for i in range(1, 100)]
    segments = segment_recording(frames, config=CONFIG)
    assert len(segments) == 1 and segments[0].status == "overflow"
    assert segments[0].frame_count <= CONFIG.max_segment_frames + CONFIG.pause_frames


def test_short_candidate_is_reported_partial():
    segment = segment_recording([pose(), pose(), pose(.03), pose(.06)], config=replace(CONFIG, min_segment_frames=10))
    assert len(segment) == 1 and segment[0].status == "partial"


def test_empty_static_and_head_only_do_not_create_words():
    assert segment_recording([], config=CONFIG) == []
    assert segment_recording([pose()] * 50, config=CONFIG) == []
    frames = []
    for i in range(40):
        frame = pose()
        frame[42:, 0] += i * .03
        frames.append(frame)
    assert segment_recording(frames, config=CONFIG) == []


def test_early_sign_is_not_discarded_for_calibration():
    config = replace(CONFIG, calibration_frames=12)
    assert len(segment_recording([pose(i * .03) for i in range(8)], config=config)) == 1


def test_adaptive_calibration_and_fixed_fallback():
    segmenter = SignSegmenter(replace(CONFIG, calibration_frames=12, adaptive_thresholds=True))
    for _ in range(12):
        segmenter.push(pose())
    assert .04 <= segmenter.pause_threshold <= .08
    assert segmenter.active_threshold > segmenter.pause_threshold
    fallback = SignSegmenter(replace(CONFIG, adaptive_thresholds=True))
    for _ in range(3):
        fallback.push(np.zeros((110, 3)))
    assert fallback.active_threshold == CONFIG.fixed_active_threshold


def test_session_limits_are_explicit_and_finish_is_idempotent():
    with pytest.raises(ValueError, match="segment count"):
        segment_recording(recording(2), config=replace(CONFIG, max_segments=1))
    with pytest.raises(ValueError, match="duration"):
        segment_recording([pose()] * 16, config=replace(CONFIG, max_capture_seconds=1))
    segmenter = SignSegmenter(CONFIG)
    for frame in recording(1):
        segmenter.push(frame)
    first = segmenter.finish()
    assert segmenter.finish() == first
    with pytest.raises(RuntimeError):
        segmenter.push(pose())
    segmenter.reset()
    assert segmenter.segments == []


def test_fingertip_motion_can_start_candidate():
    frames = [pose() for _ in range(4)]
    for i in range(1, 8):
        frame = pose()
        frame[8, 0] += .1 * i
        frames.append(frame)
    assert len(segment_recording(frames, config=CONFIG)) == 1


def test_calibration_onset_requires_consecutive_moving_frames():
    config = replace(CONFIG, calibration_frames=12)
    frames = [pose(), pose(.03), pose(.03), pose(.06), pose(.06)]
    assert segment_recording(frames, config=config) == []
