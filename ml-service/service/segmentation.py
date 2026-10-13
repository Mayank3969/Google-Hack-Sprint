"""Reference pause-based segmentation for continuous landmark recordings.

The browser will eventually run an equivalent implementation beside camera
capture. This module uses only NumPy so ML tests can replay
recorded ``(T, 110, 3)`` landmark arrays before frontend integration exists.

INTEGRATION_WARNING (B9):
    This is an offline reference, not browser code.  The frontend must preserve
    the same frame rate, landmark ordering, state transitions, and segment
    limits.  If browser landmark extraction changes the representation, update
    the shared contract and replay fixtures before changing thresholds here.
    Fixed-rate input is required: resample irregular camera frames first. Pure
    static signs and long internal holds are ambiguous; thresholds are not yet
    validated against real signers. Synthetic tests only prove state mechanics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np


RAW_SHAPE = (110, 3)


def resample_fixed_rate(
    frames: Any,
    *,
    source_fps: float,
    target_fps: int = 15,
) -> np.ndarray:
    """Select source frames at deterministic target-rate timestamps.

    The camera/browser contract samples at a fixed target rate before the
    segmenter. This helper is the offline equivalent for decoded videos. It
    uses nearest source-frame timestamps, removes duplicate selections, and
    preserves chronological order. Model preprocessing still performs its own
    60-step sequence normalization after segmentation.
    """
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    if type(target_fps) is not int or target_fps < 1:
        raise ValueError("target_fps must be a positive integer")
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != RAW_SHAPE:
        raise ValueError("frames must have shape (T, 110, 3)")
    if not len(values):
        return values.copy()
    duration = (len(values) - 1) / float(source_fps)
    # Include the final decoded frame so a terminal pose is not silently lost.
    count = int(np.ceil(duration * target_fps - 1e-9)) + 1
    target_times = np.arange(count, dtype=np.float64) / float(target_fps)
    source_indices = np.rint(target_times * float(source_fps)).astype(np.int64)
    source_indices = np.clip(source_indices, 0, len(values) - 1)
    source_indices = np.asarray(
        [index for position, index in enumerate(source_indices)
         if position == 0 or index != source_indices[position - 1]],
        dtype=np.int64,
    )
    return values[source_indices].astype(np.float32, copy=False)


class SegmenterState(str, Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    WAITING_FOR_SIGN = "WAITING_FOR_SIGN"
    ACTIVE_SIGN = "ACTIVE_SIGN"
    PAUSE_CANDIDATE = "PAUSE_CANDIDATE"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class SegmentConfig:
    target_fps: int = 15
    pre_roll_frames: int = 3
    post_roll_frames: int = 3
    calibration_frames: int = 12
    onset_frames: int = 2
    pause_frames: int = 8
    min_segment_frames: int = 6
    max_segment_frames: int = 60
    max_segments: int = 10
    max_capture_seconds: int = 30
    tracking_grace_frames: int = 4
    fixed_pause_threshold: float = 0.04
    fixed_active_threshold: float = 0.08
    adaptive_thresholds: bool = True

    def __post_init__(self):
        if type(self.target_fps) is not int or self.target_fps < 1:
            raise ValueError("target_fps must be positive")
        for name in (
            "pre_roll_frames",
            "post_roll_frames",
            "calibration_frames",
            "onset_frames",
            "pause_frames",
            "min_segment_frames",
            "max_segment_frames",
            "max_segments",
            "max_capture_seconds",
            "tracking_grace_frames",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_segment_frames > self.max_segment_frames:
            raise ValueError("min_segment_frames cannot exceed max_segment_frames")
        if self.post_roll_frames > self.pause_frames:
            raise ValueError("post_roll_frames cannot exceed pause_frames")
        if not np.isfinite([self.fixed_pause_threshold, self.fixed_active_threshold]).all():
            raise ValueError("thresholds must be finite")
        if not 0.0 <= self.fixed_pause_threshold < self.fixed_active_threshold:
            raise ValueError("pause threshold must be below active threshold")


@dataclass(frozen=True)
class SignSegment:
    index: int
    frames: np.ndarray
    start_frame: int
    end_frame: int
    status: str = "ready"

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def as_dict(self) -> dict[str, int | str]:
        return {
            "segment_index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "status": self.status,
        }


def _validate_frame(frame: Any) -> np.ndarray:
    try:
        value = np.asarray(frame, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frame must be a numeric (110, 3) array") from exc
    if value.shape != RAW_SHAPE:
        raise ValueError("frame must have shape (110, 3)")
    if np.isinf(value).any():
        raise ValueError("frame must not contain infinity")
    return value.copy()


def _point_valid(frame: np.ndarray) -> np.ndarray:
    return np.isfinite(frame).all(axis=-1) & np.any(np.abs(frame) > 0.0, axis=-1)


def _region_motion(previous: np.ndarray, current: np.ndarray, start: int, end: int) -> float:
    prev_valid = _point_valid(previous)[start:end]
    curr_valid = _point_valid(current)[start:end]
    valid = prev_valid & curr_valid
    if not valid.any():
        return 0.0
    old = previous[start:end, :2][valid]
    new = current[start:end, :2][valid]
    displacement = np.linalg.norm(new - old, axis=-1)
    current_points = current[start:end, :2][curr_valid]
    if len(current_points) >= 2:
        span = current_points.max(axis=0) - current_points.min(axis=0)
        scale = max(float(np.linalg.norm(span)), 0.03)
    else:
        scale = 1.0
    # A percentile alone can miss a sign moving only one or two fingertips.
    # The onset confirmation window filters isolated jitter spikes instead.
    return float(max(np.percentile(displacement, 75), 0.5 * displacement.max()) / scale)


def motion_score(previous: Any, current: Any) -> float:
    """Return normalized motion for hand-led pause detection.

    Face landmarks are retained for classification, but head-only movement
    cannot start or prolong a sign. Missing observations are not motion.
    """

    old = _validate_frame(previous)
    new = _validate_frame(current)
    return motion_components(old, new)["overall"]


def motion_components(previous: Any, current: Any) -> dict[str, float]:
    """Return left-hand, right-hand, face and weighted overall motion.

    The face is diagnostic and contributes only 35% to ``overall``; hand
    movement remains the sign-start signal. This lets tests and reports expose
    face motion without allowing head motion alone to create a sign candidate.
    """
    old = _validate_frame(previous)
    new = _validate_frame(current)
    left = _region_motion(old, new, 0, 21)
    right = _region_motion(old, new, 21, 42)
    face = _region_motion(old, new, 42, 110)
    hand = max(left, right)
    return {
        "left_hand": left,
        "right_hand": right,
        "hand": hand,
        "face": face,
        "overall": max(hand, 0.35 * face),
    }


def calibrate_motion_thresholds(
    scores: Iterable[float],
    *,
    fixed_pause_threshold: float = 0.04,
    fixed_active_threshold: float = 0.08,
) -> tuple[float, float]:
    """Calculate adaptive pause/active thresholds from quiet calibration scores.

    Uses the documented robust estimator: ``median + 3*MAD`` noise, pause is
    ``max(2.5*noise, 0.04)`` clipped to ``[0.04, 0.08]``, and active is a
    hysteretic threshold at least 0.02 above pause, capped at 0.12. Fewer than
    six finite scores returns the fixed fallback.
    """
    values = np.asarray(list(scores), dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) < 6:
        return float(fixed_pause_threshold), float(fixed_active_threshold)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    noise = median + 3.0 * mad
    pause = float(np.clip(max(2.5 * noise, fixed_pause_threshold), fixed_pause_threshold, 0.08))
    active = float(min(max(1.5 * pause, pause + 0.02), 0.12))
    return pause, active


class SignSegmenter:
    """Stateful fixed-rate segmenter with adaptive pause thresholds."""

    def __init__(self, config: SegmentConfig | None = None):
        self.config = config or SegmentConfig()
        self.reset()

    def reset(self) -> None:
        self.state = SegmenterState.IDLE
        self._frame_number = -1
        self._previous: np.ndarray | None = None
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self.config.pre_roll_frames)
        self._current: list[np.ndarray] = []
        self._current_start = 0
        self._low_motion = 0
        self._tracking_loss = 0
        self._onset = 0
        self._calibration_count = 0
        self._calibration_scores: list[float] = []
        self._pause_threshold = self.config.fixed_pause_threshold
        self._active_threshold = self.config.fixed_active_threshold
        self._segments: list[SignSegment] = []
        self._current_status = "ready"

    @property
    def pause_threshold(self) -> float:
        return self._pause_threshold

    @property
    def active_threshold(self) -> float:
        return self._active_threshold

    @property
    def segments(self) -> list[SignSegment]:
        return list(self._segments)

    def _calibrate(self) -> None:
        if not self.config.adaptive_thresholds:
            return
        self._pause_threshold, self._active_threshold = calibrate_motion_thresholds(
            self._calibration_scores,
            fixed_pause_threshold=self.config.fixed_pause_threshold,
            fixed_active_threshold=self.config.fixed_active_threshold,
        )

    def _commit(self, status: str = "ready") -> None:
        if not self._current:
            return
        if len(self._segments) >= self.config.max_segments:
            raise ValueError("recording exceeds maximum segment count")
        trim = max(0, self._low_motion - self.config.post_roll_frames)
        end_frame = self._frame_number - trim
        if trim:
            self._current = self._current[:-trim]
        frame_array = np.stack(self._current).astype(np.float32, copy=False)
        status = self._current_status if self._current_status != "ready" else status
        # A detector-loss tail followed by a confirmed quiet hold is a normal
        # pause boundary in camera recordings. The quiet confirmation is the
        # evidence that lets us downgrade the transient loss for this segment;
        # finish() without that confirmation still reports tracking_lost.
        if status == "tracking_lost" and self._low_motion >= self.config.pause_frames:
            status = "ready"
        if frame_array.shape[0] < self.config.min_segment_frames and status == "ready":
            status = "partial"
        if frame_array.shape[0] > self.config.max_segment_frames and status == "ready":
            status = "overflow"
        self._segments.append(
            SignSegment(
                index=len(self._segments),
                frames=frame_array,
                start_frame=self._current_start,
                end_frame=end_frame,
                status=status,
            )
        )
        self._current = []
        self._low_motion = 0
        self._tracking_loss = 0
        self._onset = 0
        self._current_status = "ready"
        self._pre_roll.clear()
        self.state = SegmenterState.WAITING_FOR_SIGN

    def push(self, frame: Any) -> None:
        """Consume one fixed-rate frame."""

        if self.state == SegmenterState.FINISHED:
            raise RuntimeError("cannot push frames after finish; call reset first")
        value = _validate_frame(frame)
        if self._frame_number + 1 >= self.config.target_fps * self.config.max_capture_seconds:
            raise ValueError("recording exceeds maximum capture duration")
        self._frame_number += 1
        valid = bool(_point_valid(value)[:42].any())
        score = None
        hand_score = None
        if self._previous is not None and valid:
            old_valid = _point_valid(self._previous)[:42]
            new_valid = _point_valid(value)[:42]
            # A detector disappearing/reappearing must not count as quiet time.
            if np.array_equal(old_valid, new_valid) and old_valid.any():
                components = motion_components(self._previous, value)
                score = components["overall"]
                hand_score = components["hand"]
        self._previous = value
        self._pre_roll.append(value)

        if self.state == SegmenterState.IDLE:
            self.state = SegmenterState.CALIBRATING

        if self.state == SegmenterState.CALIBRATING:
            if hand_score is not None and hand_score < self.config.fixed_pause_threshold:
                self._calibration_scores.append(hand_score)
            self._calibration_count += 1
            if self._calibration_count >= self.config.calibration_frames:
                self._calibrate()
                self.state = SegmenterState.WAITING_FOR_SIGN
            # Capture early signing using fixed thresholds instead of throwing
            # away the first 0.8 seconds during calibration.
            elif hand_score is None or hand_score < self._active_threshold:
                self._onset = 0
                return

        if self.state in (SegmenterState.WAITING_FOR_SIGN, SegmenterState.CALIBRATING):
            if not valid or score is None:
                self._onset = 0
                return
            if hand_score is not None and hand_score >= self._active_threshold:
                self._onset += 1
                if self._onset >= self.config.onset_frames:
                    if len(self._segments) >= self.config.max_segments:
                        raise ValueError("recording exceeds maximum segment count")
                    self._current = list(self._pre_roll)
                    self._current_start = max(0, self._frame_number - len(self._current) + 1)
                    self.state = SegmenterState.ACTIVE_SIGN
                    self._low_motion = 0
                    self._tracking_loss = 0
                    self._current_status = "ready"
            else:
                self._onset = 0
            return

        if self.state in (SegmenterState.ACTIVE_SIGN, SegmenterState.PAUSE_CANDIDATE):
            # Bounded by max segment + pause lookahead, even for a long gesture.
            if len(self._current) < self.config.max_segment_frames + self.config.pause_frames:
                self._current.append(value)
            else:
                self._current_status = "overflow"
            if not valid:
                self._tracking_loss += 1
                self._low_motion = 0
                if self._tracking_loss > self.config.tracking_grace_frames:
                    self._current_status = "tracking_lost"
                # Keep this candidate open until a real pause/Stop. Never turn
                # tracking recovery into another word from the same gesture.
                return
            self._tracking_loss = 0
            if hand_score is not None and hand_score < self._pause_threshold:
                self._low_motion += 1
                self.state = SegmenterState.PAUSE_CANDIDATE
                if self._low_motion >= self.config.pause_frames:
                    self._commit()
            else:
                self._low_motion = 0
                self.state = SegmenterState.ACTIVE_SIGN

    def finish(self) -> list[SignSegment]:
        """Finalize the active segment and return all committed segments."""

        if self.state in (SegmenterState.ACTIVE_SIGN, SegmenterState.PAUSE_CANDIDATE):
            if self._current_status == "tracking_lost" and self._current:
                tail = self._current[-self.config.tracking_grace_frames:]
                tail_has_no_hand = all(not _point_valid(frame)[:42].any() for frame in tail)
                earlier_has_hand = any(_point_valid(frame)[:42].any() for frame in self._current[:-len(tail)] or [])
                # Stop is an explicit turn boundary. If loss is only a terminal
                # tail after a usable sign, keep the candidate for prediction;
                # mid-sign loss/recovery remains tracking_lost.
                if tail_has_no_hand and earlier_has_hand:
                    self._current_status = "ready"
            self._commit()
        self.state = SegmenterState.FINISHED
        return self.segments


def segment_recording(
    frames: Iterable[Any],
    *,
    config: SegmentConfig | None = None,
) -> list[SignSegment]:
    """Convenience function for replay tests and offline evaluation."""

    segmenter = SignSegmenter(config)
    for frame in frames:
        segmenter.push(frame)
    return segmenter.finish()
