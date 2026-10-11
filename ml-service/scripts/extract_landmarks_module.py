"""Extract one-hand MediaPipe landmarks from a recorded video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

LANDMARK_COUNT = 21
COORDINATE_COUNT = 3


def _legacy_detector() -> Any | None:
    """Create the legacy detector when an older MediaPipe exposes it."""

    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "hands"):
        return None
    return solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


def _tasks_detector(model_path: Path) -> Any:
    """Create the current Tasks API detector used by recent MediaPipe builds."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe hand-landmarker model does not exist: {model_path}"
        )
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _points_from_result(result: Any) -> np.ndarray:
    hands = getattr(result, "multi_hand_landmarks", None)
    if hands is None:
        hands = getattr(result, "hand_landmarks", None)
    if not hands:
        return np.zeros((LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)

    first = hands[0]
    points = getattr(first, "landmark", first)
    if len(points) != LANDMARK_COUNT:
        raise RuntimeError(f"MediaPipe returned {len(points)} landmarks; expected 21")
    return np.asarray([[point.x, point.y, point.z] for point in points], dtype=np.float32)


def _close_detector(detector: Any) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        close()


def extract_landmarks(video_path: str | Path, model_path: str | Path | None = None) -> np.ndarray:
    """Return recorded-video landmarks with shape ``(frames, 21, 3)``.

    Frames without a detected hand are represented by 21 all-zero points.  A
    readable video containing no frames returns an empty ``(0, 21, 3)`` array.
    Both the video capture and MediaPipe detector are released on every path.
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"input video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open input video: {path}")

    default_model = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
    detector = None
    sequence: list[np.ndarray] = []
    try:
        detector = _legacy_detector()
        if detector is None:
            detector = _tasks_detector(Path(model_path) if model_path else default_model)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", 0) == 0:
                raise ValueError("input video contains an empty frame")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hasattr(detector, "process"):
                result = detector.process(rgb)
            else:
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            sequence.append(_points_from_result(result))
    finally:
        capture.release()
        if detector is not None:
            _close_detector(detector)

    if not sequence:
        return np.empty((0, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32)
    return np.stack(sequence).astype(np.float32, copy=False)
