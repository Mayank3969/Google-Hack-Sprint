"""Extract raw two-hand plus face landmarks with MediaPipe Tasks VIDEO APIs.

Every decoded frame is retained; no normalization, mirroring, sampling, padding,
or coordinate clipping is applied here. Missing hands become zero-filled frames.
OpenCV and MediaPipe are imported only when extraction is requested.
"""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path

import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
FACE_FEATURE_VERSION = "compact-v1"
FACE_FEATURE_GROUPS = {"RIGHT_EYE": (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246), "LEFT_EYE": (263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466), "RIGHT_EYEBROW": (70,63,105,66,107), "LEFT_EYEBROW": (336,296,334,293,300), "LIPS_OUTER": (61,146,91,181,84,17,314,405,321,375), "LIPS_INNER": (78,95,88,178,87,14,317,402,318,324), "NOSE": (1,2,4,5,6,168)}
FACE_LANDMARK_INDICES = tuple(i for group in FACE_FEATURE_GROUPS.values() for i in group)
FACE_POINTS = len(FACE_LANDMARK_INDICES)
FACE_SOURCE_POINTS = 478
FEATURE_SHAPE = (2 * 21 + FACE_POINTS, 3)  # (110, 3)


def _load_dependencies():
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "landmark extraction requires OpenCV and MediaPipe; install "
            "ml-service/requirements.txt with the active Python interpreter"
        ) from exc
    return cv2, mp, python, vision


def _frame_points(result: object, face_result: object, frame_index: int) -> np.ndarray:
    try:
        hands = result.hand_landmarks
        output = np.zeros(FEATURE_SHAPE, dtype=np.float32)
        handedness = getattr(result, "handedness", [])
        for i, hand in enumerate(hands or []):
            if i >= 2 or len(hand) != 21: raise ValueError("expected at most two hands with 21 points")
            label = getattr(handedness[i][0], "category_name", "") if i < len(handedness) and handedness[i] else ""
            slot = 0 if label.lower() == "left" else 1 if label.lower() == "right" else i
            output[slot * 21:(slot + 1) * 21] = [[p.x, p.y, p.z] for p in hand]
        faces = getattr(face_result, "face_landmarks", [])
        if faces:
            if len(faces[0]) != FACE_SOURCE_POINTS: raise ValueError("face model must produce exactly 478 landmarks")
            output[42:] = [[faces[0][i].x, faces[0][i].y, faces[0][i].z] for i in FACE_LANDMARK_INDICES]
        if not np.isfinite(output).all(): raise ValueError("expected finite landmarks")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid hand landmarks at frame {frame_index}: {exc}") from exc
    return output


def extract_landmarks(video_path, *, model_path=DEFAULT_MODEL_PATH, face_model_path=None,
                      frame_start=None, frame_end=None) -> np.ndarray:
    """Return float32 ``(T, 110, 3)``: hands followed by compact face points.

    Timing uses increasing, finite, nonnegative OpenCV CAP_PROP_POS_MSEC
    metadata after each successful read. Missing, repeated, or regressing
    positions fall back deterministically to ``frame_index * 1000 / FPS``
    (zero-based), using the file's finite positive CAP_PROP_FPS. No arbitrary
    default FPS or wall clock is used. Milliseconds are rounded and clamped to
    at least the previous timestamp + 1 to satisfy Tasks' strictly increasing
    integer timestamps, including sub-millisecond frames. If neither source
    provides timing, ValueError is raised.

    An opened video yielding no frames returns ``(0, 110, 3)`` even without FPS
    metadata. OpenCV's ``read() == False`` denotes end of stream; its API cannot
    reliably distinguish EOF from all decoder failures. Missing files and
    unopened captures fail explicitly. The caller supplies a local Tasks model;
    this function never downloads one. Both native resources are always closed.
    """
    if frame_start is not None and (isinstance(frame_start, bool) or int(frame_start) < 1):
        raise ValueError("frame_start must be a positive 1-based frame number")
    if frame_end is not None and (isinstance(frame_end, bool) or int(frame_end) < 1):
        raise ValueError("frame_end must be a positive 1-based frame number")
    if frame_start is not None and frame_end is not None and int(frame_end) < int(frame_start):
        raise ValueError("frame_end must be greater than or equal to frame_start")
    frame_start = int(frame_start) if frame_start is not None else 1
    frame_end = int(frame_end) if frame_end is not None else None
    video_path = Path(video_path)
    model_path = Path(model_path)
    if face_model_path is None:
        raise FileNotFoundError("face-landmarker model is required; supply --face-model (no model is downloaded)")
    face_model_path = Path(face_model_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist or is not a file: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"hand-landmarker model does not exist or is not a file: {model_path}; "
            "supply a local hand_landmarker.task with model_path or --model"
        )
    if not face_model_path.is_file():
        raise FileNotFoundError(f"face-landmarker model does not exist or is not a file: {face_model_path}; supply a local face_landmarker.task with --face-model")

    cv2, mp, python, vision = _load_dependencies()
    with ExitStack() as resources:
        capture = cv2.VideoCapture(str(video_path))
        resources.callback(capture.release)
        if not capture.isOpened():
            raise ValueError(f"could not open input video: {video_path}")

        sequence = []
        landmarker = face_landmarker = None
        last_timestamp = -1
        last_position = -1.0
        decoded_frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frame_number += 1
            if decoded_frame_number < frame_start:
                continue
            if frame_end is not None and decoded_frame_number > frame_end:
                break
            index = len(sequence)
            if frame is None or getattr(frame, "size", 0) == 0:
                raise ValueError(f"decoded frame {index} is empty")

            position = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(position) and position >= 0 and position > last_position:
                timestamp = round(position)
                last_position = position
            else:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError(
                        f"cannot derive timestamp for frame {index}: position metadata "
                        f"is missing/non-increasing and video FPS must be finite and "
                        f"positive (got {fps})"
                    )
                fallback = index * 1000.0 / fps
                if not math.isfinite(fallback):
                    raise ValueError(f"cannot derive timestamp for frame {index}: invalid FPS {fps}")
                timestamp = round(fallback)
            timestamp = max(last_timestamp + 1, timestamp)
            # Tasks converts milliseconds to signed int64 microseconds internally.
            if timestamp > (2**63 - 1) // 1000:
                raise ValueError(f"timestamp out of range at frame {index}; check video timing/FPS")

            if landmarker is None:
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                try:
                    landmarker = vision.HandLandmarker.create_from_options(options)
                except Exception as exc:
                    raise RuntimeError(f"could not load hand-landmarker model {model_path}: {exc}") from exc
                resources.callback(landmarker.close)
                face_options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                )
                face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
                resources.callback(face_landmarker.close)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
            )
            face_result = face_landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
            sequence.append(_frame_points(result, face_result, index))
            last_timestamp = timestamp

    if not sequence:
        return np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
    return np.stack(sequence).astype(np.float32, copy=False)
