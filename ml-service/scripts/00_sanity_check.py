"""Verify the two-hand + face MediaPipe landmark pipeline.

This is an M1 diagnostic only. It does not save landmarks or build a dataset.
Use --camera for an interactive webcam check, or --image/--video for a
deterministic, non-interactive check suitable for CI and machines without a
camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


EXPECTED_HANDS = 2
HAND_POINTS = 21
FACE_POINTS = 478
SELECTED_FACE_POINTS = 68


def count_landmarks(results: object, face_results: object) -> tuple[int, int]:
    """Return detected hand count and face point count."""

    hands = getattr(results, "hand_landmarks", None)
    if not hands:
        hand_count = 0
    else:
        hand_count = len(hands)
    faces = getattr(face_results, "face_landmarks", None)
    face_count = len(faces[0]) if faces else 0
    return hand_count, face_count


def inspect_frames(
    frames: Iterable[object], model_path: Path, face_model_path: Path, *, display: bool = False
) -> list[tuple[int, int]]:
    """Process BGR frames and return (hand_count, face_point_count) per frame."""

    counts: list[int] = []
    if not model_path.is_file():
        raise FileNotFoundError(
            f"hand-landmarker model does not exist: {model_path}; "
            "download hand_landmarker.task and pass it with --model"
        )
    if not face_model_path.is_file():
        raise FileNotFoundError(
            f"face-landmarker model does not exist: {face_model_path}; "
            "download face_landmarker.task and pass it with --face-model"
        )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.HandLandmarker.create_from_options(options) as hands, \
            vision.FaceLandmarker.create_from_options(face_options) as face:
        for frame_number, frame in enumerate(frames, start=1):
            if frame is None or getattr(frame, "size", 0) == 0:
                raise ValueError(f"frame {frame_number} is empty")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            face_results = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            count = count_landmarks(results, face_results)
            counts.append(count)
            print(f"frame {frame_number}: hands detected: {count[0]}; face source landmarks: {count[1]}; selected face points: {SELECTED_FACE_POINTS if count[1] == FACE_POINTS else 0}")

            if display:
                cv2.imshow("SignBridge M1 sanity check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    return counts


def frames_from_capture(capture: cv2.VideoCapture) -> Iterable[object]:
    """Yield frames from an opened OpenCV capture and always release it."""

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def run_camera(camera_index: int, model_path: Path, face_model_path: Path) -> list[tuple[int, int]]:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"could not open camera index {camera_index}; "
            "use --image or --video on a machine without a camera"
        )
    return inspect_frames(frames_from_capture(capture), model_path, face_model_path, display=True)


def run_file(path: Path, model_path: Path, face_model_path: Path) -> list[tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")

    image = cv2.imread(str(path))
    if image is not None:
        return inspect_frames([image], model_path, face_model_path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open image or video: {path}")
    return inspect_frames(frames_from_capture(capture), model_path, face_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, metavar="INDEX", help="use webcam index")
    source.add_argument("--image", type=Path, metavar="PATH", help="inspect one image")
    source.add_argument("--video", type=Path, metavar="PATH", help="inspect a prerecorded video")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
        help="path to the MediaPipe hand_landmarker.task model",
    )
    parser.add_argument("--face-model", type=Path, required=True, help="local face_landmarker.task model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image is not None:
        source_description = f"image {args.image}"
    elif args.video is not None:
        source_description = f"video {args.video}"
    else:
        source_description = f"camera {0 if args.camera is None else args.camera}"
    print(f"M1 starting: {source_description}")
    try:
        if args.image is not None:
            counts = run_file(args.image, args.model, args.face_model)
        elif args.video is not None:
            counts = run_file(args.video, args.model, args.face_model)
        else:
            counts = run_camera(0 if args.camera is None else args.camera, args.model, args.face_model)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"M1 failed: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()

    if not counts:
        print("M1 failed: no frames were read")
        return 1

    invalid_counts = [count for count in counts if count[0] not in range(EXPECTED_HANDS + 1) or count[1] not in (0, FACE_POINTS)]
    if invalid_counts:
        print(f"M1 failed: unexpected landmark counts: {invalid_counts}")
        return 1

    detected_hands = sum(count[0] for count in counts)
    detected_faces = sum(count[1] == FACE_POINTS for count in counts)
    print(f"M1 passed: processed {len(counts)} frame(s); {detected_hands} hand detections; {detected_faces} face detections; selected face points: {SELECTED_FACE_POINTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
