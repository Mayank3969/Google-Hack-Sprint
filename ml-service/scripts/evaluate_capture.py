"""Run the complete ML-side pipeline on real video fixtures.

This is T6/T9 plus derived evidence:

    video -> MediaPipe 110-point landmarks -> fixed-rate 15 FPS sampling
    -> quality/motion diagnostics -> adaptive calibration -> Keras prediction
    -> pause segmentation -> ordered batch prediction for six two-video joins
    -> deterministic missing-data corruption matrix

INTEGRATION_WARNING:
    This command is a diagnostic release report. It does not retrain, alter
    model weights, or mark ``integration_ready`` true. Filename labels are
    ground truth only for these explicitly named fixtures. A mismatch is real
    evidence of cross-signer/domain shift, not a reason to change a prediction
    to the filename. Use --strict to return exit 2 when fixture expectations or
    pair order are not met.

Run from ``ml-service``:

    python -m scripts.evaluate_capture --video-dir ..\\tests\\Videos
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from scripts.extract_landmarks_module import extract_landmarks
from service.isl_inference import ISLInferenceEngine, SegmentRequest, integration_state
from service.segmentation import (
    SegmentConfig,
    calibrate_motion_thresholds,
    motion_components,
    resample_fixed_rate,
    segment_recording,
)


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = SERVICE_ROOT.parent / "tests" / "Videos"
DEFAULT_FACE_MODEL = SERVICE_ROOT / "models" / "face_landmarker.task"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EXPECTED_LABELS = {
    "goodmorning": "Good Morning",
    "happy": "happy",
    "monday": "Monday",
}


@dataclass(frozen=True)
class VideoFixture:
    path: Path
    expected_label: str


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _expected_from_name(path: Path, labels: list[str]) -> str | None:
    stem = _norm(path.stem)
    for key, label in EXPECTED_LABELS.items():
        if key in stem:
            return label if label in labels else None
    # For an added fixture, use the longest label contained in the filename.
    matches = [label for label in labels if _norm(label) and _norm(label) in stem]
    return max(matches, key=len) if matches else None


def discover_videos(directory: Path, labels: list[str]) -> list[VideoFixture]:
    if not directory.is_dir():
        raise ValueError(f"video directory does not exist: {directory}")
    fixtures = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            expected = _expected_from_name(path, labels)
            if expected is None:
                raise ValueError(f"video filename does not identify a saved label: {path.name}")
            fixtures.append(VideoFixture(path, expected))
    if not fixtures:
        raise ValueError(f"no supported videos found in {directory}")
    return fixtures


def _video_fps(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"video FPS is not finite and positive: {path}")
    return fps


def _safe_percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _motion_report(frames: np.ndarray, config: SegmentConfig) -> dict[str, Any]:
    components = {key: [] for key in ("left_hand", "right_hand", "hand", "face", "overall")}
    for previous, current in zip(frames[:-1], frames[1:], strict=False):
        values = motion_components(previous, current)
        for key in components:
            components[key].append(values[key])
    quiet_scores = components["overall"][: config.calibration_frames]
    pause, active = calibrate_motion_thresholds(
        quiet_scores,
        fixed_pause_threshold=config.fixed_pause_threshold,
        fixed_active_threshold=config.fixed_active_threshold,
    )
    return {
        "component_displacement": {key: _safe_percentiles(values) for key, values in components.items()},
        "calibration_scores_used": len(quiet_scores),
        "adaptive_pause_threshold": pause,
        "adaptive_active_threshold": active,
        "used_fixed_fallback": len([v for v in quiet_scores if np.isfinite(v)]) < 6,
        "pause_frame_count": int(sum(value < pause for value in components["overall"])),
        "active_frame_count": int(sum(value >= active for value in components["overall"])),
    }


def _coverage_report(frames: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(frames)
    active = np.any(np.abs(frames) > 0, axis=-1) & finite.all(axis=-1)
    return {
        "frames": int(len(frames)),
        "points": int(frames.shape[1]),
        "finite": bool(finite.all()),
        "all_zero_frames": int((~active.any(axis=1)).sum()),
        "left_hand_active_frames": int(active[:, :21].any(axis=1).sum()),
        "right_hand_active_frames": int(active[:, 21:42].any(axis=1).sum()),
        "face_active_frames": int(active[:, 42:].any(axis=1).sum()),
        "point_coverage": float(active.mean()),
    }


def _prediction(engine: ISLInferenceEngine, frames: np.ndarray, index: int = 0) -> dict[str, Any]:
    return engine.predict_segment(SegmentRequest(index, frames), top_k=5).as_dict()


def _missing_variant(frames: np.ndarray, condition: str) -> np.ndarray:
    result = frames.copy()
    if condition == "clean":
        return result
    if condition == "missing_hand_0":
        result[:, :21] = 0
    elif condition == "missing_hand_1":
        result[:, 21:42] = 0
    elif condition == "missing_face":
        result[:, 42:] = 0
    elif condition == "short_internal_gap":
        start = max(1, len(result) // 3)
        result[start:start + 3] = 0
    elif condition == "long_internal_gap":
        start = max(1, len(result) // 3)
        result[start:start + max(6, len(result) // 5)] = 0
    elif condition == "missing_first_frames":
        result[:max(1, len(result) // 6)] = 0
    elif condition == "missing_final_frames":
        result[-max(1, len(result) // 6):] = 0
    elif condition == "multiple_missing_blocks":
        result[:max(1, len(result) // 10), :21] = 0
        result[len(result) // 2:len(result) // 2 + max(1, len(result) // 10), 42:] = 0
    elif condition == "all_landmarks_absent":
        result[...] = 0
    else:
        raise ValueError(f"unknown missing-data condition: {condition}")
    return result


MISSING_CONDITIONS = (
    "clean", "missing_hand_0", "missing_hand_1", "missing_face",
    "short_internal_gap", "long_internal_gap", "missing_first_frames",
    "missing_final_frames", "multiple_missing_blocks", "all_landmarks_absent",
)


def _missing_data_report(engine: ISLInferenceEngine, frames: np.ndarray) -> dict[str, Any]:
    baseline = _prediction(engine, frames)
    conditions = {}
    for condition in MISSING_CONDITIONS:
        result = _prediction(engine, _missing_variant(frames, condition))
        conditions[condition] = {
            "status": result["status"],
            "keyword": result["keyword"],
            "confidence": result["confidence"],
            "coverage": result["quality"]["coordinate_coverage"],
            "same_as_clean": result["keyword"] == baseline["keyword"] and result["status"] == baseline["status"],
        }
    return {
        "baseline_status": baseline["status"],
        "baseline_keyword": baseline["keyword"],
        "conditions": conditions,
        "all_absent_rejected": conditions["all_landmarks_absent"]["status"] in {"insufficient_data", "tracking_lost"},
    }


def _segment_and_predict(engine: ISLInferenceEngine, frames: np.ndarray, config: SegmentConfig) -> dict[str, Any]:
    segments = segment_recording(frames, config=config)
    requests = [SegmentRequest(segment.index, segment.frames) for segment in segments if segment.status == "ready"]
    predictions = engine.predict_segments(requests, top_k=3) if requests else []
    by_index = {prediction.segment_index: prediction.as_dict() for prediction in predictions}
    return {
        "segments": [segment.as_dict() for segment in segments],
        "predictions": [by_index.get(segment.index, {
            "segment_index": segment.index,
            "status": segment.status,
            "keyword": None,
        }) for segment in segments],
    }


def evaluate(video_dir: Path, face_model: Path, *, target_fps: int = 15) -> dict[str, Any]:
    from service.isl_model import get_isl_engine

    engine = ISLInferenceEngine(get_isl_engine())
    fixtures = discover_videos(video_dir, engine.model.labels)
    config = SegmentConfig(target_fps=target_fps)
    sampled: dict[str, np.ndarray] = {}
    single = []
    for fixture in fixtures:
        started = time.perf_counter()
        source_fps = _video_fps(fixture.path)
        raw = extract_landmarks(fixture.path, face_model_path=face_model)
        fixed = resample_fixed_rate(raw, source_fps=source_fps, target_fps=target_fps)
        sampled[fixture.expected_label] = fixed
        prediction = _prediction(engine, fixed)
        single.append({
            "video": fixture.path.name,
            "expected_label": fixture.expected_label,
            "source_fps": source_fps,
            "target_fps": target_fps,
            "raw_shape": list(raw.shape),
            "sampled_shape": list(fixed.shape),
            "coverage": _coverage_report(fixed),
            "motion": _motion_report(fixed, config),
            "prediction": prediction,
            "correct": prediction["keyword"] == fixture.expected_label,
            "missing_data": _missing_data_report(engine, fixed),
            "segmentation": _segment_and_predict(engine, fixed, config),
            "elapsed_seconds": time.perf_counter() - started,
        })

    pairs = []
    labels = sorted(sampled)
    for first, second in itertools.permutations(labels, 2):
        first_frames = sampled[first]
        second_frames = sampled[second]
        # Some downloaded clips end after tracking has already disappeared.
        # Use the last frame with a visible hand for the synthetic pause so the
        # pause detector exercises a held pose rather than a tracking-loss tail.
        hand_active = np.any(np.abs(first_frames[:, :42]) > 0, axis=(1, 2))
        anchor = first_frames[np.flatnonzero(hand_active)[-1]:][:1] if hand_active.any() else first_frames[-1:]
        pause = np.repeat(anchor, max(1, round(target_fps * 0.6)), axis=0)
        combined = np.concatenate([first_frames, pause, second_frames], axis=0)
        segmented = _segment_and_predict(engine, combined, config)
        recognized = [
            item["keyword"] for item in segmented["predictions"]
            if item.get("status") == "recognized" and item.get("keyword") is not None
        ]
        pairs.append({
            "order": [first, second],
            "pause_frames_inserted": int(len(pause)),
            "combined_shape": list(combined.shape),
            "motion": _motion_report(combined, config),
            "segmentation": segmented,
            "recognized_sequence": recognized,
            "expected_sequence": [first, second],
            "exact_sequence": recognized == [first, second],
        })

    correct = sum(item["correct"] for item in single)
    return {
        "evidence_kind": "real_video_fixture_diagnostic",
        "integration_ready": False,
        "warnings": integration_state()["warnings"],
        "video_directory": str(video_dir),
        "face_model": str(face_model),
        "target_fps": target_fps,
        "fixture_count": len(single),
        "single_video_accuracy": correct / len(single) if single else 0.0,
        "single_video_correct": correct,
        "single_video_total": len(single),
        "single_videos": single,
        "pair_count": len(pairs),
        "pair_exact_sequence_count": sum(item["exact_sequence"] for item in pairs),
        "pair_tests": pairs,
        "notes": [
            "A 0.6-second repeated-last-frame pause is inserted between ordered video fixtures to exercise the stated pause assumption.",
            "Fixture filename labels are diagnostic ground truth; no weights or predictions are changed when a fixture fails.",
            "Real-video fixture accuracy is not the official INCLUDE-50 test metric and cannot replace signer-held-out evaluation.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--output", type=Path, default=SERVICE_ROOT / "model" / "isl" / "video_evidence" / "capture_report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.face_model.is_file():
            raise FileNotFoundError(f"face landmarker asset is missing: {args.face_model}")
        report = evaluate(args.video_dir, args.face_model, target_fps=args.target_fps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "single_video_accuracy": report["single_video_accuracy"],
            "pair_exact_sequence_count": report["pair_exact_sequence_count"],
            "pair_count": report["pair_count"],
            "integration_ready": report["integration_ready"],
        }, indent=2))
        if args.strict and (
            report["single_video_accuracy"] < 1.0
            or report["pair_exact_sequence_count"] != report["pair_count"]
        ):
            return 2
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
