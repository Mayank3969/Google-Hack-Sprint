"""B10 offline replay: recording -> segments -> optional REAL classifier.

INTEGRATION_WARNING: synthetic replay validates wiring only. Person A/frontend
must provide annotated real captures. No fake classifier exists in this CLI.
Invoke with python -m scripts.evaluate_segmentation --help from ml-service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from service.segmentation import SegmentConfig, segment_recording


def boundary_metrics(detected, expected, tolerance_frames=3):
    """Match start/end pairs one-to-one, then report misses and extra splits."""
    for item in expected:
        if (type(item.get("start_frame")) is not int or type(item.get("end_frame")) is not int
                or not 0 <= item["start_frame"] <= item["end_frame"]):
            raise ValueError("annotations require nonnegative ordered start_frame/end_frame")
    candidates = []
    for i, prediction in enumerate(detected):
        for j, truth in enumerate(expected):
            start = abs(prediction["start_frame"] - truth["start_frame"])
            end = abs(prediction["end_frame"] - truth["end_frame"])
            if max(start, end) <= tolerance_frames:
                candidates.append((start + end, i, j))
    matched_predictions, matched_truth, errors = set(), set(), []
    for error, i, j in sorted(candidates):
        if i not in matched_predictions and j not in matched_truth:
            matched_predictions.add(i)
            matched_truth.add(j)
            errors.append(error / 2)
    tp = len(errors)
    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(expected) if expected else 0.0
    return {
        "true_positive_boundaries": tp,
        "extra_segments": len(detected) - tp,
        "missed_segments": len(expected) - tp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_boundary_error_frames": float(np.mean(errors)) if errors else None,
        "tolerance_frames": tolerance_frames,
    }


def evaluate_recording(frames, *, config=None, expected=None, predictor=None):
    """Predictor injection is test-only; CLI constructs the real Keras engine."""
    from service.isl_inference import SegmentRequest, integration_state

    values = np.asarray(frames)
    if values.ndim != 3 or values.shape[1:] != (110, 3):
        raise ValueError("recording must have shape (T, 110, 3)")
    segments = segment_recording(values, config=config)
    summary = [s.as_dict() for s in segments]
    report = {"segments": summary, "segment_count": len(segments), **integration_state()}
    if expected is not None:
        report["boundary_metrics"] = boundary_metrics(summary, expected)
    if predictor is not None:
        ready = [SegmentRequest(s.index, s.frames) for s in segments if s.status == "ready"]
        predictions = {r.segment_index: r.as_dict() for r in predictor.predict_segments(ready)} if ready else {}
        report["predictions"] = [predictions.get(s.index, {
            "segment_index": s.index, "status": s.status, "keyword": None
        }) for s in segments]
        if expected is not None and all("keyword" in item for item in expected):
            report["sequence_exact_match"] = (
                all(item["status"] == "recognized" for item in report["predictions"])
                and [item["keyword"] for item in report["predictions"]] == [item["keyword"] for item in expected]
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--annotations", type=Path, help="JSON list of start_frame/end_frame/optional keyword")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--predict", action="store_true", help="use committed real Keras weights")
    parser.add_argument("--evidence-kind", choices=["synthetic", "annotated_capture"], default="synthetic")
    args = parser.parse_args(argv)
    try:
        frames = np.load(args.recording, allow_pickle=False)
        expected = json.loads(args.annotations.read_text(encoding="utf-8")) if args.annotations else None
        if expected is not None and not isinstance(expected, list):
            raise ValueError("annotations must be a list")
        if args.evidence_kind == "annotated_capture" and expected is None:
            raise ValueError("annotated_capture requires annotations")
        predictor = None
        if args.predict:
            from service.isl_inference import get_inference_engine
            predictor = get_inference_engine()
        report = evaluate_recording(frames, expected=expected, predictor=predictor,
                                    config=SegmentConfig(adaptive_thresholds=not args.fixed))
        report["evidence_kind"] = args.evidence_kind
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
