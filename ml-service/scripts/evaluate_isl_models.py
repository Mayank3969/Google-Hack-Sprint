"""B7 compare both REAL Keras artifacts on Person A's prepared split.

INTEGRATION_WARNING: --smoke uses generated tensors and NEVER reports accuracy.
Person A supplies X=(N,60,660), y=(N,) in NPZ plus provenance JSON. Choose using
validation, then evaluate the frozen choice on test; do not select on test.
Run: python -m scripts.evaluate_isl_models --help (from ml-service).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from service.isl_model import DEFAULT_MODEL_PATH, ISLKerasEngine
from service.isl_preprocessing import PREPROCESSING_VERSION


def classification_metrics(y, probabilities, labels):
    y = np.asarray(y)
    p = np.asarray(probabilities)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in "iu"
            or (y < 0).any() or (y >= len(labels)).any()):
        raise ValueError("y must contain non-empty integer class IDs in range")
    if p.shape != (len(y), len(labels)) or not np.isfinite(p).all():
        raise ValueError("probabilities shape/values invalid")
    predicted = p.argmax(axis=1)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    support = matrix.sum(axis=1)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(len(labels)), where=matrix.sum(axis=0) > 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(labels)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(labels)), where=precision + recall > 0)
    return {
        "samples": len(y), "accuracy": float(np.mean(predicted == y)),
        "macro_f1_all_classes": float(f1.mean()),
        "macro_f1_supported_classes": float(f1[support > 0].mean()),
        "weighted_f1": float(np.sum(f1 * support) / len(y)),
        "unsupported_classes": [labels[i] for i in range(len(labels)) if support[i] == 0],
        "per_class": [{"class_id": i, "label": label, "support": int(support[i]),
                       "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                      for i, label in enumerate(labels)],
        "confusion_matrix": matrix.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data", type=Path, help="independently prepared X/y NPZ")
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args(argv)
    if args.smoke == bool(args.data) or (args.data and not args.provenance):
        parser.error("choose --smoke OR --data with --provenance")
    try:
        metadata = None
        if args.smoke:
            x = np.random.default_rng(2026).normal(0, .01, (2, 60, 660)).astype(np.float32)
            y = None
        else:
            metadata = json.loads(args.provenance.read_text(encoding="utf-8"))
            if metadata.get("split") not in {"validation", "test"}:
                raise ValueError("provenance split must be validation or test")
            if metadata.get("synthetic") is not False or not metadata.get("dataset_sha256"):
                raise ValueError("real dataset provenance and synthetic:false are required")
            if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError("preprocessing version does not match serving")
            with np.load(args.data, allow_pickle=False) as data:
                x, y = data["X"], data["y"]
            if x.ndim != 3 or x.shape[1:] != (60, 660) or not np.isfinite(x).all() or not len(x):
                raise ValueError("X must contain finite (N,60,660) tensors")
        outputs = []
        for filename in ("best_isl_model_v2.keras", "isl_1dcnn_transformer_v2.keras"):
            start = time.perf_counter()
            engine = ISLKerasEngine(DEFAULT_MODEL_PATH.with_name(filename))
            loaded = time.perf_counter() - start
            if metadata and metadata.get("labels_sha256") != engine.labels_sha256:
                raise ValueError("provenance label map hash does not match model bundle")
            start = time.perf_counter()
            p = np.concatenate([engine.predict_probabilities(x[i:i + 10]) for i in range(0, len(x), 10)])
            elapsed = time.perf_counter() - start
            output = {"artifact": filename, "model_sha256": engine.model_sha256,
                      "labels_sha256": engine.labels_sha256, "load_seconds": loaded,
                      "inference_seconds": elapsed, "output_shape": list(p.shape)}
            if y is not None:
                output["metrics"] = classification_metrics(y, p, engine.labels)
            outputs.append(output)
        report = {
            "evidence_kind": "synthetic_artifact_smoke" if args.smoke else "prepared_dataset_evaluation",
            "integration_ready": False, "preprocessing_version": PREPROCESSING_VERSION,
            "artifacts": outputs, "provenance": metadata,
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
            "selection_status": "PENDING" if args.smoke else "REVIEW_VALIDATION_BEFORE_SELECTING",
        }
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "detail": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
