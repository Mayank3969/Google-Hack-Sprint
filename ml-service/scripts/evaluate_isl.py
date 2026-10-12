"""Re-evaluate both frozen ISL models, without training or random split fallback.

Examples (from the repo root):
  python ml-service/scripts/evaluate_isl.py --csv path/to/official.csv
  python ml-service/scripts/evaluate_isl.py --arrays test.npz --official-test

NPZ accepts X_test/y_test (explicit official-test attestation still required),
or X/y/split where only test-prefixed samples are selected. Separate NPY arrays
use --arrays X.npy --labels y.npy --official-test. Already-preprocessed arrays
are never normalized again. Outputs go to model/isl/reevaluation_metrics.json;
the original archived evaluation_metrics.json is preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(SERVICE_ROOT))

from scripts.freeze_isl_artifacts import (
    DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, MODEL_NAMES, NUM_CLASSES,
    file_record, inspect_keras_archive, read_json, validate_label_maps,
)


def load_runtime():
    """Lazy imports keep --help and artifact-only tests TensorFlow-free."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import tensorflow as tf
        import keras
    except ImportError as exc:
        raise RuntimeError(
            "ISL evaluation requires TensorFlow and compatible Keras (artifact metadata: Keras 3.13.2). "
            "Install them in a supported Python environment; no evaluation was performed."
        ) from exc
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError("ISL evaluation requires the TensorFlow Keras backend")
    return tf, keras


def custom_objects(keras) -> dict:
    """Inference implementations of the three classes serialized by the notebook.

    Child layer construction/order and names match the notebook. Augmentation is
    identity during inference; no notebook execution or training code is needed.
    """
    layers = keras.layers

    class LandmarkAugmentation(layers.Layer):
        def __init__(self, jitter_std=0.005, mask_prob=0.05, **kwargs):
            super().__init__(**kwargs)
            self.jitter_std, self.mask_prob = jitter_std, mask_prob

        def call(self, inputs, training=None):
            if training:
                raise ValueError("evaluate_isl supports inference only")
            return inputs

    class Conv1DTokenBlock(layers.Layer):
        def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
            super().__init__(**kwargs)
            self.filters, self.kernel_size, self.dilation_rate = filters, kernel_size, dilation_rate

        def build(self, input_shape):
            self.dw_conv = layers.DepthwiseConv1D(kernel_size=self.kernel_size, padding="same",
                                                dilation_rate=self.dilation_rate, use_bias=False)
            self.pw_conv = layers.Conv1D(self.filters, kernel_size=1, padding="same", use_bias=False)
            self.bn = layers.BatchNormalization()
            self.act = layers.Activation("swish")
            super().build(input_shape)

        def call(self, inputs, training=None):
            return self.act(self.bn(self.pw_conv(self.dw_conv(inputs)), training=training))

    class TransformerEncoderBlock(layers.Layer):
        def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.d_model, self.num_heads, self.ff_dim = d_model, num_heads, ff_dim
            self.dropout_rate = dropout

        def build(self, input_shape):
            self.mha = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                key_dim=self.d_model // self.num_heads,
                                                dropout=self.dropout_rate)
            self.ffn = keras.Sequential([layers.Dense(self.ff_dim, activation="swish"),
                                         layers.Dropout(self.dropout_rate), layers.Dense(self.d_model)])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(self.dropout_rate)
            self.dropout2 = layers.Dropout(self.dropout_rate)
            super().build(input_shape)

        def call(self, inputs, training=None):
            attention = self.dropout1(self.mha(inputs, inputs, training=training), training=training)
            out1 = self.layernorm1(inputs + attention)
            ffn = self.dropout2(self.ffn(out1, training=training), training=training)
            return self.layernorm2(out1 + ffn)

    return {cls.__name__: cls for cls in (LandmarkAugmentation, Conv1DTokenBlock, TransformerEncoderBlock)}


def validate_test_arrays(X, y):
    import numpy as np

    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 3 or X.shape[1:] != (60, 660) or len(X) == 0:
        raise ValueError("preprocessed test arrays must have nonempty shape (N,60,660)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("test tensors must contain finite real numbers")
    X = X.astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("test tensors must fit finite float32 values")
    if y.ndim != 1 or len(y) != len(X) or y.dtype.kind not in "iu":
        raise ValueError("test labels must be a matching one-dimensional integer array")
    if not np.isin(y, np.arange(NUM_CLASSES)).all():
        raise ValueError("test labels must be IDs 0..49 from the saved label map")
    return X, y.astype(np.int64)


def load_array_test(path: Path, labels_path: Path | None, official_test: bool):
    import numpy as np

    provenance = {"arrays": file_record(path)}
    if path.suffix.lower() == ".npz":
        if labels_path is not None:
            raise ValueError("--labels is only for separate .npy arrays")
        with np.load(path, allow_pickle=False) as data:
            if "X_test" in data and "y_test" in data:
                if not official_test:
                    raise ValueError("X_test/y_test arrays require --official-test to attest their official test origin")
                X, y = data["X_test"], data["y_test"]
                split = data["split_test"] if "split_test" in data else None
            elif "X" in data and "y" in data:
                X, y = data["X"], data["y"]
                split = data["split"] if "split" in data else None
            else:
                raise ValueError("NPZ must contain X/y or X_test/y_test")
        if split is not None:
            if split.ndim != 1 or len(split) != len(X) or split.dtype.kind not in "US":
                raise ValueError("split must be a matching string array")
            test_mask = np.char.startswith(np.char.lower(split.astype(str)), "test")
            X, y = X[test_mask], y[test_mask]
            provenance["split_selection"] = "provided split array, test-prefixed samples only"
        elif not official_test:
            raise ValueError("arrays without split metadata require --official-test to attest their official test origin")
        else:
            provenance["split_selection"] = "caller attested official test-only arrays; not independently verifiable"
    elif path.suffix.lower() == ".npy":
        if labels_path is None or not official_test:
            raise ValueError("NPY inputs require --labels and --official-test")
        X = np.load(path, allow_pickle=False)
        y = np.load(labels_path, allow_pickle=False)
        provenance["labels"] = file_record(labels_path)
        provenance["split_selection"] = "caller attested official test-only arrays; not independently verifiable"
    else:
        raise ValueError("--arrays must be .npz or .npy")
    return (*validate_test_arrays(X, y), provenance)


def load_csv_test(path: Path, label_to_id: dict):
    """Stream official test frames; delegate all tensor math to the shared module."""
    import numpy as np
    try:
        import pandas as pd
        preprocessing = importlib.import_module("service.isl_preprocessing")
    except ImportError as exc:
        raise RuntimeError("CSV evaluation requires pandas and service.isl_preprocessing; no alternate preprocessing is used") from exc
    # Person A's shared module owns the exact resampling/normalization/velocity
    # contract.  Keep the fallback name for a compatible future module revision,
    # but never reproduce that math in this evaluator.
    preprocess = getattr(preprocessing, "prepare_model_input", None)
    preprocess_name = "service.isl_preprocessing.prepare_model_input"
    if not callable(preprocess):
        preprocess = getattr(preprocessing, "preprocess_sequence", None)
        preprocess_name = "service.isl_preprocessing.preprocess_sequence"
    if not callable(preprocess):
        raise RuntimeError("service.isl_preprocessing must expose prepare_model_input(raw_sequence) -> (60,660)")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    seq_col = next((c for c in ("sequence_id", "sample_id", "video_name") if c in header), None)
    frame_col = next((c for c in ("frame_id", "frame_idx", "frame") if c in header), None)
    label_col = next((c for c in ("sign", "official_sign", "word", "gloss", "label", "sign_id") if c in header), None)
    split_col = next((c for c in header if c.lower() in {"split", "subset", "type"}), None)
    if seq_col is None or label_col is None or split_col is None:
        raise ValueError("CSV must identify sequence, label and official split columns; no random split fallback")
    metadata_cols = {"sign_id", "category", "sign", "official_sign", "class_key", "sample_id",
                     "video_name", "video_relpath", "sequence_id", "frame_id", "frame_idx", "frame",
                     "split", "subset", "type", "word", "gloss", "label"}
    missing_cols = {c for c in header if c.lower().startswith("missing_")}
    feature_cols = [c for c in header if c not in metadata_cols and c not in missing_cols]
    if len(feature_cols) != 330:
        raise ValueError(f"CSV must contain exactly 330 raw coordinate columns in trained order; got {len(feature_cols)}")
    sequence_rows = {}
    for chunk in pd.read_csv(path, usecols=[seq_col, label_col, split_col, *([frame_col] if frame_col else []), *feature_cols],
                              chunksize=2000, low_memory=False):
        if chunk[seq_col].isna().any():
            raise ValueError("CSV sequence IDs must not be missing")
        test_rows = chunk[chunk[split_col].astype(str).str.lower().str.startswith("test")]
        for sequence_id, group in test_rows.groupby(seq_col, sort=False):
            state = sequence_rows.setdefault(str(sequence_id), [])
            state.append(group)
    X, y = [], []
    for sequence_id, pieces in sequence_rows.items():
        group = pd.concat(pieces, ignore_index=True)
        if group[split_col].astype(str).str.lower().nunique() != 1:
            raise ValueError(f"sequence {sequence_id!r} has multiple official splits")
        if group[label_col].isna().any() or group[label_col].astype(str).nunique() != 1:
            raise ValueError(f"sequence {sequence_id!r} has multiple or missing labels")
        label = group[label_col].iloc[0]
        label_key = label if label in label_to_id else str(label)
        if label_key not in label_to_id:
            raise ValueError(f"CSV label {label!r} is not in the saved label map; labels are never re-encoded")
        if frame_col is not None:
            if group[frame_col].isna().any():
                raise ValueError(f"sequence {sequence_id!r} has missing frame IDs")
            group = group.sort_values(frame_col, kind="stable")
        X.append(preprocess(group[feature_cols].to_numpy(dtype=np.float32)))
        y.append(label_to_id[label_key])
    if not X:
        raise ValueError("CSV contains no official test sequences")
    provenance = {"csv": file_record(path), "split_selection": "official CSV test-prefixed sequences only",
                  "preprocessing_module": preprocess_name,
                  "preprocessing_version": getattr(preprocessing, "PREPROCESSING_VERSION", None),
                  "coordinate_columns": feature_cols}
    return (*validate_test_arrays(np.stack(X), np.asarray(y)), provenance)


def evaluate_model(model, X, y, names: list[str], batch_size: int) -> dict:
    import numpy as np
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("ISL evaluation requires scikit-learn for fixed 50-class metrics") from exc
    if list(model.input_shape) != [None, 60, 660] or list(model.output_shape) != [None, NUM_CLASSES]:
        raise ValueError("runtime model signature must be (None,60,660) -> (None,50)")
    # Warm up and materialize output to synchronize device work before timing.
    model(X[:1], training=False).numpy()
    start = time.perf_counter()
    probabilities = np.concatenate([model(X[i:i + batch_size], training=False).numpy()
                                    for i in range(0, len(X), batch_size)])
    elapsed = time.perf_counter() - start
    if probabilities.shape != (len(X), NUM_CLASSES) or not np.isfinite(probabilities).all():
        raise ValueError("model returned invalid probability shape or nonfinite output")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("model did not return normalized probabilities")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y
    latency_ms = []
    for sample in X:
        start = time.perf_counter()
        model(sample[None], training=False).numpy()
        latency_ms.append((time.perf_counter() - start) * 1000)
    labels = list(range(NUM_CLASSES))
    return {
        "num_test_samples": len(X), "num_classes": NUM_CLASSES, "labels": labels,
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, labels=labels, average="weighted", zero_division=0)),
        "per_class_report": classification_report(y, predicted, labels=labels, target_names=names,
                                                     zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
        "confidence": {"mean": float(confidence.mean()), "median": float(np.median(confidence)),
                       "correct_mean": float(confidence[correct].mean()) if correct.any() else None,
                       "incorrect_mean": float(confidence[~correct].mean()) if (~correct).any() else None,
                       "predicted_ids": predicted.tolist(), "true_ids": y.tolist(),
                       "predicted_probabilities": confidence.tolist(),
                       "true_class_probabilities": probabilities[np.arange(len(y)), y].tolist()},
        "latency": {"method": "warmed eager model call with training=False, numpy synchronization; preprocessing/loading excluded",
                    "single_sequence_count": len(latency_ms), "single_sequence_mean_ms": float(np.mean(latency_ms)),
                    "single_sequence_p50_ms": float(np.percentile(latency_ms, 50)),
                    "single_sequence_p95_ms": float(np.percentile(latency_ms, 95)),
                    "batch_size": batch_size, "batched_total_ms": elapsed * 1000,
                    "batched_per_sequence_ms": elapsed * 1000 / len(X)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--arrays", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--official-test", action="store_true", help="attest that arrays without split metadata are official test-only")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "reevaluation_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.csv and args.labels:
            raise ValueError("--labels cannot be used with --csv")
        tf, keras = load_runtime()
        label_to_id = read_json(args.models_dir / "label_to_id.json")
        names = validate_label_maps(label_to_id, read_json(args.models_dir / "id_to_label.json"))
        X, y, provenance = (load_csv_test(args.csv, label_to_id) if args.csv else
                            load_array_test(args.arrays, args.labels, args.official_test))
        result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
                  "dataset": provenance, "label_names": names, "zero_division": 0, "models": {}}
        for filename in MODEL_NAMES:
            path = args.models_dir / filename
            inspection = inspect_keras_archive(path)
            model = keras.models.load_model(path, custom_objects=custom_objects(keras), compile=False, safe_mode=True)
            result["models"][filename] = {"artifact": file_record(path), "serialized_signature": inspection,
                                          **evaluate_model(model, X, y, names, args.batch_size)}
            keras.backend.clear_session()
        inputs = [args.models_dir / name for name in (*MODEL_NAMES, "label_to_id.json", "id_to_label.json", "evaluation_metrics.json")]
        inputs += [p for p in (args.csv, args.arrays, args.labels) if p is not None]
        if any(args.output.resolve() == path.resolve() for path in inputs):
            raise ValueError("evaluation output must not overwrite an input artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ISL evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluated both saved models on {len(y)} test sequences: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
