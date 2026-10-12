"""Freeze ISL provenance using only the standard library, never TensorFlow.

Run from any directory: python ml-service/scripts/freeze_isl_artifacts.py
--notebook and --csv name exact files; no search or guessed substitutions occur.
ZIP/config checks describe the serialized graph, not a runtime load or evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = SERVICE_ROOT / "models"
DEFAULT_OUTPUT_DIR = SERVICE_ROOT / "model" / "isl"
MODEL_NAMES = ("best_isl_model_v2.keras", "isl_1dcnn_transformer_v2.keras")
EXPECTED_INPUT_SHAPE = [None, 60, 660]
NUM_CLASSES = 50


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def validate_label_maps(label_to_id: dict, id_to_label: dict) -> list[str]:
    """Require a bijection with exactly the canonical JSON IDs 0..49."""
    if not isinstance(label_to_id, dict) or not isinstance(id_to_label, dict):
        raise ValueError("both label maps must be JSON objects")
    if len(label_to_id) != NUM_CLASSES or set(id_to_label) != {str(i) for i in range(NUM_CLASSES)}:
        raise ValueError("label maps must contain exactly IDs 0..49")
    if any(not isinstance(label, str) or not label.strip() for label in label_to_id):
        raise ValueError("labels must be nonempty strings")
    if any(type(value) is not int for value in label_to_id.values()):
        raise ValueError("label IDs must be integers (not bool, float or string)")
    if set(label_to_id.values()) != set(range(NUM_CLASSES)):
        raise ValueError("label maps must contain exactly IDs 0..49")
    for label, class_id in label_to_id.items():
        if id_to_label[str(class_id)] != label:
            raise ValueError("label maps are not consistent in both directions")
    return [id_to_label[str(i)] for i in range(NUM_CLASSES)]


def _endpoint_name(value, description: str) -> str:
    # Keras versions serialize a single endpoint either directly or in a list.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, list) or len(value) != 3 or not isinstance(value[0], str):
        raise ValueError(f"expected one serialized {description} endpoint")
    if value[1:] != [0, 0]:
        raise ValueError(f"unsupported {description} node/tensor reference")
    return value[0]


def inspect_keras_archive(path: str | Path) -> dict:
    """Check the actual graph endpoints, not an unrelated Dense layer's width."""
    with zipfile.ZipFile(path) as archive:
        for member in ("config.json", "model.weights.h5"):
            if archive.namelist().count(member) != 1:
                raise ValueError(f"Keras archive must contain exactly one {member}")
        graph = json.loads(archive.read("config.json"), object_pairs_hook=_unique_object)
        metadata = (json.loads(archive.read("metadata.json"), object_pairs_hook=_unique_object)
                    if "metadata.json" in archive.namelist() else {})
    config = graph.get("config", {})
    layers = config.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise ValueError("expected a Functional Keras graph with layers")
    by_name = {}
    for layer in layers:
        name = layer.get("name", layer.get("config", {}).get("name"))
        if name in by_name:
            raise ValueError("duplicate layer names in Keras graph")
        by_name[name] = layer
    input_name = _endpoint_name(config.get("input_layers"), "input")
    output_name = _endpoint_name(config.get("output_layers"), "output")
    if input_name not in by_name or output_name not in by_name:
        raise ValueError("model endpoints reference missing layers")
    input_layer, output_layer = by_name[input_name], by_name[output_name]
    input_config = input_layer.get("config", {})
    shape = input_config.get("batch_shape", input_config.get("batch_input_shape"))
    if input_layer.get("class_name") != "InputLayer" or shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"model input batch shape must be {EXPECTED_INPUT_SHAPE}; got {shape}")
    output_config = output_layer.get("config", {})
    width = output_config.get("units")
    if output_layer.get("class_name") != "Dense" or type(width) is not int or width != NUM_CLASSES:
        raise ValueError(f"model output probability width must be 50; got {width}")
    if output_config.get("activation") != "softmax":
        raise ValueError("model output must be softmax probabilities")
    custom_layers = sorted({layer.get("class_name") for layer in layers if layer.get("module") is None})
    return {
        "model_name": config.get("name"), "input_batch_shape": shape,
        "output_probability_width": width, "output_activation": "softmax",
        "keras_version": metadata.get("keras_version"),
        "tensorflow_version": metadata.get("tensorflow_version", metadata.get("tf_version")),
        "model_metadata": metadata, "custom_layer_classes": custom_layers,
        "validation": "serialized ZIP config only; weights not runtime-loaded",
    }


def file_record(path: str | Path, *, source_path: str | None = None) -> dict:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"source_path": source_path if source_path is not None else str(source),
            "size_bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def inspect_notebook_versions(path: str | Path | None) -> dict:
    """Extract explicitly printed runtime versions from the supplied notebook."""
    if path is None:
        return {"tensorflow": None, "keras": None, "source": None}
    text = Path(path).read_text(encoding="utf-8")
    tensorflow = re.search(r"TensorFlow version:\s*([0-9][^\s\"]*)", text)
    keras = re.search(r"Keras version:\s*([0-9][^\s\"]*)", text)
    return {
        "tensorflow": tensorflow.group(1).rstrip("\\n").strip() if tensorflow else None,
        "keras": keras.group(1).rstrip("\\n").strip() if keras else None,
        "source": "notebook execution output" if tensorflow or keras else None,
    }


def build_manifest(models_dir: str | Path = DEFAULT_MODELS_DIR, *, notebook=None, csv=None) -> dict:
    root = Path(models_dir)
    labels = validate_label_maps(read_json(root / "label_to_id.json"), read_json(root / "id_to_label.json"))
    artifacts = {}
    for name in MODEL_NAMES:
        artifacts[name] = {**file_record(root / name), **inspect_keras_archive(root / name)}
    for name in ("label_to_id.json", "id_to_label.json", "evaluation_metrics.json"):
        artifacts[name] = file_record(root / name)
    # Parsing confirms valid JSON; supplied metrics remain historical, not new evidence.
    historical_metrics = read_json(root / "evaluation_metrics.json")
    sources = {"notebook": (file_record(notebook, source_path=str(notebook))
                             if notebook is not None else None),
               "csv": (file_record(csv, source_path=str(csv)) if csv is not None else None)}
    notebook_versions = inspect_notebook_versions(notebook)
    tensorflow_available = importlib.util.find_spec("tensorflow") is not None
    limitations = [
        "Freezing performs static config/label/hash checks only; no model runtime load or inference is attempted.",
        "Archived evaluation_metrics.json is historical and has not been reproduced by this command.",
        "A supplied notebook hash identifies that exact file, not proof it trained these model weights.",
    ]
    if not tensorflow_available:
        limitations.append("TensorFlow is unavailable in this Python runtime; model loading and evaluation cannot be verified here.")
    if csv is None:
        limitations.append("Exact training/evaluation CSV was not supplied; its identity and official test split are unverified.")
    if notebook is None:
        limitations.append("Exact training notebook was not supplied; notebook provenance is unverified.")
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_input_batch_shape": EXPECTED_INPUT_SHAPE, "num_classes": NUM_CLASSES,
        "label_names": labels, "label_maps_valid": True, "artifacts": artifacts,
        "sources": sources, "historical_evaluation_metrics": historical_metrics,
        "training_environment": notebook_versions,
        "runtime_validation": {"attempted": False, "tensorflow_available": tensorflow_available},
        "limitations": limitations,
    }


def checksum_text(manifest: dict) -> str:
    lines = ["ISL artifact freeze (SHA256; exact source files)", f"Created: {manifest['created_at_utc']}", ""]
    for name, record in {**manifest["artifacts"], **manifest["sources"]}.items():
        if record is None:
            lines.append(f"{name}: not supplied")
            continue
        lines.extend([f"{name}", f"  Source: {record['source_path']}",
                      f"  Size: {record['size_bytes']} bytes", f"  SHA256: {record['sha256']}"])
        if "input_batch_shape" in record:
            lines.extend([f"  Input batch shape: {record['input_batch_shape']}",
                          f"  Output: {record['output_probability_width']} softmax probabilities",
                          f"  Keras: {record['keras_version'] or 'not recorded'}",
                          f"  TensorFlow: {record['tensorflow_version'] or 'not recorded'}"])
    environment = manifest.get("training_environment", {})
    lines.extend([
        "",
        f"Training environment from notebook output: TensorFlow {environment.get('tensorflow') or 'not recorded'}, Keras {environment.get('keras') or 'not recorded'}",
        "Label maps: exact IDs 0..49, consistent in both directions",
        "",
        "Limitations:",
    ])
    lines.extend(f"- {item}" for item in manifest["limitations"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, help="exact notebook to hash (no automatic search)")
    parser.add_argument("--csv", type=Path, help="exact CSV to hash (no automatic search)")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.models_dir, notebook=args.notebook, csv=args.csv)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "artifact_checksums.txt").write_text(checksum_text(manifest), encoding="utf-8")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Artifact freeze failed: {exc}", file=sys.stderr)
        return 1
    print(f"Frozen manifest/checksums: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
