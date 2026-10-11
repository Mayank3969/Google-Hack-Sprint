"""Train the M4 isolated-word landmark classifier.

The trainer consumes the M3 WLASL sharded output (dataset.index.json,
dataset.manifest.jsonl, and shard-*.npz) and preserves the official train/val/
test split. It also accepts a monolithic dataset.npz only when a companion
manifest provides split values.

Run from the ml-service directory:

    python scripts/04_train_bigru.py \
        --data-dir /path/to/signbridge_wlasl_m3 \
        --output-dir model/bigru

The model input is 110 points x 3 coordinates per frame. Because M3 does not
persist source FPS per video, preprocessing trims right-padding and temporally
resamples each sample to 60 normalized positions. This makes training robust
to WLASL's mixed FPS and the mobile client's planned 15 FPS, but the same
preprocessing must be reused by the future inference wrapper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_SHAPE = (110, 3)
MAX_FRAMES = 60
PREPROCESSING_VERSION = "temporal-normalize-nonzero-v1"
OFFICIAL_SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _valid_length(sample: np.ndarray) -> int:
    active = np.any(np.abs(sample) > 0, axis=(1, 2))
    if not active.any():
        return 1
    return int(np.flatnonzero(active)[-1]) + 1


def temporal_normalize(sample: np.ndarray, target_frames: int = MAX_FRAMES) -> np.ndarray:
    """Trim M3 right-padding and resample a clip to fixed temporal positions."""

    arr = np.asarray(sample, dtype=np.float32)
    if isinstance(target_frames, bool) or not isinstance(target_frames, (int, np.integer)):
        raise ValueError("target_frames must be a positive integer")
    if target_frames < 1:
        raise ValueError("target_frames must be a positive integer")
    if arr.ndim != 3 or arr.shape[1:] != FEATURE_SHAPE:
        raise ValueError(f"sample must have shape (frames, {FEATURE_SHAPE[0]}, {FEATURE_SHAPE[1]})")
    if arr.shape[0] < 1:
        raise ValueError("sample must contain at least one frame")
    if not np.isfinite(arr).all():
        raise ValueError("sample contains non-finite coordinates")
    length = _valid_length(arr)
    sequence = arr[:length]
    if length == 1:
        return np.repeat(sequence, target_frames, axis=0).astype(np.float32, copy=False)

    old_positions = np.linspace(0.0, 1.0, length, dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    flat = sequence.reshape(length, -1)
    normalized = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for feature_index in range(flat.shape[1]):
        normalized[:, feature_index] = np.interp(
            new_positions, old_positions, flat[:, feature_index]
        )
    return normalized.reshape(target_frames, *FEATURE_SHAPE)


def _validate_max_samples(max_samples: int | None) -> None:
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer")


def _read_sharded(data_dir: Path, max_samples: int | None) -> dict[str, Any]:
    _validate_max_samples(max_samples)
    index = _load_json(data_dir / "dataset.index.json")
    if index.get("format") != "signbridge-m3-sharded":
        raise ValueError("dataset.index.json is not a SignBridge M3 sharded dataset")
    if index.get("MAX_FRAMES") != MAX_FRAMES or tuple(index.get("feature_shape", ())) != FEATURE_SHAPE:
        raise ValueError("M3 dataset shape metadata must be (60, 110, 3)")
    shard_names = index.get("shards")
    if not isinstance(shard_names, list) or not shard_names or not all(
        isinstance(name, str) and name and Path(name).name == name for name in shard_names
    ):
        raise ValueError("dataset.index.json must list shard filenames")
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("dataset.index.json contains duplicate shards")
    listed_shards = set(shard_names)
    manifest_path = data_dir / "dataset.manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing M3 manifest: {manifest_path}")

    raw_labels = index.get("labels", {})
    if not isinstance(raw_labels, dict):
        raise ValueError("dataset.index.json labels must be an object")
    try:
        labels = {int(key): str(value) for key, value in raw_labels.items()}
    except (TypeError, ValueError):
        raise ValueError("dataset.index.json contains invalid class labels") from None

    # Read only the requested prefix of each official split. Grouping references
    # by shard lets a full M3 run load one shard at a time instead of retaining
    # all ~12k samples in a cache while also stacking the split arrays.
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in OFFICIAL_SPLITS}
    references: dict[str, list[tuple[str, int, int | None]]] = {}
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        split = str(record.get("subset", "")).lower()
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"manifest line {line_number} has unsupported subset: {split!r}")
        shard_name = record.get("shard")
        if not isinstance(shard_name, str) or shard_name not in listed_shards:
            raise ValueError(f"manifest line {line_number} references an unknown shard")
        shard_path = (data_dir / shard_name).resolve()
        if shard_path.parent != data_dir.resolve():
            raise ValueError(f"manifest line {line_number} has an unsafe shard path")
        row_value = record.get("row_in_shard")
        if isinstance(row_value, bool) or not isinstance(row_value, (int, np.integer, str)):
            raise ValueError(f"manifest line {line_number} has an invalid row_in_shard")
        try:
            row = int(row_value)
        except (TypeError, ValueError):
            raise ValueError(f"manifest line {line_number} has an invalid row_in_shard") from None
        if row < 0 or (shard_name, row) in seen:
            raise ValueError(f"manifest line {line_number} has a duplicate or negative row_in_shard")
        seen.add((shard_name, row))
        class_value = record.get("class_id")
        if class_value is not None:
            if isinstance(class_value, bool):
                raise ValueError(f"manifest line {line_number} has an invalid class_id")
            try:
                class_value = int(class_value)
            except (TypeError, ValueError):
                raise ValueError(f"manifest line {line_number} has an invalid class_id") from None
        records[split].append(record)
        references.setdefault(shard_name, []).append((split, row, class_value))

    if max_samples is not None:
        records = {split: rows[:max_samples] for split, rows in records.items()}
    selected: dict[str, list[tuple[str, int, int | None]]] = {split: [] for split in OFFICIAL_SPLITS}
    selected_keys: set[tuple[str, int]] = set()
    for split in OFFICIAL_SPLITS:
        for record in records[split]:
            key = (record["shard"], int(record["row_in_shard"]))
            selected[split].append((key[0], key[1], record.get("class_id")))
            selected_keys.add(key)
    if any(not records[split] for split in OFFICIAL_SPLITS):
        missing = next(split for split in OFFICIAL_SPLITS if not records[split])
        raise ValueError(f"M3 dataset has no {missing} samples")

    loaded: dict[str, np.ndarray] = {
        split: np.full((len(records[split]), MAX_FRAMES, *FEATURE_SHAPE), np.nan, dtype=np.float32)
        for split in OFFICIAL_SPLITS
    }
    loaded_y: dict[str, np.ndarray] = {
        split: np.empty(len(records[split]), dtype=np.int64) for split in OFFICIAL_SPLITS
    }
    output_positions = {split: {} for split in OFFICIAL_SPLITS}
    for split in OFFICIAL_SPLITS:
        for position, (shard_name, row, _) in enumerate(selected[split]):
            output_positions[split][(shard_name, row)] = position

    for shard_name, shard_refs in references.items():
        refs = [ref for ref in shard_refs if (shard_name, ref[1]) in selected_keys]
        if not refs:
            continue
        shard_path = data_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing M3 shard: {shard_path}")
        try:
            with np.load(shard_path, allow_pickle=False) as shard:
                if "X" not in shard or "y" not in shard:
                    raise ValueError(f"M3 shard {shard_name} must contain X and y arrays")
                shard_x = np.asarray(shard["X"])
                shard_y = np.asarray(shard["y"])
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("M3 shard"):
                raise
            raise ValueError(f"could not read M3 shard {shard_name}") from exc
        if shard_x.ndim != 4 or shard_x.shape[1:] != (MAX_FRAMES, *FEATURE_SHAPE):
            raise ValueError(f"M3 shard {shard_name} has invalid X shape {shard_x.shape}")
        if shard_x.dtype.kind not in "fiu":
            raise ValueError(f"M3 shard {shard_name} coordinates must be numeric")
        if shard_y.ndim != 1 or len(shard_y) != len(shard_x):
            raise ValueError(f"M3 shard {shard_name} has mismatched X/y lengths")
        if not np.issubdtype(shard_y.dtype, np.integer):
            raise ValueError(f"M3 shard {shard_name} labels must be integer class IDs")
        if not np.isfinite(shard_x).all():
            raise ValueError(f"M3 shard {shard_name} contains non-finite coordinates")
        for split, row, manifest_class in refs:
            if row >= len(shard_x):
                raise ValueError(f"manifest references row {row} outside M3 shard {shard_name}")
            target = int(shard_y[row])
            if manifest_class is not None and target != int(manifest_class):
                raise ValueError(f"manifest/shard class mismatch for {shard_name} row {row}")
            position = output_positions[split][(shard_name, row)]
            loaded[split][position] = shard_x[row].astype(np.float32, copy=False)
            loaded_y[split][position] = target

    for split in OFFICIAL_SPLITS:
        if not np.all(np.isfinite(loaded[split])):
            raise ValueError(f"M3 {split} split contains uninitialized or non-finite samples")
    if "sample_count" in index:
        try:
            indexed_count = int(index["sample_count"])
        except (TypeError, ValueError):
            raise ValueError("dataset.index.json sample_count is invalid") from None
        if max_samples is None and indexed_count != sum(len(records[s]) for s in OFFICIAL_SPLITS):
            raise ValueError("dataset.index.json sample_count does not match the manifest")

    return {
        "splits": loaded, "split_targets": loaded_y, "index": index, "labels": labels
    }


def load_dataset(data_dir: str | Path, max_samples: int | None = None) -> dict[str, Any]:
    """Load M3 data and return raw arrays grouped by official split."""

    _validate_max_samples(max_samples)
    root = Path(data_dir)
    if (root / "dataset.index.json").is_file():
        return _read_sharded(root, max_samples)

    dataset_path = root / "dataset.npz"
    manifest_path = root / "dataset.manifest.jsonl"
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "expected dataset.index.json for sharded M3 output, or dataset.npz "
            "with dataset.manifest.jsonl"
        )
    with np.load(dataset_path, allow_pickle=False) as data:
        X, y = data["X"].astype(np.float32), data["y"]
    if X.ndim != 4 or X.shape[1:] != (MAX_FRAMES, *FEATURE_SHAPE):
        raise ValueError(f"dataset X must have shape (samples, {MAX_FRAMES}, 110, 3)")
    if y.ndim != 1 or len(y) != len(X) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("dataset y must be integer labels with one value per sample")
    if not np.isfinite(X).all():
        raise ValueError("dataset contains non-finite coordinates")
    rows = {split: [] for split in OFFICIAL_SPLITS}
    seen_rows = set()
    record_count = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        split = str(record.get("subset", "")).lower()
        if split not in rows:
            raise ValueError(f"manifest line {line_number} has unsupported subset: {split!r}")
        row_value = record.get("row", record.get("row_in_dataset", record_count))
        try:
            row = int(row_value)
        except (TypeError, ValueError):
            raise ValueError(f"manifest line {line_number} has an invalid dataset row") from None
        if row < 0 or row >= len(X) or row in seen_rows:
            raise ValueError(f"manifest line {line_number} references an invalid dataset row")
        seen_rows.add(row)
        rows[split].append(row)
        record_count += 1
    payload: dict[str, Any] = {"splits": {}, "split_targets": {}, "index": {}, "labels": {}}
    for split, indices in rows.items():
        if not indices:
            raise ValueError(f"dataset has no {split} samples")
        indices = indices[:max_samples] if max_samples is not None else indices
        payload["splits"][split] = X[indices]
        payload["split_targets"][split] = y[indices].astype(np.int64)
    payload["labels"] = {int(v): str(v) for v in np.unique(y)}
    return payload


@dataclass
class TrainConfig:
    input_size: int = 330
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.25
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 8
    grad_clip: float = 1.0


def build_model(num_classes: int, config: TrainConfig):
    import torch.nn as nn

    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if config.input_size != int(np.prod(FEATURE_SHAPE)):
        raise ValueError("input_size must equal 110 * 3 = 330")
    if config.hidden_size < 1 or config.layers < 1:
        raise ValueError("hidden_size and layers must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")

    class LandmarkBiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.LayerNorm(config.input_size),
                nn.Linear(config.input_size, config.hidden_size),
                nn.GELU(),
            )
            self.gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.layers,
                batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size * 2),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 2, num_classes),
            )

        def forward(self, x):
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = self.input_projection(x)
            sequence, _ = self.gru(x)
            return self.classifier(sequence.mean(dim=1))

    return LandmarkBiGRU()


def _metrics(logits, targets, top_k: int = 5) -> dict[str, float]:
    import torch

    top1 = (logits.argmax(dim=1) == targets).float().mean().item()
    k = min(top_k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    top5 = top.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": top5}


def _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    metric_logits, metric_targets = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        count = targets.shape[0]
        total_loss += loss.item() * count
        total_count += count
        metric_logits.append(logits.detach().cpu())
        metric_targets.append(targets.detach().cpu())
    if not metric_logits:
        raise ValueError("cannot run an epoch on an empty dataset")
    logits = torch.cat(metric_logits)
    targets = torch.cat(metric_targets)
    return {"loss": total_loss / total_count, **_metrics(logits, targets)}


def train(data_dir: str | Path, output_dir: str | Path, *, config=None, max_samples=None, seed=42) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    config = config or TrainConfig()
    if config.input_size != np.prod(FEATURE_SHAPE):
        raise ValueError("input_size must equal 110 * 3 = 330")
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip <= 0:
        raise ValueError("learning_rate must be positive; weight_decay and grad_clip must be valid")
    if not all(np.isfinite(value) for value in (
        config.learning_rate, config.weight_decay, config.grad_clip, config.dropout
    )):
        raise ValueError("training configuration values must be finite")
    _validate_max_samples(max_samples)
    torch.manual_seed(seed)
    np.random.seed(seed)
    payload = load_dataset(data_dir, max_samples=max_samples)
    labels = payload["labels"]
    all_ids = sorted({int(x) for y in payload["split_targets"].values() for x in np.asarray(y)})
    # The index owns the vocabulary. A smoke prefix or failed extraction may
    # omit some IDs; shrinking the head to sampled IDs corrupts that mapping.
    if sorted(labels) != list(range(len(labels))) or not labels:
        raise ValueError("M3 vocabulary class IDs must be contiguous starting at zero")
    if any(class_id not in labels for class_id in all_ids):
        raise ValueError("M3 index is missing a label name for a class ID")
    label_names = [labels[class_id] for class_id in range(len(labels))]

    tensors = {}
    for split in ("train", "val", "test"):
        raw = payload["splits"][split]
        # Each sample already has the target shape; normalize in place to avoid
        # keeping raw arrays, a sample list, and another full stacked dataset.
        for row in range(len(raw)):
            raw[row] = temporal_normalize(raw[row])
        tensors[split] = TensorDataset(
            torch.from_numpy(raw),
            torch.from_numpy(np.asarray(payload["split_targets"][split], dtype=np.int64)),
        )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
        )
        for split, dataset in tensors.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(label_names), config).to(device)
    train_targets = np.asarray(payload["split_targets"]["train"], dtype=np.int64)
    counts = np.bincount(train_targets, minlength=len(label_names)).astype(np.float32)
    weights = np.ones_like(counts)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    # Evaluation must count all official held-out examples, including classes
    # absent from a limited training prefix, rather than assigning them zero loss.
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, loaders["train"], criterion, device, optimizer, config.grad_clip)
        val_metrics = _run_epoch(model, loaders["val"], eval_criterion, device)
        scheduler.step(val_metrics["top1"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d}: train loss={train_metrics['loss']:.4f} "
            f"top1={train_metrics['top1']:.3f} | val loss={val_metrics['loss']:.4f} "
            f"top1={val_metrics['top1']:.3f} top5={val_metrics['top5']:.3f}"
        )
        if val_metrics["top1"] > best_val:
            best_val = val_metrics["top1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "label_names": label_names,
                    "feature_shape": FEATURE_SHAPE,
                    "max_frames": MAX_FRAMES,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "target_mobile_fps": 15,
                    "seed": seed,
                    "best_val_top1": best_val,
                },
                output / "bigru_best.pt",
            )
        else:
            stale += 1
            if stale >= config.patience:
                print(f"early stopping at epoch {epoch}")
                break

    try:
        checkpoint = torch.load(output / "bigru_best.pt", map_location=device, weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        checkpoint = torch.load(output / "bigru_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _run_epoch(model, loaders["test"], eval_criterion, device)
    report = {
        "device": str(device),
        "classes": len(label_names),
        "label_names": label_names,
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test": test_metrics,
        "history": history,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_shape": list(FEATURE_SHAPE),
        "input_size": config.input_size,
        "target_mobile_fps": 15,
        "official_splits": {split: len(payload["split_targets"][split]) for split in ("train", "val", "test")},
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "label_mapping.json").write_text(
        json.dumps(
            {
                "id_to_label": {str(index): label for index, label in enumerate(label_names)},
                "label_to_id": {label: index for index, label in enumerate(label_names)},
                "num_classes": len(label_names),
                "feature_shape": list(FEATURE_SHAPE),
                "input_size": config.input_size,
                "target_frames": MAX_FRAMES,
                "preprocessing_version": PREPROCESSING_VERSION,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"test: loss={test_metrics['loss']:.4f} top1={test_metrics['top1']:.3f} top5={test_metrics['top5']:.3f}")
    print(f"saved: {output / 'bigru_best.pt'}")
    print(f"saved: {output / 'label_mapping.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model/bigru"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, help="limit each official split for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    train(args.data_dir, args.output_dir, config=config, max_samples=args.max_samples, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
