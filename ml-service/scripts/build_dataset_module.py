"""M3 dataset construction: original frame order, first 60 frames, right padding.

Defaults are anchored to ml-service; supplied paths are relative to the caller.
The extractor is imported only when a build actually needs it. Each video is
one sample, including valid all-zero (missing-hand) sequences. Metadata contains
aggregate labels/counts and preprocessing policy, never recording/model paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable, Iterable

import numpy as np
from scripts.extract_landmarks_module import FACE_FEATURE_GROUPS, FACE_FEATURE_VERSION, FACE_LANDMARK_INDICES, FACE_POINTS, FEATURE_SHAPE


MAX_FRAMES = 60
HAND_SLOTS = ("left", "right")
SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = SERVICE_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_PATH = SERVICE_ROOT / "model" / "dataset.npz"
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp"}
)


def load_wlasl_class_list(path: str | Path) -> dict[int, str]:
    """Load the WLASL ``id gloss`` vocabulary (tolerating JSON-like variants)."""
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"malformed class list line {number}")
        try:
            result[int(parts[0])] = parts[1].strip()
        except ValueError:
            raise ValueError(f"malformed class id on class list line {number}") from None
    if not result:
        raise ValueError("class list is empty")
    return result


def _wlasl_master(root: Path) -> dict[str, dict]:
    path = root / "WLASL_v0.3.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for entry in data if isinstance(data, list) else data.values():
        for inst in entry.get("instances", []):
            if "video_id" in inst:
                out[str(inst["video_id"])] = {**inst, "gloss": entry.get("gloss")}
    return out


def load_wlasl_benchmark(archive_root: str | Path, subset_file: str | Path,
                         vocabulary: dict[int, str] | None = None) -> list[dict]:
    """Return normalized benchmark records, without touching/copying videos."""
    root = Path(archive_root)
    data = json.loads(Path(subset_file).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("benchmark JSON must be an object keyed by video id")
    master = _wlasl_master(root)
    records = []
    for video_id, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("action"), list) or len(item["action"]) < 1:
            raise ValueError(f"malformed benchmark record for video {video_id}")
        action = item["action"]
        class_id = int(action[0])
        if vocabulary is not None and class_id not in vocabulary:
            continue
        video_key = str(video_id)
        meta = master.get(video_key, master.get(video_key.zfill(5), {}))
        video = root / "videos" / f"{video_key}.mp4"
        if not video.is_file() and video_key.isdigit():
            video = root / "videos" / f"{int(video_key):05d}.mp4"
        if not video.is_file():
            continue
        records.append({"video_id": str(video_id), "video_path": video,
                        "subset": item.get("subset", meta.get("split", "")),
                        "class_id": class_id, "gloss": (vocabulary or {}).get(class_id, meta.get("gloss", str(class_id))),
                        "frame_start": action[1] if len(action) > 1 else meta.get("frame_start"),
                        "frame_end": action[2] if len(action) > 2 else meta.get("frame_end"),
                        "signer_id": meta.get("signer_id")})
    return records


def build_wlasl_manifest(archive_root: str | Path, benchmark: str | Path = "1000",
                         *, subset: str = "all", vocabulary_filter: Iterable[int] | None = None,
                         max_samples: int | None = None) -> list[dict]:
    root = Path(archive_root)
    class_list = load_wlasl_class_list(root / "wlasl_class_list.txt")
    allowed = set(vocabulary_filter) if vocabulary_filter is not None else None
    path = Path(benchmark)
    if not path.is_file():
        path = root / (str(benchmark) if str(benchmark).endswith(".json") else "nslt_" + str(benchmark) + ".json")
    records = load_wlasl_benchmark(root, path, {k: v for k, v in class_list.items() if allowed is None or k in allowed})
    # Build one mapping from the complete selected benchmark before filtering a
    # split. Train/val/test must share class IDs even when built separately.
    ids = sorted({r["class_id"] for r in records})
    remap = {old: new for new, old in enumerate(ids)}
    for r in records:
        r["original_class_id"] = r["class_id"]
        r["class_id"] = remap[r["class_id"]]
        r["gloss"] = class_list.get(r["original_class_id"], r["gloss"])
    if subset != "all":
        records = [r for r in records if r["subset"] == subset]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def pad_or_truncate(seq: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
    """Validate all input frames and return float32 (max_len, 110, 3).

    No spatial normalization, mirroring, interpolation, or temporal resampling
    is applied. Nonfinite values in discarded frames are also rejected.
    """
    if isinstance(max_len, bool) or not isinstance(max_len, (int, np.integer)) or max_len < 1:
        raise ValueError("max_len must be a positive integer")
    try:
        arr = np.asarray(seq)
    except (TypeError, ValueError):
        raise ValueError("landmarks must be a rectangular numeric array") from None
    if arr.ndim != 3 or arr.shape[1:] != FEATURE_SHAPE:
        raise ValueError(f"landmarks must have shape (frames, {FEATURE_SHAPE[0]}, {FEATURE_SHAPE[1]})")
    if not len(arr):
        raise ValueError("landmarks must contain at least one frame")
    if arr.dtype.kind not in "fiu":
        raise ValueError("landmark coordinates must be real numbers")
    if not np.isfinite(arr).all():
        raise ValueError("landmark coordinates must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("landmark coordinates must fit finite float32 values")
    result = np.zeros((max_len, *FEATURE_SHAPE), dtype=np.float32)
    length = min(len(arr), max_len)
    result[:length] = arr[:length]
    return result


def metadata_path(output: str | Path) -> Path:
    """Return the companion path, e.g. dataset.metadata.json."""
    return Path(output).with_suffix(".metadata.json")


def _visible(path: Path) -> bool:
    if path.name.startswith(".") or path.is_symlink():
        return False
    # Windows hidden attributes matter in addition to portable dotfiles.
    return not (getattr(path.stat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_HIDDEN)


def _discover(raw_dir: Path) -> list[tuple[str, list[Path]]]:
    if not raw_dir.is_dir():
        raise ValueError("raw directory must exist and be a directory")
    labels = sorted(
        (entry for entry in raw_dir.iterdir() if _visible(entry) and entry.is_dir()),
        key=lambda entry: entry.name,
    )
    if not labels:
        raise ValueError("raw directory contains no usable label folders or samples")
    samples = []
    for index, folder in enumerate(labels, start=1):
        videos = sorted(
            (
                entry for entry in folder.iterdir()
                if _visible(entry) and entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=lambda entry: entry.name,
        )
        if not videos:
            raise ValueError(f"label folder #{index} contains no usable video samples")
        samples.append((folder.name, videos))
    return samples


def _check_output_paths(
    raw_dir: Path, output: Path, model_paths: tuple[Path, ...], overwrite: bool
) -> None:
    if output.suffix.lower() != ".npz":
        raise ValueError("output must have an .npz extension")
    raw_root = raw_dir.resolve()
    for target in (output, metadata_path(output)):
        resolved = target.resolve()
        if resolved == raw_root or raw_root in resolved.parents:
            raise ValueError("outputs must be outside the raw input directory")
        if target.is_symlink():
            raise ValueError("output files must not be symbolic links")
        for model_path in model_paths:
            if resolved == model_path.resolve() or (
                target.exists() and model_path.exists() and target.samefile(model_path)
            ):
                raise ValueError("outputs must not replace an extractor model")
        if target.exists():
            if not target.is_file():
                raise ValueError("output destinations must be regular files")
            # Refuse hard-linked aliases even when replacement is requested.
            if target.stat().st_nlink > 1:
                raise ValueError("output files must not have hard-linked aliases")
            if not overwrite:
                raise ValueError("dataset or metadata already exists; use --overwrite to replace")
        parent = target.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError("output parent must be a directory")


def _save_dataset(output: Path, X: np.ndarray, y: np.ndarray, metadata: dict, overwrite: bool) -> None:
    """Stage both artifacts before publication. No validation happens after this.

    Exclusive hard-link publication prevents a no-overwrite race. Replacements
    are atomic per file, but the pair is not a crash-atomic transaction.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".dataset-", delete=False) as stream:
            staged.append(Path(stream.name))
            np.savez(stream, X=X, y=y)
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".dataset-", mode="w", encoding="utf-8", delete=False
        ) as stream:
            staged.append(Path(stream.name))
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for source, target in zip(staged, (output, metadata_path(output))):
            if overwrite:
                os.replace(source, target)
            else:
                os.link(source, target)
                published.append(target)
    except BaseException:
        if not overwrite:
            for target in published:
                target.unlink(missing_ok=True)
        raise
    finally:
        for source in staged:
            source.unlink(missing_ok=True)


def build_dataset(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    model_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    overwrite: bool = False,
    extractor: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build/save X, Unicode y, and metadata; return (X, y, metadata).

    ``extractor(video_path, *, model_path=..., face_model_path=...)`` can be
    injected for tests. The default extractor requires a local face model; no
    face model is downloaded. No output is created until all samples validate.
    """
    raw_dir, output = Path(raw_dir), Path(output)
    model = Path(model_path) if model_path is not None else None
    face_model = Path(face_model_path) if face_model_path is not None else None
    models = tuple(path for path in (model, face_model) if path is not None)
    _check_output_paths(raw_dir, output, models, overwrite)
    samples = _discover(raw_dir)
    if extractor is None:
        if face_model is None:
            raise ValueError("face-landmarker model is required; supply --face-model")
        from scripts.extract_landmarks_module import extract_landmarks

        extractor = extract_landmarks
    features, targets = [], []
    for label, videos in samples:
        for video in videos:
            try:
                kwargs = {}
                if model is not None:
                    kwargs["model_path"] = model
                if face_model is not None:
                    kwargs["face_model_path"] = face_model
                seq = extractor(video, **kwargs)
            except Exception:
                # Decoder errors can contain identifying recording paths.
                raise ValueError(f"landmark extraction failed for sample #{len(features) + 1}") from None
            try:
                features.append(pad_or_truncate(seq))
            except ValueError as exc:
                raise ValueError(f"invalid sample #{len(features) + 1}: {exc}") from None
            targets.append(label)
    X = np.stack(features)
    y = np.asarray(targets, dtype=np.str_)
    metadata = {
        "MAX_FRAMES": MAX_FRAMES,
        "labels": [label for label, _ in samples],
        "per_label_counts": {label: len(videos) for label, videos in samples},
        "sample_count": len(y),
        "X_shape": list(X.shape),
        "X_dtype": "float32",
        "feature_shape": list(FEATURE_SHAPE),
        "hand_slots": list(HAND_SLOTS),
        "face_points": FACE_POINTS,
        "face_feature_version": FACE_FEATURE_VERSION,
        "face_landmark_indices": list(FACE_LANDMARK_INDICES),
        "face_feature_groups": {k: list(v) for k, v in FACE_FEATURE_GROUPS.items()},
        "face_source_points": 478,
        "y_dtype": "unicode",
        "ordering": "label name then video filename, ascending case-sensitive Unicode order",
        "sampling_policy": "original decoded frame order; no resampling; retain first 60 frames",
        "padding_policy": "right-zero-pad to 60 frames; preserve zero missing-hand and missing-face blocks",
        "coordinate_policy": "preserve raw extractor x, y, z; no normalization or mirroring",
    }
    # Check again after extraction in case a destination has appeared meanwhile.
    _check_output_paths(raw_dir, output, models, overwrite)
    _save_dataset(output, X, y, metadata, overwrite)
    return X, y, metadata


def build_wlasl_dataset(archive_root: str | Path, output: str | Path, *, benchmark="1000",
                        subset="all", max_samples=None, model_path=None, face_model_path=None,
                        overwrite=False, extractor=None):
    """Build directly from an existing WLASL archive; videos remain in place."""
    output = Path(output); root = Path(archive_root)
    if not root.is_dir():
        raise ValueError(f"archive root does not exist or is not a directory: {root}")
    if model_path is not None and not Path(model_path).is_file():
        raise ValueError("hand-landmarker model does not exist; pass the actual --model path")
    if face_model_path is not None and not Path(face_model_path).is_file():
        raise ValueError("face-landmarker model does not exist; pass the actual --face-model path")
    _check_output_paths(root, output, tuple(Path(p) for p in (model_path, face_model_path) if p), overwrite)
    manifest = build_wlasl_manifest(root, benchmark, subset=subset, max_samples=max_samples)
    if not manifest:
        raise ValueError("archive contains no usable benchmark samples")
    if extractor is None:
        if face_model_path is None:
            raise ValueError("face-landmarker model is required; supply --face-model")
        from scripts.extract_landmarks_module import extract_landmarks
        extractor = extract_landmarks
    features = []
    for sample in manifest:
        kwargs = {"model_path": model_path, "face_model_path": face_model_path,
                  "frame_start": sample["frame_start"], "frame_end": sample["frame_end"]}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            features.append(pad_or_truncate(extractor(sample["video_path"], **kwargs)))
        except FileNotFoundError as exc:
            raise ValueError(f"landmark extraction failed for sample {sample['video_id']}: required model asset is missing") from exc
        except Exception as exc:
            raise ValueError(
                f"landmark extraction failed for sample {sample['video_id']}: {type(exc).__name__}"
            ) from exc
    X = np.stack(features)
    y = np.asarray([r["class_id"] for r in manifest], dtype=np.int64)
    metadata = {"MAX_FRAMES": MAX_FRAMES, "sample_count": len(y), "X_shape": list(X.shape),
                 "X_dtype": "float32", "feature_shape": list(FEATURE_SHAPE), "y_dtype": "int64",
                 "face_feature_version": FACE_FEATURE_VERSION, "face_points": FACE_POINTS,
                 "face_landmark_indices": list(FACE_LANDMARK_INDICES),
                 "face_feature_groups": {k: list(v) for k, v in FACE_FEATURE_GROUPS.items()},
                 "face_source_points": 478,
                "labels": sorted({r["gloss"] for r in manifest}),
                "per_label_counts": {gloss: sum(r["gloss"] == gloss for r in manifest) for gloss in sorted({r["gloss"] for r in manifest})},
                "split": subset, "benchmark": str(benchmark),
                "frame_range_policy": "use benchmark 1-based inclusive action frame range before padding/truncation"}
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists() and not overwrite:
        raise ValueError("manifest already exists; use --overwrite to replace it")
    _save_dataset(output, X, y, metadata, overwrite)
    manifest_path.write_text(json.dumps([{k: v for k, v in r.items() if k != "video_path"} for r in manifest], ensure_ascii=False, indent=2), encoding="utf-8")
    return X, y, metadata


def main(argv: list[str] | None = None, *, extractor: Callable[..., np.ndarray] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the fixed-shape SignBridge M3 dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="labeled video root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="output .npz path")
    parser.add_argument("--model", type=Path, help="MediaPipe .task asset (extractor default if omitted)")
    parser.add_argument("--face-model", type=Path, required=True,
                        help="local MediaPipe face_landmarker.task asset (required; never downloaded)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing dataset and metadata")
    parser.add_argument("--archive-root", type=Path, help="WLASL archive root (enables archive mode)")
    parser.add_argument("--benchmark", default="1000", help="100, 300, 1000, 2000, or benchmark JSON path")
    parser.add_argument("--subset", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    try:
        builder = build_wlasl_dataset if args.archive_root else build_dataset
        call_args = (args.archive_root, args.output) if args.archive_root else (args.raw_dir, args.output)
        X, y, metadata = builder(
            *call_args, model_path=args.model, face_model_path=args.face_model,
            overwrite=args.overwrite, extractor=extractor, **({"benchmark": args.benchmark, "subset": args.subset, "max_samples": args.max_samples} if args.archive_root else {})
        )
    except ValueError as exc:
        parser.exit(2, f"dataset build failed: {exc}\n")
    except (OSError, ImportError):
        parser.exit(2, "dataset build failed: check filesystem access and extractor dependencies\n")
    print(f"dataset built: X={X.shape}, y={y.shape}")
    print("per-label counts: " + json.dumps(metadata["per_label_counts"], ensure_ascii=True))
    return 0
