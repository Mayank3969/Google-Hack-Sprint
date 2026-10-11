"""Save raw (frames, 110, 3) two-hand plus compact-face landmarks as NumPy .npy.

From ml-service, run either:
    python scripts/01_extract_landmarks.py clip.mp4 --model models/hand_landmarker.task
    python -m scripts.01_extract_landmarks clip.mp4 --output clip.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import json

import numpy as np

if __package__:
    from .extract_landmarks_module import DEFAULT_MODEL_PATH, extract_landmarks
else:
    from extract_landmarks_module import DEFAULT_MODEL_PATH, extract_landmarks


def _check_output(output: Path, source: Path, model: Path, face_model: Path, overwrite: bool) -> None:
    if output.suffix.lower() != ".npy":
        raise ValueError("output must have a .npy suffix")
    if output.is_symlink():
        raise ValueError("output must not be a symbolic link")
    for protected in (source, model, face_model):
        if output.resolve() == protected.resolve() or (
            output.exists() and protected.exists() and output.samefile(protected)
        ):
            raise ValueError("output must not overwrite the input video or model")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; use --overwrite to replace it")
    if not output.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output.parent}")
    metadata = output.with_suffix(".json")
    if metadata.is_symlink() or (metadata.exists() and not overwrite):
        raise FileExistsError(f"metadata output already exists: {metadata}; use --overwrite to replace it")
    if not metadata.parent.is_dir():
        raise ValueError(f"metadata output directory does not exist: {metadata.parent}")


def _save_array(output: Path, array: np.ndarray, source: Path, model: Path, face_model: Path, overwrite: bool) -> None:
    """Publish a fully written file; failure never truncates an existing output."""
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".landmarks-", delete=False) as stream:
            staged = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        _check_output(output, source, model, face_model, overwrite)
        if overwrite:
            os.replace(staged, output)
        else:
            # Exclusive publication: fail rather than clobber a concurrent writer.
            os.link(staged, output)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="local recorded video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="local MediaPipe hand_landmarker.task model")
    parser.add_argument("--face-model", type=Path, required=True,
                        help="local MediaPipe face_landmarker.task model (never downloaded)")
    parser.add_argument("--output", type=Path, help="destination .npy (default: input with .npy suffix)")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    args = parser.parse_args(argv)
    try:
        output = args.output if args.output is not None else args.video.with_suffix(".npy")
        _check_output(output, args.video, args.model, args.face_model, args.overwrite)
        array = extract_landmarks(args.video, model_path=args.model, face_model_path=args.face_model)
        _save_array(output, array, args.video, args.model, args.face_model, args.overwrite)
        metadata = output.with_suffix(".json")
        metadata.write_text(json.dumps({
            "feature_shape": [110, 3],
            "hand_slots": ["left", "right"],
            "hand_points_per_slot": 21,
            "face_points": 68,
            "face_source_points": 478,
            "face_feature_version": "compact-v1",
        }, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"M2 failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved {output} with shape {array.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
