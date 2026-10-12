"""Create a deterministic missing-landmark robustness matrix for the ISL model.

This is a Person A evaluation harness. It never retrains, changes the frozen
model, or invents an unknown class. Run it in a TensorFlow/Keras environment
after producing official validation/test arrays; it writes corrupted arrays and
a manifest that ``evaluate_isl.py`` can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    "clean",
    "missing_hand_0",
    "missing_hand_1",
    "missing_face",
    "short_internal_gap",
    "long_internal_gap",
    "missing_first_frames",
    "missing_final_frames",
    "multiple_missing_blocks",
    "all_landmarks_absent",
)


def _block_slice(condition: str) -> slice | None:
    if condition == "missing_hand_0":
        return slice(0, 63)
    if condition == "missing_hand_1":
        return slice(63, 126)
    if condition == "missing_face":
        return slice(126, 330)
    return None


def corrupt(x: np.ndarray, condition: str) -> np.ndarray:
    """Apply one corruption to already raw ``(N,T,330)`` frames."""
    if x.ndim != 3 or x.shape[2] != 330:
        raise ValueError("raw arrays must have shape (N,T,330)")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    out = np.array(x, dtype=np.float32, copy=True)
    if condition == "clean":
        return out
    if condition == "all_landmarks_absent":
        out[...] = 0.0
        return out
    block = _block_slice(condition)
    if block is not None:
        out[:, :, block] = 0.0
        return out
    frames = out.shape[1]
    if condition == "short_internal_gap":
        start = max(1, frames // 3)
        out[:, start:start + 3, :] = 0.0
    elif condition == "long_internal_gap":
        start = max(1, frames // 3)
        out[:, start:start + max(6, frames // 5), :] = 0.0
    elif condition == "missing_first_frames":
        out[:, :max(1, frames // 6), :] = 0.0
    elif condition == "missing_final_frames":
        out[:, -max(1, frames // 6):, :] = 0.0
    elif condition == "multiple_missing_blocks":
        out[:, :max(1, frames // 10), :63] = 0.0
        out[:, frames // 2:frames // 2 + max(1, frames // 10), 126:] = 0.0
    return out


def generate(raw: np.ndarray, labels: np.ndarray, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"conditions": {}, "raw_shape": list(raw.shape), "label_shape": list(labels.shape)}
    for condition in CONDITIONS:
        corrupted = corrupt(raw, condition)
        path = output_dir / f"{condition}.npz"
        np.savez_compressed(path, X=corrupted, y=labels)
        manifest["conditions"][condition] = {"path": path.name, "samples": len(corrupted)}
    (output_dir / "missing_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="NPZ containing raw X (N,T,330) and y")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with np.load(args.raw, allow_pickle=False) as data:
        if "X" not in data or "y" not in data:
            raise SystemExit("--raw NPZ must contain X and y")
        manifest = generate(data["X"], data["y"], args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
