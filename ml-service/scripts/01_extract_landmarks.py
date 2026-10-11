"""Extract landmarks from a recorded video and save a deterministic NumPy file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Makes ``python ml-service/scripts/01_extract_landmarks.py ...`` work from
# the repository root as well as ``python scripts/01_extract_landmarks.py ...``
# from ml-service.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_landmarks_module import extract_landmarks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="recorded video to process")
    parser.add_argument("--output", type=Path, help="output .npy path (default: video path with .npy suffix)")
    parser.add_argument("--model", type=Path, help="MediaPipe hand-landmarker.task path")
    args = parser.parse_args(argv)
    output = args.output or args.video.with_suffix(".npy")
    try:
        landmarks = extract_landmarks(args.video, args.model)
        output.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(output, landmarks)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"M2 failed: {exc}", file=sys.stderr)
        return 2
    print(f"saved {output} with shape {landmarks.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
