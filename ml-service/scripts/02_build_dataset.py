"""Build ``model/dataset.npz`` from labeled recorded videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataset_builder import MAX_FRAMES, build_dataset, pad_or_truncate


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=service_root / "data" / "raw_videos")
    parser.add_argument("--output", type=Path, default=service_root / "model" / "dataset.npz")
    args = parser.parse_args(argv)
    try:
        metadata = build_dataset(args.raw_dir, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"M3 failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"dataset built: X=({metadata['num_samples']}, 60, 21, 3), "
        f"y=({metadata['num_samples']},), per-label counts={metadata['per_label_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
