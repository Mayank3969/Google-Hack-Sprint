"""Train the M4 RandomForest classifier from ``model/dataset.npz``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_classifier import DEFAULT_SEED, format_report, train_classifier


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=service_root / "model" / "dataset.npz")
    parser.add_argument("--output", type=Path, default=service_root / "model" / "classifier.pkl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        metadata = train_classifier(args.dataset, args.output, seed=args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"M4 failed: {exc}", file=sys.stderr)
        return 2
    print(format_report(metadata))
    print(f"saved classifier: {args.output}")
    print(f"saved metadata: {args.output.with_name('classifier_metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
