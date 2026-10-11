"""Run via python scripts/02_build_dataset.py or python -m scripts.02_build_dataset."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset_module import main


if __name__ == "__main__":
    raise SystemExit(main())
