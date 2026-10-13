"""Replay a saved ``(T, 110, 3)`` NumPy landmark recording through the segmenter.

Usage from ``ml-service``:

    python -m scripts.segment_signs recording.npy

The script is an offline reference for the future browser implementation.  It
does not upload video or call the Keras model.

INTEGRATION_WARNING (B9): fixed-rate replay only. Person A/frontend must validate
thresholds on annotated captures and port the reference state machine to JS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from service.segmentation import SegmentConfig, segment_recording


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path, help=".npy file with shape (T, 110, 3)")
    parser.add_argument("--fixed", action="store_true", help="disable adaptive thresholds")
    args = parser.parse_args()
    try:
        frames = np.load(args.recording, allow_pickle=False)
        if frames.ndim != 3 or frames.shape[1:] != (110, 3):
            raise ValueError("recording must have shape (T, 110, 3)")
        segments = segment_recording(frames, config=SegmentConfig(adaptive_thresholds=not args.fixed))
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}))
        return 2
    print(json.dumps({"integration_ready": False, "segments": [s.as_dict() for s in segments]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
