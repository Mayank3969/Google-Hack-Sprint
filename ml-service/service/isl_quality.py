"""B6 quality and abstention policy.

INTEGRATION_WARNING: these are UNCALIBRATED development thresholds, not measured
unknown-sign detection. Person A must supply held-out coverage/confidence/margin
results before release. Preserve per-hand checks: a missing face or non-signing
hand alone must not reject a usable one-hand sign. See PERSON_B_HANDOFF.md.
"""

from dataclasses import dataclass
import math
import os

from service.isl_preprocessing import SegmentQuality


@dataclass(frozen=True)
class QualityPolicy:
    min_coverage: float = 0.05
    min_valid_frames: int = 3
    min_confidence: float = 0.50
    min_margin: float = 0.05

    def __post_init__(self):
        for name in ("min_coverage", "min_confidence", "min_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if type(self.min_valid_frames) is not int or not 1 <= self.min_valid_frames <= 450:
            raise ValueError("min_valid_frames must be an integer from 1 to 450")

    @classmethod
    def from_env(cls):
        return cls(
            min_coverage=float(os.getenv("SIGNBRIDGE_ISL_MIN_COVERAGE", "0.05")),
            min_valid_frames=int(os.getenv("SIGNBRIDGE_ISL_MIN_VALID_FRAMES", "3")),
            min_confidence=float(os.getenv("SIGNBRIDGE_ISL_MIN_CONFIDENCE", "0.50")),
            min_margin=float(os.getenv("SIGNBRIDGE_ISL_MIN_MARGIN", "0.05")),
        )

    def reject_quality(self, quality: SegmentQuality) -> tuple[str, str] | None:
        if not quality.has_signal or quality.hand_valid_frames < self.min_valid_frames:
            return "insufficient_data", "not enough frames with observed hand landmarks"
        if quality.coordinate_coverage < self.min_coverage:
            return "tracking_lost", "landmark coverage is below the configured minimum"
        return None

    def accepts(self, confidence: float, margin: float) -> bool:
        return confidence >= self.min_confidence and margin >= self.min_margin
