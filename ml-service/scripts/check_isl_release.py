"""B3/B7/B12 release-evidence gate; exits 2 while Person A evidence is missing.

INTEGRATION_WARNING: NEVER substitute synthetic fixture outputs for independent
training exports to make this check green. This script deliberately has no
--skip/--assume-passed mode. See PERSON_B_HANDOFF.md for the evidence schema.
Successful unit tests do not remove the pending state in API responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from service.isl_model import get_isl_engine
from service.isl_preprocessing import PREPROCESSING_VERSION, prepare_segment
from service.isl_quality import QualityPolicy


REQUIRED_FILES = (
    "golden.npz", "provenance.json", "selection.json", "calibration.json", "live_validation.json",
)


def check_release(evidence: Path, engine=None):
    missing = [filename for filename in REQUIRED_FILES if not (evidence / filename).is_file()]
    if missing:
        return {"status": "blocked", "integration_ready": False,
                "blockers": [f"Person A evidence missing: {filename}" for filename in missing]}
    model = engine or get_isl_engine()
    provenance = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
    blockers = []
    if provenance.get("source") != "independent_training_export" or provenance.get("synthetic") is not False:
        blockers.append("golden fixture must be an independent non-synthetic training export")
    if provenance.get("preprocessing_version") != PREPROCESSING_VERSION:
        blockers.append("preprocessing version mismatch")
    if provenance.get("model_sha256") != model.model_sha256 or provenance.get("labels_sha256") != model.labels_sha256:
        blockers.append("golden provenance model/label hash mismatch")
    expected_hash = hashlib.sha256((evidence / "golden.npz").read_bytes()).hexdigest()
    if provenance.get("fixture_sha256") != expected_hash:
        blockers.append("golden fixture hash mismatch")
    columns = provenance.get("feature_columns", [])
    if (not isinstance(columns, list) or len(columns) != 330
            or not all(isinstance(c, str) and c for c in columns) or len(set(columns)) != 330):
        blockers.append("exact ordered 330 unique feature columns required")
    for field in ("dataset_sha256", "sequence_id", "camera_policy", "hand_slot_policy", "face_index_policy"):
        if not provenance.get(field):
            blockers.append(f"missing provenance field: {field}")
    with np.load(evidence / "golden.npz", allow_pickle=False) as fixture:
        raw, expected_features, expected_p = fixture["raw"], fixture["features"], fixture["probabilities"]
        mask = fixture["observed_mask"] if "observed_mask" in fixture else None
    actual = prepare_segment(raw, observed_mask=mask)
    if actual.quality.hand_valid_frames < 3 or not np.any(actual.features):
        blockers.append("golden fixture must contain a usable moving sign, not all-zero/constant tensors")
    if expected_features.shape != (60, 660) or not np.allclose(actual.features, expected_features, atol=1e-6, rtol=1e-5):
        blockers.append("training/serving preprocessing mismatch")
    probabilities = model.predict_probabilities(actual.features[None])[0]
    if expected_p.shape != (50,) or not np.allclose(probabilities, expected_p, atol=1e-5, rtol=1e-4):
        blockers.append("training/serving model probability mismatch")
    for filename in ("selection.json", "calibration.json", "live_validation.json"):
        report = json.loads((evidence / filename).read_text(encoding="utf-8"))
        if report.get("synthetic") is not False or report.get("model_sha256") != model.model_sha256:
            blockers.append(f"{filename}: non-synthetic evidence for selected model required")
        if type(report.get("sample_count")) is not int or report["sample_count"] < 1:
            blockers.append(f"{filename}: positive sample_count required")
        if not isinstance(report.get("metrics"), dict) or not report["metrics"]:
            blockers.append(f"{filename}: measured metrics required")
        if filename == "selection.json" and report.get("split") != "validation":
            blockers.append("choose checkpoint on validation, not test")
        if filename == "calibration.json":
            fields = report.get("policy", {})
            policy = QualityPolicy(**fields)
            if set(fields) != {"min_coverage", "min_valid_frames", "min_confidence", "min_margin"}:
                blockers.append("calibration policy must explicitly specify all four thresholds")
            if policy != QualityPolicy.from_env():
                blockers.append("serving thresholds differ from calibrated policy")
        if filename == "live_validation.json" and report.get("boundary_annotations_verified") is not True:
            blockers.append("annotated live segmentation evidence required")
    return {"status": "blocked" if blockers else "evidence_checks_passed", "integration_ready": not blockers,
            "model_sha256": model.model_sha256, "blockers": blockers,
            "next_step": "Review evidence and update API/manifest pending markers together; this command edits nothing."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("models/person_a_evidence"))
    args = parser.parse_args(argv)
    try:
        report = check_release(args.evidence_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report = {"status": "blocked", "integration_ready": False, "blockers": [str(exc)]}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["integration_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
