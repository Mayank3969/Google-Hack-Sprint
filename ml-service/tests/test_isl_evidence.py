"""B7/B12 evidence checker tests; generated data is deliberately SYNTHETIC.

INTEGRATION_WARNING: tests never create a passing release evidence bundle.
Person A must run check_isl_release on independently exported real evidence.
"""

import numpy as np

from scripts.check_isl_release import check_release, main
from scripts.evaluate_isl_models import classification_metrics


def test_missing_evidence_stays_blocked(tmp_path):
    report = check_release(tmp_path)
    assert not report["integration_ready"]
    assert len(report["blockers"]) == 5
    assert main(["--evidence-dir", str(tmp_path)]) == 2


def test_f1_reports_all_classes_and_unsupported_classes_separately():
    y = np.array([0, 0, 1])
    p = np.eye(3)[y]
    report = classification_metrics(y, p, ["a", "b", "c"])
    assert report["accuracy"] == 1
    assert report["macro_f1_all_classes"] == 2 / 3
    assert report["macro_f1_supported_classes"] == 1
    assert report["unsupported_classes"] == ["c"]


def test_confusion_counts_misclassification():
    report = classification_metrics(np.array([0, 1, 1]), np.eye(2)[[0, 0, 1]], ["a", "b"])
    assert report["confusion_matrix"] == [[1, 0], [1, 1]]
    assert report["accuracy"] == 2 / 3
