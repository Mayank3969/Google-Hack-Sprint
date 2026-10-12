"""Focused v2 preprocessing and streaming-audit checks."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from service.isl_preprocessing import OUTPUT_SHAPE, PREPROCESSING_VERSION, prepare_model_input


def test_prepare_model_input_resamples_with_integer_duplicate_indices() -> None:
    raw = np.zeros((2, 110, 3), dtype=np.float32)
    raw[1, 0, 0] = 2.0
    output = prepare_model_input(raw)

    assert output.shape == OUTPUT_SHAPE == (60, 660)
    assert output.dtype == np.float32
    indices = np.linspace(0, 1, 60).astype(int)
    positions = raw.reshape(2, 330)[indices, 0]
    expected = positions - positions.mean()
    np.testing.assert_allclose(output[:, 0], expected)
    np.testing.assert_allclose(output[1, 330], 0.0)  # duplicate source frame
    np.testing.assert_allclose(output[indices.tolist().index(1), 330], expected[indices.tolist().index(1)] - expected[indices.tolist().index(0)])


def test_prepare_model_input_accepts_flattened_nan_and_zeroes_missing() -> None:
    raw = np.full((3, 330), np.nan, dtype=np.float32)
    raw[:, 1] = [1.0, 3.0, 5.0]
    output = prepare_model_input(raw)

    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    # An entirely missing feature uses the notebook's false branch.
    np.testing.assert_allclose(output[:, 0], 0.0)
    assert PREPROCESSING_VERSION == "isl-preprocessing-v2"


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_prepare_model_input_rejects_infinity(value: float) -> None:
    raw = np.zeros((1, 330), dtype=np.float32)
    raw[0, 0] = value
    with pytest.raises(ValueError, match="infinite"):
        prepare_model_input(raw)


def test_prepare_model_input_rejects_wrong_shape_and_object_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        prepare_model_input(np.zeros((2, 109, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="numeric"):
        prepare_model_input([["not-a-number"] * 330])


def test_golden_fixture_matches_notebook_contract() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "isl_preprocessing" / "golden_case.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    raw = np.zeros(tuple(fixture["raw_shape"]), dtype=np.float32)
    for key, value in fixture["assignments"].items():
        row, column = (int(part) for part in key.split(","))
        raw[row, column] = value
    output = prepare_model_input(raw)
    assert list(output.shape) == fixture["expected_output_shape"]
    assert str(output.dtype) == fixture["expected_dtype"]
    assert hashlib.sha256(output.tobytes(order="C")).hexdigest() == fixture["expected_sha256_c_order"]


def _coordinate_columns() -> list[str]:
    return [
        f"{group}_{point}_{axis}"
        for group, points in (("hand_0", 21), ("hand_1", 21), ("face", 68))
        for point in range(points)
        for axis in "xyz"
    ]


def test_audit_streams_rows_and_reports_gaps_missing_cells(tmp_path) -> None:
    audit = importlib.import_module("scripts.audit_include50")
    csv_path = tmp_path / "tiny.csv"
    output = tmp_path / "audit"
    columns = ["sequence_id", "frame_id", "sign", "split"] + _coordinate_columns()
    base = ["s1", "0", "hello", "train"] + ["0"] * 330
    second = ["s1", "2", "hello", "train"] + ["0"] * 330
    second[4] = ""
    other = ["s2", "0", "bye", "test"] + ["0"] * 330
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow(base)
        writer.writerow(second)
        writer.writerow(other)

    report = audit.audit_csv(csv_path, output)
    assert report["reader"].startswith("csv.reader")
    assert report["counts"]["rows"] == 3
    assert report["counts"]["sequences"] == 2
    assert report["schema"]["coordinate_count"] == 330
    assert report["missing"]["cells_by_group"]["hand_0"] == 1
    assert report["missing"]["cells_by_group"]["hand_0"] == report["missing"]["cells_by_column"]["hand_0_0_x"]
    assert report["sequence_issues"]["missing_frame_ids"] == 1
    assert report["sequence_issues"]["sequences_with_frame_gaps"] == 1
    assert not report["passed"]
    assert (output / "audit_report.json").is_file()
    assert json.loads((output / "feature_columns.json").read_text()) == _coordinate_columns()
    assert json.loads((output / "preprocessing_manifest.json").read_text())["version"] == PREPROCESSING_VERSION
