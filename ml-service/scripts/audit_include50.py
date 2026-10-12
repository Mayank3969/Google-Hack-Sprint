"""Streaming audit for the Include-50 compact-68 landmark CSV.

This diagnostic intentionally processes one CSV row at a time.  It retains
sequence-level counters and frame-id sets, never a pandas DataFrame or the
landmark matrix, so it is suitable for the 1.8GB source file.  It does not
prepare model tensors and does not serve HTTP.

Example::

    python scripts/audit_include50.py dataset/include50.csv --output-dir audit

The report records the chosen sequence, label, split, and frame columns.  The
default choices match the v2 notebook; use ``--sequence-key`` when a source
uses a different explicit key.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.isl_preprocessing import (
    FACE_SOURCE_INDICES,
    FEATURE_COLUMNS,
    OUTPUT_FEATURES,
    PREPROCESSING_VERSION,
    RAW_FEATURES,
    TARGET_FRAMES,
)


SEQUENCE_KEYS = ("sequence_id", "sample_id", "video_name")
FRAME_KEYS = ("frame_id", "frame_idx", "frame")
LABEL_KEYS = ("sign", "official_sign", "word", "gloss", "label", "sign_id")
SPLIT_KEYS = ("split", "subset", "type")
METADATA_COLUMNS = {
    "sign_id", "category", "sign", "official_sign", "class_key", "word", "gloss", "label",
    "sample_id", "video_name", "video_relpath", "sequence_id", "frame_id", "frame_idx", "frame",
    "split", "subset", "type", "signer_id", "session_id", "timestamp", "time", "fps",
}
MISSINGNESS_PREFIX = "missing_"
AXES = {"x": 0, "y": 1, "z": 2}
GROUP_ORDER = {"hand_0": 0, "hand_1": 1, "face": 2}


def _column_choice(headers: list[str], choices: Iterable[str], explicit: str | None = None) -> str | None:
    by_lower = {name.lower(): name for name in headers}
    if explicit is not None:
        if explicit not in headers:
            raise ValueError(f"column {explicit!r} is not present in the CSV header")
        return explicit
    return next((by_lower[name] for name in choices if name in by_lower), None)


def _descriptor(column: str) -> tuple[str, int, int] | None:
    """Return ``(semantic_group, point_index, axis_index)`` for known names.

    Include-50 exports have appeared with both ``hand_0_12_x`` and
    ``hand_0_x_12`` naming.  The global ``landmark_12_x``/``x_12`` forms are
    also accepted.  All forms map to the same required hand-0, hand-1, face
    order, while preserving the header spelling in the emitted manifest.
    """
    name = column.strip().lower()
    patterns = (
        (r"^hand[_-]?([01])[_-]?(\d+)[_-]?([xyz])$", "hand-first"),
        (r"^hand[_-]?([01])[_-]?([xyz])[_-]?(\d+)$", "axis-first"),
        (r"^face[_-]?(\d+)[_-]?([xyz])$", "face-first"),
        (r"^face[_-]?([xyz])[_-]?(\d+)$", "face-axis-first"),
        (r"^(?:landmark|point)[_-]?(\d+)[_-]?([xyz])$", "global-first"),
        (r"^(?:landmark|point)[_-]?([xyz])[_-]?(\d+)$", "global-axis-first"),
        (r"^([xyz])[_-]?(\d+)$", "global-axis-first"),
        (r"^(\d+)[_-]?([xyz])$", "global-first"),
    )
    for pattern, kind in patterns:
        match = re.match(pattern, name)
        if not match:
            continue
        groups = match.groups()
        if kind == "hand-first":
            hand, point, axis = int(groups[0]), int(groups[1]), groups[2]
            return (f"hand_{hand}", point, AXES[axis]) if point < 21 else None
        if kind == "axis-first":
            hand, axis, point = int(groups[0]), groups[1], int(groups[2])
            return (f"hand_{hand}", point, AXES[axis]) if point < 21 else None
        if kind == "face-first":
            source_index, axis = int(groups[0]), groups[1]
            if source_index not in FACE_SOURCE_INDICES:
                return None
            return ("face", FACE_SOURCE_INDICES.index(source_index), AXES[axis])
        if kind == "face-axis-first":
            axis, source_index = groups[0], int(groups[1])
            if source_index not in FACE_SOURCE_INDICES:
                return None
            return ("face", FACE_SOURCE_INDICES.index(source_index), AXES[axis])
        if kind == "global-first":
            point, axis = int(groups[0]), groups[1]
        else:
            axis, point = groups[0], int(groups[1])
        if point < 21:
            return ("hand_0", point, AXES[axis])
        if point < 42:
            return ("hand_1", point - 21, AXES[axis])
        return ("face", point - 42, AXES[axis]) if point < 110 else None
    return None


def _ordered_coordinate_columns(
    headers: list[str], selected_metadata: set[str]
) -> tuple[list[str], list[tuple[str, int, int]], list[str]]:
    # The notebook identifies landmarks as numeric, non-metadata columns.  A
    # header-only streaming audit cannot infer pandas dtypes without reading a
    # data row, so use the same metadata exclusion and validate numeric values
    # while streaming below.  Recognized semantic names get an additional
    # hand/face order check; opaque names retain the trained header order.
    coordinate_columns = [
        name for name in headers
        if name.lower() not in METADATA_COLUMNS
        and not name.lower().startswith(MISSINGNESS_PREFIX)
        and name not in selected_metadata
    ]
    named_descriptors = [_descriptor(name) for name in coordinate_columns]
    expected = [
        (group, point, axis)
        for group in ("hand_0", "hand_1", "face")
        for point in range(21 if group != "face" else 68)
        for axis in range(3)
    ]
    order_errors: list[str] = []
    if len(coordinate_columns) != RAW_FEATURES:
        order_errors.append(f"expected exactly {RAW_FEATURES} named coordinate columns; found {len(coordinate_columns)}")
    if all(descriptor is None for descriptor in named_descriptors):
        # Opaque columns are valid under the notebook's numeric-column
        # contract. Their header order is the trained feature order.
        return coordinate_columns, expected[:len(coordinate_columns)], order_errors
    if any(descriptor is None for descriptor in named_descriptors):
        order_errors.append("coordinate columns use a partially unrecognized naming scheme")
        return coordinate_columns, expected[:len(coordinate_columns)], order_errors
    coordinate_descriptors = [descriptor for descriptor in named_descriptors if descriptor is not None]
    if len(set(coordinate_descriptors)) != len(coordinate_descriptors):
        order_errors.append("coordinate columns contain duplicate point/axis descriptors")
    if coordinate_descriptors != expected:
        for index, (actual, wanted) in enumerate(zip(coordinate_descriptors, expected)):
            if actual != wanted:
                order_errors.append(f"coordinate order mismatch at index {index}: expected {wanted}, found {actual}")
                break
        if len(coordinate_descriptors) < len(expected):
            order_errors.append("coordinate columns do not cover all 110 points")
        elif len(coordinate_descriptors) > len(expected):
            order_errors.append("coordinate columns contain extra point/axis descriptors")
    return coordinate_columns, coordinate_descriptors, order_errors


def _parse_number(value: str) -> tuple[float | None, str | None]:
    text = value.strip()
    if not text or text.lower() in {"nan", "na", "null", "none"}:
        return None, "missing"
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None, "non_numeric"
    if not math.isfinite(number):
        return None, "nonfinite"
    return number, None


def _parse_frame_id(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


class _SequenceState:
    def __init__(self) -> None:
        self.labels: set[str] = set()
        self.splits: set[str] = set()
        self.frame_ids: set[int] = set()
        self.previous_frame: int | None = None
        self.duplicate_frames = 0
        self.nonmonotonic_frames = 0
        self.invalid_frame_ids = 0
        self.rows = 0
        self.missing_cells = Counter({"hand_0": 0, "hand_1": 0, "face": 0, "coordinate": 0, "entire_frame": 0})
        self.non_numeric_cells = 0
        self.nonfinite_cells = 0


def _safe_json(value: object) -> object:
    """Convert counters/defaultdicts to JSON-safe ordinary containers."""
    if isinstance(value, Counter):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, defaultdict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    return value


def audit_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    sequence_key: str | None = None,
    label_key: str | None = None,
    split_key: str | None = None,
    frame_key: str | None = None,
    expected_sequences: Iterable[str] | None = None,
    example_limit: int = 20,
) -> dict:
    """Audit without retaining coordinate rows; write three JSON artifacts.

    ``expected_sequences`` optionally declares the sequence inventory. A frame
    CSV alone cannot reveal sequences which have no rows. A fully missing frame
    is not an empty sequence; these are reported separately. Frame-id gaps are
    measured only between the observed minimum and maximum (no assumed origin).
    The sequence state is O(sequences + unique frame IDs), not O(coordinates).
    """
    if example_limit < 0:
        raise ValueError("example_limit must be nonnegative")
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    states: dict[str, _SequenceState] = {}
    rows_by_class: Counter = Counter()
    rows_by_split: Counter = Counter()
    rows_by_class_split: dict[str, Counter] = defaultdict(Counter)
    missing = Counter({"hand_0": 0, "hand_1": 0, "face": 0, "coordinate": 0, "entire_frame": 0})
    missing_by_axis = Counter({axis: 0 for axis in AXES})
    missing_by_column: Counter = Counter()
    entirely_missing_blocks = Counter({group: 0 for group in GROUP_ORDER})
    missing_sequence_rows = missing_label_rows = missing_split_rows = 0
    non_numeric_cells = nonfinite_cells = malformed_rows = rows = 0
    errors: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if not headers:
            raise ValueError("CSV is empty or has no header")
        if len(set(headers)) != len(headers):
            errors.append("CSV header contains duplicate column names")
        sequence_key = _column_choice(headers, SEQUENCE_KEYS, sequence_key)
        label_key = _column_choice(headers, LABEL_KEYS, label_key)
        split_key = _column_choice(headers, SPLIT_KEYS, split_key)
        frame_key = _column_choice(headers, FRAME_KEYS, frame_key)
        if sequence_key is None or label_key is None:
            raise ValueError("CSV must have a sequence grouping column and a label column (or explicit keys)")
        if frame_key is None:
            errors.append("no frame ordering column; duplicate pairs, ordering, and gaps cannot be audited")
        selected_metadata = {sequence_key, label_key, split_key, frame_key}
        columns, descriptors, order_errors = _ordered_coordinate_columns(headers, selected_metadata)
        errors.extend(order_errors)
        unknown_columns = [name for name in headers if name not in columns
                           and name.lower() not in METADATA_COLUMNS
                           and not name.lower().startswith(MISSINGNESS_PREFIX)
                           and name not in selected_metadata]
        if unknown_columns:
            errors.append("unrecognized columns; coordinate schema cannot be certified")
        indexes = {name: index for index, name in enumerate(headers)}
        feature_indexes = [indexes[name] for name in columns]
        axis_names = ("x", "y", "z")
        group_sizes = Counter(descriptor[0] for descriptor in descriptors)
        for row in reader:
            # An empty physical line has no frame fields and is not a sample.
            if not row:
                continue
            rows += 1
            if len(row) != len(headers):
                malformed_rows += 1
                row = (row + [""] * len(headers))[:len(headers)]
            sequence_id = row[indexes[sequence_key]].strip()
            label = row[indexes[label_key]].strip()
            split = row[indexes[split_key]].strip() if split_key else "<unspecified>"
            missing_sequence_rows += not bool(sequence_id)
            missing_label_rows += not bool(label)
            missing_split_rows += bool(split_key) and not bool(split)
            rows_by_class[label] += 1
            rows_by_split[split] += 1
            rows_by_class_split[label][split] += 1
            # Missing keys must not be merged into a fabricated sequence.
            state = states.setdefault(sequence_id, _SequenceState()) if sequence_id else None
            if state is not None:
                state.rows += 1
                state.labels.add(label)
                state.splits.add(split)
                if frame_key:
                    frame_id = _parse_frame_id(row[indexes[frame_key]])
                    if frame_id is None:
                        state.invalid_frame_ids += 1
                    else:
                        if frame_id in state.frame_ids:
                            state.duplicate_frames += 1
                        if state.previous_frame is not None and frame_id < state.previous_frame:
                            state.nonmonotonic_frames += 1
                        state.frame_ids.add(frame_id)
                        state.previous_frame = frame_id
            row_missing: Counter = Counter()
            for name, index, (group, _, axis) in zip(columns, feature_indexes, descriptors):
                _, issue = _parse_number(row[index])
                if issue == "missing":
                    missing[group] += 1
                    missing["coordinate"] += 1
                    missing_by_axis[axis_names[axis]] += 1
                    missing_by_column[name] += 1
                    row_missing[group] += 1
                    if state:
                        state.missing_cells[group] += 1
                        state.missing_cells["coordinate"] += 1
                elif issue == "non_numeric":
                    non_numeric_cells += 1
                    if state:
                        state.non_numeric_cells += 1
                elif issue == "nonfinite":
                    nonfinite_cells += 1
                    if state:
                        state.nonfinite_cells += 1
            for group in GROUP_ORDER:
                if group_sizes[group] and row_missing[group] == group_sizes[group]:
                    entirely_missing_blocks[group] += 1
            if len(columns) == RAW_FEATURES and sum(row_missing.values()) == RAW_FEATURES:
                missing["entire_frame"] += 1
                if state:
                    state.missing_cells["entire_frame"] += 1

    sequences_by_class: Counter = Counter()
    sequences_by_split: Counter = Counter()
    sequences_by_class_split: dict[str, Counter] = defaultdict(Counter)
    issues = Counter({name: 0 for name in (
        "inconsistent_labels", "inconsistent_splits", "duplicate_frame_rows",
        "nonmonotonic_frame_transitions", "invalid_or_missing_frame_id_rows",
        "missing_frame_ids", "sequences_with_frame_gaps", "all_coordinates_missing_sequences",
    )})
    examples: dict[str, list] = defaultdict(list)

    def example(kind: str, data: dict) -> None:
        if len(examples[kind]) < example_limit:
            examples[kind].append(data)

    for sequence_id, state in states.items():
        if len(state.labels) != 1:
            issues["inconsistent_labels"] += 1
            example("inconsistent_labels", {"sequence_id": sequence_id, "labels": sorted(state.labels)})
        else:
            sequences_by_class[next(iter(state.labels))] += 1
        if len(state.splits) != 1:
            issues["inconsistent_splits"] += 1
            example("inconsistent_splits", {"sequence_id": sequence_id, "splits": sorted(state.splits)})
        else:
            sequences_by_split[next(iter(state.splits))] += 1
        if len(state.labels) == len(state.splits) == 1:
            sequences_by_class_split[next(iter(state.labels))][next(iter(state.splits))] += 1
        issues["duplicate_frame_rows"] += state.duplicate_frames
        issues["nonmonotonic_frame_transitions"] += state.nonmonotonic_frames
        issues["invalid_or_missing_frame_id_rows"] += state.invalid_frame_ids
        if state.duplicate_frames or state.nonmonotonic_frames or state.invalid_frame_ids:
            example("frame_id_issues", {"sequence_id": sequence_id, "duplicate_rows": state.duplicate_frames,
                    "nonmonotonic_transitions": state.nonmonotonic_frames, "invalid_rows": state.invalid_frame_ids})
        if state.frame_ids:
            gap_count = max(state.frame_ids) - min(state.frame_ids) + 1 - len(state.frame_ids)
            issues["missing_frame_ids"] += gap_count
            if gap_count:
                issues["sequences_with_frame_gaps"] += 1
                example("frame_gaps", {"sequence_id": sequence_id, "missing_count": gap_count,
                        "observed_min": min(state.frame_ids), "observed_max": max(state.frame_ids)})
        if len(columns) == RAW_FEATURES and state.missing_cells["coordinate"] == state.rows * RAW_FEATURES:
            issues["all_coordinates_missing_sequences"] += 1
            example("all_coordinates_missing", {"sequence_id": sequence_id, "rows": state.rows})

    empty_count = 0 if expected_sequences is not None else None
    if expected_sequences is not None:
        # Retain only the IDs seen in the inventory for deduplication, not frames.
        seen_inventory: set[str] = set()
        for sequence_id in expected_sequences:
            sequence_id = str(sequence_id).strip()
            if not sequence_id or sequence_id in seen_inventory:
                continue
            seen_inventory.add(sequence_id)
            if sequence_id not in states:
                empty_count += 1
                example("empty_sequences", {"sequence_id": sequence_id})

    if not rows:
        errors.append("CSV contains no frame rows")
    data_errors = (non_numeric_cells + nonfinite_cells + malformed_rows + missing_sequence_rows
                   + missing_label_rows + missing_split_rows + sum(issues.values()) + (empty_count or 0))
    report = _safe_json({
        "audit_version": "include50-audit-v2", "preprocessing_version": PREPROCESSING_VERSION,
        "source": str(csv_path), "reader": "csv.reader (one row at a time)",
        "memory_policy": "coordinates: O(330); metadata: O(sequences + unique frame IDs)",
        "chosen_columns": {"sequence": sequence_key, "label": label_key, "split": split_key, "frame": frame_key},
        "schema": {"valid": not errors, "coordinate_count": len(columns),
                   "coordinate_order_valid": not order_errors,
                   "coordinate_order_policy": "semantic hand_0/hand_1/face descriptors when named; otherwise CSV header order matches the notebook's numeric-column order",
                   "errors": errors, "unrecognized_columns": unknown_columns},
        "counts": {"rows": rows, "sequences": len(states), "rows_by_class": rows_by_class,
                   "rows_by_split": rows_by_split, "rows_by_class_split": rows_by_class_split,
                   "sequences_by_class": sequences_by_class, "sequences_by_split": sequences_by_split,
                   "sequences_by_class_split": sequences_by_class_split},
        "missing": {"cells_by_group": missing, "cells_by_axis": missing_by_axis,
                    "cells_by_column": {column: missing_by_column[column] for column in columns},
                    "entirely_missing_block_frames": entirely_missing_blocks,
                    "sequence_key_rows": missing_sequence_rows, "label_rows": missing_label_rows,
                    "split_rows": missing_split_rows},
        "invalid": {"non_numeric_cells": non_numeric_cells, "nonfinite_cells": nonfinite_cells,
                    "malformed_rows": malformed_rows},
        "sequence_issues": issues,
        "empty_sequences": {"count": empty_count, "inventory_provided": expected_sequences is not None,
                            "policy": "zero CSV rows; not inferable without an expected sequence inventory"},
        "frame_id_policy": "integer IDs; gaps within observed min/max; nonmonotonic means decreasing in source row order",
        "sequence_count_policy": "inconsistent metadata excluded from the affected sequence count; row counts include all rows",
        "missing_policy": "blank/NaN/NA/null/None are missing; infinity/non-numeric are invalid; zero is a valid coordinate",
        "examples": examples, "example_limit": example_limit,
        "passed": not errors and not data_errors,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "audit_report.json": report,
        "feature_columns.json": columns,
        "preprocessing_manifest.json": {
            "version": PREPROCESSING_VERSION,
            "reference_notebook": "notebooks/Train_ISL_1DCNN_Transformer_compact68_v2.ipynb",
            "raw_shapes": [["T", 110, 3], ["T", RAW_FEATURES]],
            "output_shape": [TARGET_FRAMES, OUTPUT_FEATURES], "dtype": "float32",
            "coordinate_order": "hand_0[0:21], hand_1[0:21], compact face[0:68]; xyz per point",
            "feature_columns_file": "feature_columns.json", "feature_schema_valid": not errors,
            "resampling": "np.linspace(0, n_frames - 1, 60).astype(int); duplicates allowed",
            "normalization": "float32 before resampling; per-feature nanmean over resampled frames; missing values -> 0",
            "velocity": "np.diff(seq_norm, axis=0, prepend=seq_norm[:1, ...])",
            "concatenation": "[position, velocity]", "nan_policy": "allowed and mapped to zero after centering",
            "infinity_policy": "reject", "comparison_tolerance": {"atol": 1e-6, "rtol": 1e-6},
            "frame_order_policy": "caller sorts by chosen frame key before prepare_model_input; function preserves supplied order",
            "chosen_columns": report["chosen_columns"],
        },
    }
    for filename, artifact in artifacts.items():
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, nargs="?", help="input CSV (also accepted as --csv)")
    parser.add_argument("--csv", "--input", dest="csv_option", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-key")
    parser.add_argument("--label-key")
    parser.add_argument("--split-key")
    parser.add_argument("--frame-key")
    parser.add_argument("--expected-sequences", type=Path, help="UTF-8 text inventory, one sequence ID per line")
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args(argv)
    csv_path = args.csv_option or args.csv_path
    if csv_path is None:
        parser.error("an input CSV path is required (positional or --csv)")
    try:
        options = {"sequence_key": args.sequence_key, "label_key": args.label_key,
                   "split_key": args.split_key, "frame_key": args.frame_key, "example_limit": args.example_limit}
        if args.expected_sequences:
            with args.expected_sequences.open("r", encoding="utf-8-sig") as inventory:
                report = audit_csv(csv_path, args.output_dir, expected_sequences=inventory, **options)
        else:
            report = audit_csv(csv_path, args.output_dir, **options)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"audit error: {exc}\n")
    print(f"Audit {'passed' if report['passed'] else 'found issues'}: {report['counts']['rows']} rows, "
          f"{report['counts']['sequences']} sequences; {args.output_dir / 'audit_report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
