# Frozen ISL v2 model bundle

This directory contains Person A's reproducibility artifacts for the ISL
INCLUDE-50 model. The service runtime must not silently use the legacy PyTorch
M4 checkpoint for this bundle.

## Frozen contract

- Raw frame: 110 points × 3 coordinates = 330 values.
- Point order: `hand_0` 21 points, `hand_1` 21 points, then the 68 ordered
  `compact-v1` face points. The CSV manifest records the exact source column
  names and source face indices.
- Prepared model input: `(60, 660)` float32.
- Feature order: 60 resampled/mean-centred position frames followed by their
  first-order frame differences, concatenated on the final axis.
- Classes: exactly 50, IDs 0–49 from the frozen label maps.
- Missing values: NaN cells are accepted by preprocessing and become zero after
  per-feature `nanmean` centering. Infinite values are rejected.
- Frame order: the caller sorts raw CSV/video frames by `frame_id`; the shared
  preprocessing function preserves the supplied order.

## Artifacts

- `../models/best_isl_model_v2.keras`
- `../models/isl_1dcnn_transformer_v2.keras`
- `../models/id_to_label.json`
- `../models/label_to_id.json`
- `../models/evaluation_metrics.json` (historical notebook metrics)
- `model_manifest.json` and `artifact_checksums.txt`
- `model_selection.json` (selection remains pending until both files are
  runtime-evaluated on the official test sequences)
- `dataset_audit/audit_report.json`
- `dataset_audit/feature_columns.json`
- `dataset_audit/preprocessing_manifest.json`

The manifest records the SHA-256 of the exact CSV and notebook used for the
freeze. It also records that local runtime model loading/evaluation was not
attempted when TensorFlow was unavailable.

## Checks

From `ml-service`:

```powershell
python -m pytest -q
python scripts/freeze_isl_artifacts.py --notebook PATH_TO_TRAIN_NOTEBOOK --csv PATH_TO_COMPACT68_CSV
python scripts/audit_include50.py PATH_TO_COMPACT68_CSV --output-dir model/isl/dataset_audit
python scripts/evaluate_isl.py --csv PATH_TO_COMPACT68_CSV
```

The last command requires the isolated environment described in
`requirements-isl.txt` and a TensorFlow/Keras runtime compatible with Keras
3.13.2. It must evaluate the official test split; it must not fall back to a
random split.

## Scope note

This is Person A's data/model artifact and preprocessing handoff. Keras model
caching, HTTP prediction, abstention thresholds, segmentation, and browser
integration belong to the serving/integration work and are intentionally not
implemented here.
