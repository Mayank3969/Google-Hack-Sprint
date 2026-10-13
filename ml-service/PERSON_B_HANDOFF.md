# Person B implementation and Person A merge handoff

**INTEGRATION_WARNING — read before changing readiness flags.**

This implementation uses the two committed REAL `.keras` artifacts and the
integrated frozen `isl-preprocessing-v2` module. Test doubles exist only in tests.
Official runtime reevaluation, calibrated rejection, and real-camera/multi-sign
validation are **pending**, not implied by a green test suite.

## Goal and scope

Receive the browser's ordered isolated-sign buffers after Stop, classify each,
return words/confidence/status in order. Raw `(T,110,3)` becomes `(60,660)` per
segment. ISL INCLUDE-50, Keras 1D-CNN/Transformer; not the legacy ASL/PyTorch model.
The model remains server-side. Segmentation code here is a Python offline
reference to be ported by the frontend team, not working browser camera code.

No backend provider, UI, speech, or final application integration is implemented
in this scope. Team members can work on those independently using `isl-contract.md`.

## B0–B13 task files and evidence status

| Task | Files/deliverable | Current boundary |
|---|---|---|
| B0 ownership | This handoff; `isl-contract.md` | A takes over preprocessing seam after pull |
| B1 Keras runtime | `service/isl_layers.py`, `service/isl_model.py` | Both real artifacts load; custom objects, labels, shapes, hashes checked |
| B2 dependencies | `requirements-isl.txt`, `requirements-isl-dev.txt` | Keras/TF serving pins separate from extraction/legacy dependencies |
| B3 preprocessing | `service/isl_preprocessing.py`, `models/preprocessing_manifest.json` | Provisional real transform, no dummy; A golden parity pending |
| B4 single inference | `service/isl_inference.py`, `/v2/predict` | Real model execution and per-slot status |
| B5 batching | `/v2/predict/batch`, `service/body_limit.py` | One model call, ordered slots, no partial keyword phrase |
| B6 quality | `service/isl_quality.py` | Implemented configurable policy; calibration pending |
| B7 model comparison | `scripts/evaluate_isl_models.py`, model manifest | Smoke checks real weights; held-out selection pending data |
| B8 versioned API | `service/app.py`, `isl-contract.md` | Legacy path separate; readiness/warnings explicit |
| B9 segmenter | `service/segmentation.py`, `scripts/segment_signs.py` | Fixed-rate offline state machine; real threshold validation pending |
| B10 replay | `scripts/evaluate_segmentation.py` | Segments → preprocessing → batch inference, optional annotated metrics |
| B11 packaging | `Dockerfile`, serving requirements | Keras image recipe; Docker verification needs Docker host |
| B12 verification | `tests/test_isl_*.py`, `tests/test_sign_segmentation.py`, release-check CLI | Synthetic unit vs real-artifact execution vs release evidence separated |
| B13 documentation | README, contract, handoff, `PERSON_B_VERIFICATION.md` | Commands, warnings, dependencies and actual checks recorded |

## File ownership for parallel work

After pulling B's changes, **Person A owns** the implementation behind
`service/isl_preprocessing.py`, the verified feature schema, model/label manifests,
independent fixtures, retraining/evaluation data and calibration recommendations.

**Person B owns** the API, runtime model loader/custom-layer serialization,
inference orchestration, HTTP limits, offline segmentation/replay, packaging and
associated tests/docs. A coordinates architecture changes with B instead of
independently editing `isl_layers.py`. A supplies selected weights/checksums;
B updates runtime selection and packaging. Both should use separate checkouts.

## Exact A → B preprocessing seam

Preserve these public names when integrating (or update all callers together):

```python
prepare_segment(landmarks, *, observed_mask=None) -> PreparedSegment
prepare_model_input(landmarks, *, observed_mask=None) -> tuple[np.ndarray, SegmentQuality]
PREPROCESSING_VERSION: str
```

`PreparedSegment.features`: finite float32 `(60,660)`.
`PreparedSegment.quality`: input_frames, valid_frames, hand_valid_frames,
coordinate_coverage, left_hand_coverage, right_hand_coverage, face_coverage,
has_signal. Quality is computed from raw observation masks before normalization.
`PreparedSegment.preprocessing_version` must match the selected model manifest.

Current compatibility rules:
1. Raw hand/face absence is JSON null, NumPy NaN or false boolean mask entries.
2. If no mask is supplied, whole-zero modality blocks become NaN. Individual
   zero coordinates are valid. This detector adapter still needs A confirmation.
3. Full-sequence integer index selection to 60 frames.
4. Per-coordinate temporal nanmean centering, missing normalized values zero.
5. First-order differences AFTER centering; concatenate 330+330.

The v2 inference guard also rejects buffers with no temporal variation after
this transform. Revisit that guard if A trains a static-pose-aware representation.

Do not replace this with wrist normalization, interpolation, missing-transition
velocity suppression or mask input features without a new training/evaluation
decision. Do not apply both A preprocessing and B preprocessing to one request.

### Important shape correction

60 is the model's resampled length, not a raw capture limit. The v2 service
allows up to 450 raw frames (also 450 total per batch). Long sequences are
resampled in full. The offline segmenter independently uses a conservative
60-frame candidate bound and explicitly marks overflow rather than inventing
another word. Person A/frontend can tune that bound with recorded evidence.

## Missing dependencies Person A supplies

- Exact compact68 CSV or verified feature extraction code and ordered 330-column
  list. `include50_official_mediapipe_with_split.csv` is not a verified substitute.
- What `hand_0/hand_1` mean: left/right anatomy or detection order, including
  source-video mirroring and coordinate semantics.
- Exact 68 face indices/order and 468/478 source model convention.
- Independently produced raw → features → probabilities fixture(s).
- Validation-split comparison of both model artifacts; freeze selected checkpoint
  before final test evaluation. Do not pick the better test score repeatedly.
- Confidence/margin/coverage calibration including no-sign/unknown/partial clips.
- Annotated multi-sign recordings with pauses, repeated signs, static holds and
  tracking loss, preferably more than one signer/device.

Notebook metrics (91.6667% accuracy, 88.9039% macro F1) remain notebook-reported.
They are not a new score for both files or an end-to-end live recognition score.
The notebook restores best val-loss weights for the final file but checkpoints
best val-accuracy separately. Both files have different SHA-256 hashes.

## No-dummy release gate

Run from `ml-service`:

```powershell
python -m scripts.check_isl_release --evidence-dir models/person_a_evidence
```

Currently exits **2** and lists missing evidence. This is an expected blocker,
not a test to delete/skip. Supply these files:

### `golden.npz`

Export **from A's training/evaluation pipeline**, not this serving module:
- `raw`: one representative moving-sign sequence `(T,110,3)`.
- `observed_mask`: optional strictly boolean same shape (recommended).
- `features`: independently produced `(60,660)` float32.
- `probabilities`: selected artifact's independently produced `(50,)` output.

Use `allow_pickle=False` compatible arrays. Start with one nontrivial fixture;
add fixtures for missing hands/face, short/long sequences and camera parity
before claiming broad robustness. One matching fixture is necessary, not
sufficient recognition-quality evidence.

### `provenance.json`

Required keys:
- `source: "independent_training_export"`, `synthetic: false`.
- `preprocessing_version`, `model_sha256`, `labels_sha256`, `fixture_sha256`.
- `feature_columns`: exact ordered list of 330 unique strings.
- `dataset_sha256`, `sequence_id`.
- `camera_policy`, `hand_slot_policy`, `face_index_policy`: explicit documented
  conventions, not guesses. B's current schema does not prove these fields.

### `selection.json`

`synthetic: false`, selected `model_sha256`, positive `sample_count`,
`split: "validation"`, and nonempty measured `metrics` object. Retain both
candidate comparison reports alongside it. The gate does not choose a model.
Set `SIGNBRIDGE_ISL_MODEL_PATH` (or update `DEFAULT_MODEL_PATH` with B) for the
chosen file. Editing the manifest's descriptive `model_file` alone does not
switch the serving process. Restart the process after changing the selection.

### `calibration.json`

`synthetic: false`, selected `model_sha256`, positive `sample_count`, nonempty
`metrics`, and `policy` containing `min_coverage`, `min_valid_frames`,
`min_confidence`, `min_margin`. Serving environment/defaults must match the
calibrated policy. Include false-accept/rejection results, not only accuracy.

### `live_validation.json`

`synthetic: false`, selected `model_sha256`, positive `sample_count`, nonempty
`metrics`, and `boundary_annotations_verified: true`. Supply the actual
annotated-session evaluation reports; the gate validates evidence structure and
parity, not the truth of user-entered metric claims or product acceptance targets.

After all checks pass, review the evidence, then update `integration_state()` in
`service/isl_inference.py` and both manifests in the SAME integration commit.
Do not flip readiness because synthetic tests passed or because one fixture is
recognized. The checker never edits files or clears pending flags automatically.

## B7 prepared-split evaluation format

`validation.npz`: `X` finite `(N,60,660)` and integer `y` `(N,)`.
`validation.json`: `split: "validation"`, `synthetic: false`, `dataset_sha256`,
`labels_sha256`, `preprocessing_version`. For final testing, use `split: "test"`.

```powershell
python -m scripts.evaluate_isl_models --data validation.npz --provenance validation.json
```

Outputs both artifact hashes, times, confusion matrices, per-label support and
macro-F1 across all 50 versus supported classes. Empty-support classes remain
visible. `--smoke` has no accuracy output and never selects a checkpoint.

## Final merge sequence

1. A pulls B's code; run current tests as a baseline.
2. Confirm raw schema and camera policies; adapt only the preprocessing seam.
3. Export independent fixtures and compare features AND probabilities.
4. If semantics change, version/retrain and update weights, labels, manifest,
   custom layers (with B), thresholds and tests as one compatible bundle.
5. Evaluate validation split and unknown/partial/tracking-loss recordings.
6. Run annotated multi-sign replay and real browser landmark parity tests.
7. Run unit/real-artifact tests AND release gate; inspect both results separately.
8. Review evidence and clear only the warnings actually resolved; frontend
   integration then uses ordered results and repeat flow before backend calls.

No commits or pushes are performed by these scripts. The user chooses what to
stage/push; do not include unrelated attachments, CSVs or local documents.
