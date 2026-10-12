# Person A — ISL model/data status

## Completed in this workspace

- Frozen both `.keras` artifacts, label maps, historical metrics, the exact
  v2 notebook (copied to `notebooks/Train_ISL_1DCNN_Transformer_compact68_v2_exact.ipynb`), and the exact compact68 CSV by SHA-256 in `model_manifest.json`
  and `artifact_checksums.txt`.
- Static graph checks confirm both Keras archives declare
  `(None, 60, 660) -> (None, 50)` with softmax output.
- The compact CSV audit passed:
  - 60,897 frame rows;
  - 958 `sequence_id` sequences;
  - 689 train, 77 validation, 192 test sequences;
  - 50 labels;
  - 330 coordinate columns in hand_0, hand_1, compact-v1 face order;
  - no inconsistent labels/splits, duplicate frame pairs, frame gaps, or
    nonmonotonic frame transitions.
- Missingness is measured rather than filtered:
  - hand_0 missing coordinate cells: 734,328;
  - hand_1 missing coordinate cells: 1,026,144;
  - face missing coordinate cells: 2,448;
  - entirely missing coordinate frames: 1.
- `service/isl_preprocessing.py` reproduces the v2 notebook's resampling,
  per-feature `nanmean` centering, NaN-to-zero policy, velocity, and
  position-plus-velocity concatenation.
- A golden fixture and 73-project-test pass protect the preprocessing and
  artifact checks.
- A deterministic missing-data corruption matrix generator is available at
  `scripts/missing_data_experiment.py`.

## Runtime limitation

This Windows development interpreter does not have TensorFlow, Keras,
MediaPipe, or OpenCV installed. Therefore local runtime model loading, the
two-model evaluation, and the missing-data accuracy experiment are not claimed
as passed. Run them in the isolated environment described by
`requirements-isl.txt` (Colab/Linux with TensorFlow 2.21.0 and Keras 3.13.2).

The current historical `evaluation_metrics.json` is preserved as provenance;
it is not treated as a fresh reproduction of both artifacts.

## MediaPipe video extraction

The repository already contains MediaPipe Tasks extraction code in
`scripts/extract_landmarks_module.py`, and `requirements.txt` pins MediaPipe
and OpenCV. The current interpreter lacks those packages. The hand landmarker
and face landmarker assets are present in `ml-service/models/`; the extraction
code requires both assets explicitly and emits raw `(T, 110, 3)` compact-v1
frames. The supplied `signbridge_wlasl_12k_colab_v2.ipynb` remains the resumable
Colab fallback for bulk video-to-landmark extraction.

## Still requiring a TensorFlow runtime

1. Load both models with their custom Keras layers.
2. Evaluate both on the official CSV test sequences using fixed labels 0–49.
3. Select the artifact using those reproduced metrics and write the selection
   into the model manifest.
4. Run the missing-data robustness matrix and decide whether retraining is
   justified.

Serving, HTTP inference, abstention thresholds, segmentation, and browser
integration are intentionally outside Person A's scope.
