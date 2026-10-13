# SignBridge ML service — Person B

> **INTEGRATION_WARNING:** the Keras serving implementation is runnable with the
> committed real weights. The frozen Person A preprocessing contract is integrated;
> checkpoint selection, threshold calibration and live segmentation still need
> official evidence. Passing unit tests does not mean release-ready.

Start with [PERSON_B_HANDOFF.md](PERSON_B_HANDOFF.md) for B0–B13 status, ownership,
integration instructions and the exact evidence files Person A must supply.
The versioned request/response contract is [isl-contract.md](isl-contract.md).

## Local run

Run commands **from `ml-service`** in your virtual environment:

```powershell
python -m pip install -r requirements-isl-dev.txt
python -m uvicorn service.app:app --host 127.0.0.1 --port 8080
```

Endpoints:

| Endpoint | Meaning |
|---|---|
| `GET /health` | Cheap liveness; no TensorFlow import/model load |
| `GET /v2/readiness` | Loads/checks the selected artifact; separately reports `integration_ready: false` |
| `POST /v2/predict` | One raw landmark segment → candidate/status |
| `POST /v2/predict/batch` | 1–10 isolated buffers → ordered results; one real model batch call |
| `POST /predict` | Legacy PyTorch interface; requires its own dependencies and `.pt` checkpoint |

The new Keras image does not bundle the legacy PyTorch runtime/checkpoint.
The legacy source route remains available for existing full-development setups.
`requirements.txt` retains that full extraction/legacy setup; use the smaller
`requirements-isl*.txt` files for this work.

Raw input is `(T,110,3)`, up to 450 raw frames total per capture (30 seconds at
15 FPS). The **model** always receives `(N,60,660)`. The 60 model timesteps are
not a reason to discard a raw recording after 60 frames. The reference segmenter
has a separate conservative 60-frame candidate limit, reported as `overflow`.

## Configuration

| Variable | Default / meaning |
|---|---|
| `SIGNBRIDGE_ISL_MODEL_PATH` | `models/best_isl_model_v2.keras` (provisional selection) |
| `SIGNBRIDGE_ISL_LABEL_PATH` | `models/id_to_label.json` |
| `SIGNBRIDGE_ISL_MANIFEST_PATH` | `models/model_manifest.json` |
| `SIGNBRIDGE_ISL_MIN_COVERAGE` | `0.05`, uncalibrated |
| `SIGNBRIDGE_ISL_MIN_VALID_FRAMES` | `3` hand-visible frames, uncalibrated |
| `SIGNBRIDGE_ISL_MIN_CONFIDENCE` | `0.50`, uncalibrated |
| `SIGNBRIDGE_ISL_MIN_MARGIN` | `0.05`, uncalibrated |
| `SIGNBRIDGE_ALLOWED_ORIGINS` | Empty; comma-separated browser origins when integrating |

Model and label hashes are verified against the manifest. New weights, label
maps and preprocessing versions must be changed as one reviewed bundle. Restart
the service after updating artifacts. Load failure returns 503, not fake words.

## Incremental tests

```powershell
python -m pytest tests/test_isl_model.py -q
python -m pytest tests/test_isl_preprocessing.py -q
python -m pytest tests/test_isl_inference.py tests/test_isl_batch.py -q
python -m pytest tests/test_sign_segmentation.py tests/test_isl_replay.py -q
python -m pytest tests/test_isl_evidence.py -q
python -m pytest tests -q -rs
```

Tests marked `real_model` load actual committed artifacts and fail if TensorFlow
or weights are missing. They use generated input to test execution, not sign
accuracy. Other new tests explicitly identify their synthetic/test-double data.
Legacy MediaPipe/OpenCV tests can skip when extraction dependencies are absent.

## Artifact comparison and release gate

```powershell
python -m scripts.evaluate_isl_models --smoke
python -m scripts.evaluate_isl_models --data validation.npz --provenance validation.json
python -m scripts.check_isl_release --evidence-dir models/person_a_evidence
```

The first command runs both real artifacts on generated tensors and reports no
accuracy. The second needs Person A's independent prepared validation data.
The final command **exits 2 while evidence is missing or mismatched**; this is
intentional and must not be replaced by a dummy passing fixture.

## Offline segmentation/replay

```powershell
python -m scripts.segment_signs recording.npy
python -m scripts.evaluate_segmentation recording.npy --annotations annotations.json --predict --evidence-kind annotated_capture
```

`recording.npy` is a fixed-rate `(T,110,3)` NumPy array, no pickle. Annotations
are a list of `{start_frame, end_frame, keyword?}` objects with inclusive frame
indices. Calibration adapts the motion magnitude threshold; the pause-confirmation
duration is fixed initially. Static signs and long internal holds remain ambiguous.
The browser implementation and UI are a later integration task.

## Container

From `ml-service`:

```powershell
docker build --tag signbridge-ml:isl .
docker run --rm --publish 8080:8080 --env PORT=8080 signbridge-ml:isl
```

The Dockerfile installs the pinned Keras runtime and smoke-checks both actual
artifacts during build. It does not claim the Person A release gate has passed.
Docker/Cloud Run verification must be performed on a host with Docker available.
Provider/backend integration is documented elsewhere and is not part of this build.
