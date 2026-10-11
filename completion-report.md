# SignBridge Completion Report

Generated after the backend/ML integration pass. This report separates code
that exists and has automated evidence from external gates that could not be
verified in this workspace.

## Executive status

The repository now contains a connected backend + ML vertical slice:

```text
landmarks -> ml-service /predict -> isolated word(s)
          -> backend /process -> sentence + audio URL
          -> backend session messages
microphone WAV -> backend /transcribe -> transcript -> session messages
```

The complete local loopback integration passed with two real Uvicorn child
processes, a synthetic compatible BiGRU checkpoint, and explicitly mocked cloud
providers. The synthetic checkpoint proves service wiring and checkpoint
loading; it is not recognition-quality evidence.

## Verified test evidence

All commands below used the project root `.venv`.

| Check                                                       |                          Result |
| ----------------------------------------------------------- | ------------------------------: |
| Root combined suite: backend + ML + integration tests       | **227 passed, 5 skipped** |
| Backend suite from repository root                          | **181 passed, 1 skipped** |
| Backend suite from`backend/`                              | **181 passed, 1 skipped** |
| ML suite from`ml-service/`                                |  **43 passed, 4 skipped** |
| Cross-service targeted vertical slice                       |              **3 passed** |
| Python compileall for`backend`, `ml-service`, `tests` |                          Passed |
| `git diff --check`                                        |                          Passed |

Commands used:

```powershell
.venv\Scripts\python.exe -m pytest -q -rs
.venv\Scripts\python.exe -m pytest -q -rs backend\tests
cd ml-service
..\.venv\Scripts\python.exe -m pytest -q -rs
cd ..
.venv\Scripts\python.exe -m compileall -q backend ml-service tests
```

Skipped tests:

- one opt-in Firestore emulator test; it requires Firebase emulators and
  `RUN_FIRESTORE_EMULATOR_TESTS=1`;
- four MediaPipe integration tests; `ml-service/models/face_landmarker.task`
  is not present in this workspace.

There is one existing Starlette/AnyIO deprecation warning from the installed
test-client dependency. It does not fail the suite.

## Shared contract

Implemented and reconciled in `shared/contracts.md`:

- ML input: 1–60 frames of exactly 110 points × 3 coordinates;
- point order: 21 left-hand + 21 right-hand + 68 compact face points;
- current M4/M6 output: `{"keywords":["word"]}` for one isolated word;
- backend `/process`, `/audio/{id}.mp3`, `/transcribe`, and session routes;
- UTC ISO-8601 message timestamps;
- mobile service base URL and error conventions.

## ML-service status

### M1 — environment/sanity

Implemented:

- MediaPipe Tasks hand-landmarker sanity script;
- camera, image, and video modes;
- no-camera deterministic fallback;
- project virtual-environment dependencies.

Pending evidence:

- real webcam hand detection in this workspace;
- real face-landmarker execution because the face model asset is not present.

### M2 — landmark extraction

Implemented and unit-tested:

- two hand slots;
- 68 selected compact face points;
- output shape `(frames, 110, 3)`;
- missing-hand/face zero blocks;
- deterministic timestamp handling;
- CLI output and metadata.

The actual phone implementation must reproduce the same point ordering and
camera mirroring convention.

### M3 — dataset builder

Implemented and unit-tested:

- fixed shape `(samples, 60, 110, 3)`;
- right zero-padding and first-60-frame truncation;
- deterministic ordering and metadata;
- WLASL sharded Colab extraction path;
- official split and class mapping support.

No real training-quality claim is made without the actual extracted dataset.

### M4 — classifier training

Implemented:

- Colab notebook: `notebooks/signbridge_m4_bigru_colab.ipynb`;
- Python trainer: `ml-service/scripts/04_train_bigru.py`;
- official WLASL train/validation/test split loading;
- 110×3 input contract and temporal normalization;
- small BiGRU checkpoint export;
- metrics and label mapping export.

Pending:

- run on the real WLASL/team dataset;
- save `model/bigru/bigru_best.pt`;
- record held-out accuracy, per-label counts, and model version;
- verify the target of at least 80% held-out accuracy.

### M5 — inference

Implemented and tested in `ml-service/service/inference.py`:

- M4 checkpoint loading;
- exact M4 preprocessing reuse;
- CPU/CUDA selection;
- confidence and top-k diagnostic output;
- simple `predict()` result for the API.

### M6 — API

Implemented in `ml-service/service/app.py`:

- `GET /health`;
- validated `POST /predict`;
- 110×3 shape checks;
- finite numeric checks;
- request-body limit;
- controlled 422/413/503 responses.

### M7 — container/deployment readiness

Implemented:

- `ml-service/Dockerfile`;
- `.dockerignore`;
- runtime requirements;
- model artifact build guard;
- Cloud Run `PORT` handling;
- deployment README.

Not verified because Docker and `gcloud` are unavailable here, and the real
M4 checkpoint is not in the repository. The image intentionally fails until
`ml-service/model/bigru/bigru_best.pt` is supplied.

## Backend status

The backend files now cover the full application path even though the original
team handoff identified only backend module 7 as complete.

### Implemented and locally tested

- BE-M1 `/health`;
- BE-M2 Gemini client with lazy provider setup and explicit mock mode;
- BE-M3 Text-to-Speech client with mock MP3 fixture;
- BE-M4 `/process` and retrievable `/audio/{id}.mp3`;
- BE-M5 Speech-to-Text client with WAV validation and finite timeout;
- BE-M6 `/transcribe`;
- BE-M7 Firestore client with in-memory mock repository;
- BE-M8 session creation, message writes, and ordered message reads;
- BE-M9 backend Dockerfile and Cloud Run instructions.

Provider calls now have validated finite deadlines and disabled automatic
retries where supported. Deployed mode requires an explicit HTTPS
`PUBLIC_BASE_URL`. Local mode uses port 8081 while ML uses port 8080.

### Production work remaining

- real Gemini smoke test with configured API key;
- real Text-to-Speech smoke test;
- real Speech-to-Text smoke test;
- Firebase/Firestore emulator or real Firestore smoke test;
- durable audio storage such as Cloud Storage signed URLs;
- authentication, rate limits, retention, and least-privilege IAM.

## Integration tests added

`tests/test_vertical_slice.py` verifies the in-process contract with mocked
providers, including:

- ML `/predict` response;
- backend `/process` consumption;
- audio retrieval and content type;
- session creation and ordered messages;
- transcribe result insertion;
- missing-room and invalid-message behavior.

`tests/test_http_integration.py` starts both real FastAPI applications with
Uvicorn on dynamic loopback ports and verifies:

1. ML and backend health;
2. real M5 checkpoint loading with a controlled synthetic checkpoint;
3. ML prediction;
4. backend sentence/audio processing;
5. absolute audio URL retrieval;
6. signer and speaker session messages;
7. transcript insertion;
8. UTC timestamp ordering;
9. invalid payload recovery.

## How to run locally

Use two terminals from the repository root:

```powershell
# Terminal 1
$env:SIGNBRIDGE_MODEL_PATH = "C:\path\to\bigru_best.pt"
.venv\Scripts\python.exe -m uvicorn service.app:app --app-dir ml-service --host 127.0.0.1 --port 8080

# Terminal 2
$env:MOCK_EXTERNAL_SERVICES = "1"
$env:PUBLIC_BASE_URL = "http://127.0.0.1:8081"
$env:REQUIRE_HTTPS_PUBLIC_BASE_URL = "0"
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8081
```

For a real ML prediction, the trained checkpoint must be present at the path
provided by `SIGNBRIDGE_MODEL_PATH` or at
`ml-service/model/bigru/bigru_best.pt`.

## Remaining project gates

1. Run M4 on the real dataset and install the verified checkpoint.
2. Add the face model asset where the real M2/M3 integration tests can use it.
3. Build both Docker images on a Linux-compatible Docker environment.
4. Deploy both services to Cloud Run with Secret Manager, ADC, IAM, and HTTPS.
5. Execute live `/health`, `/predict`, `/process`, `/transcribe`, and Firestore
   checks outside the development machine.
6. Complete the physical mobile-device test: camera permissions, two-hand + face
   landmarks, audio playback, microphone capture, and room polling.
7. Replace local ephemeral MP3 storage before relying on multiple Cloud Run
   instances.

## Bottom line

The code and automated local integration are in place for the backend/ML
vertical slice. The project is not yet production- or demo-deployment complete
because the real trained checkpoint, face asset, cloud credentials, Docker/
Cloud Run execution, and physical mobile smoke test remain outstanding.
