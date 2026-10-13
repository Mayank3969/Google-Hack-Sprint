# SignBridge backend service

The backend consumes the isolated word returned by `ml-service /predict`, turns
it into a sentence, synthesizes audio, handles speech transcription, and stores
conversation messages.

## Development environment and tests

Use Python 3.13 and one `.venv` at the repository root. From the repository
root, install the combined development requirements into that environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The root test command includes `backend/tests`, `ml-service/tests`, and the
cross-service `tests` directory. The `integration` marker remains available
for local MediaPipe/OpenCV checks. The Firestore emulator test requires its
explicit opt-in environment and otherwise skips.

For backend-only tests, run this from `backend/`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

The BE-M12 deterministic bilingual mock matrix can be run directly from the
repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q backend\tests\test_m12_mock_matrix.py
```

It covers the five plumbing scenarios (`pay medicine ?`, `hello`, `need water`,
`doctor tomorrow ?`, and `emergency hospital`). In mock mode it verifies the
English/Hindi subtitle fields, English-then-Hindi playback order, both
retrievable `audio/mpeg` segments, repeatability, and that Gemini/Google TTS
provider setup is never reached. These checks are integration plumbing evidence,
not translation-quality or model-accuracy evidence.

## BE-M13 verification gate

Run the privacy-safe local pre-deployment gate from the repository root:

```powershell
.venv\Scripts\python.exe -m backend.verification_gate --json
```

The default gate runs the complete backend mock suite, the cross-service
loopback suite, and contract-documentation checks. It captures timings but does
not print subprocess output, prompts, generated text, credentials, or provider
exceptions.

After configuring live credentials and starting a reachable backend, add the
live checks:

```powershell
$env:MOCK_EXTERNAL_SERVICES = "0"
$env:GEMINI_API_KEY = "<set outside source control>"
$env:SIGNBRIDGE_BACKEND_URL = "https://<backend-url>"
.venv\Scripts\python.exe -m backend.verification_gate --live --json
```

The live gate checks all five Gemini scenarios, both TTS languages, the
bilingual `/process` contract and both audio URLs, health, transcription,
sessions, missing-audio behavior, and documentation alignment. A passing mock
gate is not a substitute for live provider or physical-device verification.

## Live provider credentials and secret safety

The GitHub-visible template is `backend/.env.example`. Copy it to an ignored
local file instead of editing the example:

```powershell
Copy-Item backend\.env.example backend\.env
```

For a real local smoke test, the backend needs:

| Capability | Credential/configuration | Secret? |
| --- | --- | --- |
| Gemini sentence/translation | `GEMINI_API_KEY` from Google AI Studio, or `GEMINI_USE_VERTEX_AI=1` with ADC and `GOOGLE_CLOUD_PROJECT`; `GEMINI_MODEL` selects the model | API keys and ADC are sensitive |
| Text-to-Speech | Google Cloud Application Default Credentials and a project with Cloud Text-to-Speech API enabled | No separate API key; ADC is sensitive local credential state |
| Speech-to-Text | Google Cloud Application Default Credentials and Cloud Speech-to-Text API enabled | No separate API key; ADC is sensitive local credential state |
| Firestore | Google Cloud Application Default Credentials and a Firestore database/project | No separate API key; ADC is sensitive local credential state |

Preferred local Google authentication is:

```powershell
gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
```

For Gemini through Vertex AI, set `GEMINI_USE_VERTEX_AI=1` and
`GOOGLE_CLOUD_LOCATION` (default `global`). This path uses ADC and needs no
`GEMINI_API_KEY`. AI Studio is selected explicitly when the Vertex flag is off.
The backend uses `google-genai==2.29.0`, keeps the existing configurable default
`GEMINI_MODEL=gemini-3.8-flash`, and applies the 1–60 second deadline with one
SDK attempt (no automatic retries). Set a model available to your project.

If ADC is not available, `GOOGLE_APPLICATION_CREDENTIALS` may point to a local
service-account JSON file, but that file must remain outside Git and outside the
Docker build context. Do not paste API keys, OAuth tokens, ADC payloads, or
service-account JSON into chat, source files, tests, logs, or
`backend/.env.example`.

For live testing set `MOCK_EXTERNAL_SERVICES=0`, configure either Gemini auth
path above, use an HTTPS `PUBLIC_BASE_URL` for a deployed backend, and
run the privacy-safe gate:

```powershell
$env:MOCK_EXTERNAL_SERVICES = "0"
$env:GEMINI_API_KEY = "<local-secret-not-for-source-control>"
$env:SIGNBRIDGE_BACKEND_URL = "https://<backend-url>"
.venv\Scripts\python.exe -m backend.verification_gate --live --json
```

The gate intentionally reports only check names and timings.

`POST /process` accepts either the legacy `{"keywords": [...]}` body or the ML
capture-results body with `capture_id` and up to 10 indexed `results` (the full
`/v2/predict/batch` envelope, including `complete` and `integration_ready`, is
also accepted). Results are processed in `segment_index` order. The backend
uses the recognized `keyword` from each slot; `top_k` is diagnostic candidate
data and is never expanded into phrase words. Duplicate labels are preserved
because repeated signs are meaningful. Every submitted slot must be recognized
with confidence and coordinate coverage at or above the inclusive
`SIGNBRIDGE_RECOGNITION_THRESHOLD` (default `0.50`), and an explicitly false
batch completeness/readiness flag rejects the whole capture.

Gemini performs a completeness check before translation. If any ML slot is
uncertain, invalid, incomplete, below threshold, or missing its recognized
keyword, `/process` returns HTTP 200 with
`{"status":"record_again","record_again":true}` before calling Gemini or
TTS; it writes no audio. The mobile client should start a new capture. Accepted
input keeps the bilingual subtitle and English-then-Hindi audio response
described above.

With the root virtual environment activated, `python -m pytest -q` works from
either directory. Backend runtime dependencies live in `requirements.txt`;
`requirements-dev.txt` adds pytest and the HTTP test client. The Docker image
installs only the runtime requirements.

## Local two-service run

Run the ML service on port `8080` and the backend on port `8081` so the services
do not collide:

```powershell
# Terminal 1, from the repository root
$env:SIGNBRIDGE_MODEL_PATH = "C:\path\to\bigru_best.pt"
.venv\Scripts\python.exe -m uvicorn service.app:app --app-dir ml-service --host 127.0.0.1 --port 8080

# Terminal 2, from the repository root
$env:MOCK_EXTERNAL_SERVICES = "1"
$env:PUBLIC_BASE_URL = "http://127.0.0.1:8081"
$env:REQUIRE_HTTPS_PUBLIC_BASE_URL = "0"
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8081
```

These loopback addresses work only for clients on the development desktop:

```text
ML_BASE_URL=http://127.0.0.1:8080
BACKEND_BASE_URL=http://127.0.0.1:8081
```

For a physical phone on the same LAN, start both servers with `--host 0.0.0.0`,
use the desktop's LAN address (for example `http://192.168.1.20:8080` and
`http://192.168.1.20:8081`), and set backend `PUBLIC_BASE_URL` to that same
phone-reachable backend address. Allow the two ports through the desktop
firewall. `0.0.0.0` is a server bind address, not a client URL.

For the standard Android emulator, use `http://10.0.2.2:8080` and
`http://10.0.2.2:8081` to reach the host; set `PUBLIC_BASE_URL` to
`http://10.0.2.2:8081` too. Other emulator/simulator environments may use a
different host address. A device's `localhost` or `127.0.0.1` normally refers
to the device itself. Local HTTP also requires the mobile development build
to permit cleartext traffic.

`.env.example` documents configuration; the commands above set environment
variables explicitly and do not automatically load an `.env` file. Backend
mock mode is deterministic and does not call cloud providers. ML prediction
still requires a real trained checkpoint at `SIGNBRIDGE_MODEL_PATH` or
`ml-service/model/bigru/bigru_best.pt`.

For browser clients, export `CORS_ALLOWED_ORIGINS` with exact comma-separated
frontend origins (for example `http://localhost:5173`) and restart the backend.
CORS is disabled by default; origins must not include paths or trailing slashes.

Use HTTPS URLs and set `REQUIRE_HTTPS_PUBLIC_BASE_URL=1` in deployed
environments. The local `/tmp` audio directory is an MVP shortcut and is not
durable across Cloud Run instances.

## Backend container and Cloud Run

Build and run from `backend/` as the Docker build context. This local example
publishes the backend on desktop port `8081` and explicitly enables mocks:

```powershell
docker build --tag signbridge-backend:local .
docker run --rm --publish 8081:8080 `
  --env PORT=8080 `
  --env MOCK_EXTERNAL_SERVICES=1 `
  --env PUBLIC_BASE_URL=http://127.0.0.1:8081 `
  --env REQUIRE_HTTPS_PUBLIC_BASE_URL=0 `
  signbridge-backend:local
```

Use the LAN/emulator address instead of `127.0.0.1` for `PUBLIC_BASE_URL` when
testing this container from a mobile client. The image defaults to real
providers and HTTPS enforcement when these local overrides are omitted.

For Cloud Run, configure the target project, region, Secret Manager secret,
and service-account permissions first. From `backend/`, replace `REGION` and
`<backend-service-host>` with the target region and backend HTTPS service host:

```powershell
gcloud run deploy signbridge-backend `
  --source . `
  --region REGION `
  --allow-unauthenticated `
  --set-env-vars "MOCK_EXTERNAL_SERVICES=0,REQUIRE_HTTPS_PUBLIC_BASE_URL=1,PUBLIC_BASE_URL=https://<backend-service-host>" `
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
```

On the first deployment, if the service URL is not known yet, omit
`PUBLIC_BASE_URL` from `--set-env-vars`, obtain the created service's HTTPS URL,
then set it before testing `/process`:

```powershell
gcloud run services update signbridge-backend `
  --region REGION `
  --update-env-vars "PUBLIC_BASE_URL=https://<backend-service-host>"
```

Cloud Run supplies `PORT`; the container binds `0.0.0.0` on that port. Deploy
`ml-service/` separately using its own Dockerfile and build context, with the
trained checkpoint installed as described in `../ml-service/README.md`.

Speech, Text-to-Speech, and Firestore use the Cloud Run service account through
Application Default Credentials. Do not put those credentials in this image or
in the mobile app. For the AI Studio deployment above, the backend service
account also needs access to the Gemini secret. Vertex AI deployments instead
set `GEMINI_USE_VERTEX_AI=1`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`
and grant the service account Vertex AI access. Verify the configured
`GEMINI_MODEL` is available to the project.

After deployment, verify `/health`, a real `/process` request and playable
HTTPS audio URL, `/transcribe`, and the session lifecycle against the live
service. Audio stored on one instance is not shared with another instance;
durable shared storage is required for reliable scaled audio retrieval.
These commands are deployment instructions, not evidence that a deployment
has succeeded.
