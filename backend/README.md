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
in the mobile app. The backend service account also needs access to the Gemini
secret. Verify the configured `GEMINI_MODEL` is available to the project.

After deployment, verify `/health`, a real `/process` request and playable
HTTPS audio URL, `/transcribe`, and the session lifecycle against the live
service. Audio stored on one instance is not shared with another instance;
durable shared storage is required for reliable scaled audio retrieval.
These commands are deployment instructions, not evidence that a deployment
has succeeded.
