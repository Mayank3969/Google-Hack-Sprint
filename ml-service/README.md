# SignBridge ML service

This service contains the M2–M7 landmark pipeline.

## Run locally

From the repository root with the project virtual environment active:

```powershell
.venv\Scripts\python.exe -m uvicorn service.app:app --app-dir ml-service --host 127.0.0.1 --port 8080
```

The service requires a trained M4 checkpoint for `/predict`:

```text
ml-service/model/bigru/bigru_best.pt
```

If the checkpoint is stored elsewhere, configure it before starting Uvicorn:

```powershell
$env:SIGNBRIDGE_MODEL_PATH = "C:\path\to\bigru_best.pt"
```

Check health:

```powershell
curl http://127.0.0.1:8080/health
```

Predict from a JSON file containing the `(frames, 110, 3)` landmark contract:

```powershell
curl -X POST http://127.0.0.1:8080/predict `
  -H "Content-Type: application/json" `
  --data-binary "@payload.json"
```

Successful response:

```json
{"keywords":["pay"]}
```

## Build and run the M7 container

The image includes the runtime code, MediaPipe hand model, and the trained M4
checkpoint. The source tree contains only `model/.gitkeep` as a placeholder;
place the real checkpoint at `model/bigru/bigru_best.pt` before building:

```powershell
docker build --tag signbridge-ml:local .
docker run --rm --publish 8080:8080 --env PORT=8080 signbridge-ml:local
```

The Dockerfile fails the build when that checkpoint is missing or empty, so a
placeholder-only image cannot be mistaken for a deployable model service.

The container listens on the Cloud Run `PORT` environment variable, defaulting
to `8080`, and uses CPU inference by default. Set `SIGNBRIDGE_DEVICE=cuda` only
on a runtime that actually provides CUDA.

## Cloud Run handoff

Deploy this directory as the Cloud Run build context. Configure:

- unauthenticated invocation for the hackathon mobile demo;
- `PORT` supplied by Cloud Run;
- `SIGNBRIDGE_DEVICE=cpu`;
- a trained `model/bigru/bigru_best.pt` included in the image; and
- a region appropriate for the mobile users.

From `ml-service`, a source deployment can be started after the checkpoint is
present (replace the service name and region for the target project):

```powershell
gcloud run deploy signbridge-ml `
  --source . `
  --region REGION `
  --allow-unauthenticated `
  --set-env-vars SIGNBRIDGE_DEVICE=cpu
```

Cloud Run supplies `PORT`; do not hard-code a different listen port. Verify
`/health` as the cheap liveness check, then send the same known landmark
fixture used locally to `/predict`. Prediction returns HTTP 503 if the model
cannot be loaded, while a missing checkpoint is rejected earlier by the image
build guard.
