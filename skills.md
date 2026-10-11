# SignBridge — Build Guide, Mobile-Readiness Contract, and Development Roadmap

Scope: `ml-service/` (Builder-core), `backend/` (Builder-integration), and the
mobile-app integration contract. The mobile UI and platform-specific code may
be built separately, but this guide makes the services and the mobile handoff
explicit enough that a mobile developer can implement against stable APIs
without guessing about payload shapes, permissions, or failure states.

The target is a demonstrable vertical slice, not just an exported model:

`mobile camera -> two-hand + face landmarks -> ml-service -> keywords -> backend -> sentence + audio -> mobile playback`

and, in the opposite direction:

`mobile microphone -> base64 LINEAR16 audio -> backend -> transcript -> mobile session timeline`.

---

## Roadmap (new in this revision)

This roadmap sits on top of the runbook in sections 0–9. The module IDs it
references (`ML-M1`, `BE-M4`, ...) are the existing modules in sections 4 and 5.
Nothing in the runbook is removed; the roadmap says **when** to do each part,
**which Google tool** to prefer, and **what gate** must pass before the next
phase starts.

### R.1 Planning assumptions (change these if they are wrong)

- **Goal sequence:** working vertical slice first (about two days of build), then
  harden it, then improve the model, then grow it into a product.
- **Team shape:** `ml-service` (Builder-core), `backend` (Builder-integration), and
  a mobile client built separately. After the shared contract is approved, ML,
  backend, and mobile run in parallel.
- **Google-first rule:** use a Google product whenever it does the job as well as
  an alternative. Keep the model itself simple (scikit-learn) until data, not
  tooling, is the bottleneck.
- **Credentials rule (unchanged):** no Google credential ever ships in the mobile
  app. The mobile app talks only to the two Cloud Run URLs (and, from Phase 6,
  to Firebase through Firebase's own security rules and App Check).
- **Defaults chosen for you** (full reasoning in section 10):
  - Landmark extraction: **MediaPipe** on-device (Hand Landmarker, Tasks API).
  - Mobile: **Android-first demo**; **Flutter** only if the mobile developer
    already knows it (see decision D1 in section 11).
  - Training: local venv for the baseline; **Google Colab** if more compute is needed.
  - Data and files: **Google Cloud Storage** (private bucket) for recordings,
    **Google Sheets** for the checklist, dataset manifest, and test log.
  - AI: **Gemini** via AI Studio key for the demo, **Vertex AI with Application
    Default Credentials** afterwards, so the API key disappears.
  - Hosting/CI: **Cloud Run + Cloud Build + Artifact Registry + Secret Manager**.
  - Data store: **Firestore** (already in the plan).

### R.2 Phase overview

| Phase | Window | Outcome | Main Google tools | Exit gate |
|---|---|---|---|---|
| 0. Foundations | before any module work (~½ day) | contract signed off, cloud project ready, tracker live | Cloud Console, Cloud Storage, Sheets, Billing budgets | Section 2.1 preflight ticked; `contracts.md` reviewed by mobile dev |
| 1. Data + model | Day 1 | labeled dataset and a ≥80% held-out classifier behind `predict()` | MediaPipe, Colab (optional), Cloud Storage | `ML-M1…M5` pass; accuracy evidence recorded |
| 2. Services live | Day 1 evening → Day 2 morning | both services deployed and verified from outside the dev machine | Cloud Run, Cloud Build, Artifact Registry, Secret Manager, Gemini, Speech-to-Text, Text-to-Speech, Firestore | `ML-M6…M7`, `BE-M1…M9`, section 6.1 checklist |
| 3. Mobile integration + demo | Day 2 | physical-device vertical slice | MediaPipe Tasks (mobile), Android Studio / Flutter | Section 8 final integration test, all 7 checks |
| 4. Hardening | week 1 after demo | no public secrets, durable audio, least privilege, cost guard-rails | Cloud Storage signed URLs, IAM, Vertex AI, Cloud Logging/Monitoring, Cloud Build tests | Hardening checklist (R.3, Phase 4) |
| 5. Model quality | weeks 2–5 | larger vocabulary, multiple signers, measurable accuracy | Colab, TensorFlow/Keras, TFLite, Cloud Storage | ≥85% on a held-out *signer* the model never trained on |
| 6. Product | weeks 5–10 | realtime conversation, auth, analytics, pilot | Firebase Auth, App Check, Firestore realtime, Crashlytics, App Distribution | Pilot with real signers/interpreters completed, findings logged |

**Critical path:** contract → dataset → classifier → `/predict` → deploy →
mobile smoke test. Backend M1–M8 are off the critical path and can use mock mode
until Phase 2, but the *deploy* of both services is on it.

### R.3 Phase detail

#### Phase 0 — Foundations (do before Module 0)

1. Complete section 2 (project, billing, APIs, Firestore, Gemini route) and 2.1.
2. **Set a billing budget with alert thresholds** (Billing → Budgets & alerts, for
   example at 50%, 90%, and 100% of the expected spend). The Cloud Run endpoints
   are public in the demo, so a budget alert is the cheapest abuse guard.
3. Create a **private Cloud Storage bucket** (same region as Firestore, e.g.
   `asia-south1`) with uniform bucket-level access and *no* public access. Use
   prefixes `raw-recordings/`, `datasets/`, `models/`, `audio/`. This replaces
   shared drives or email for recordings and is where Phase 4 audio will live.
4. Create one **Google Sheet** "SignBridge tracker" with tabs: `Checklist`
   (mirrors section 7), `Dataset manifest` (label, filename, signer id, date,
   camera/background, consent yes/no), `Test log` (module, command, result,
   date), and `Latency log` (endpoint, p50, notes). Export the manifest to CSV
   into the repo (`ml-service/data/manifest.csv`) whenever the dataset changes
   so accuracy claims stay reproducible.
5. Write `shared/contracts.md` (section 3) and get the mobile developer's sign-off.
6. Record decisions D1–D5 from section 11 in the mobile README.

**Gate:** preflight ticked, contract reviewed, budget alert active.

#### Phase 1 — Data and model (Day 1)

1. `ML-M1` (sanity check). Prefer the current MediaPipe **Hand Landmarker (Tasks
   API)** over the legacy `mp.solutions.hands` shown in the snippets, because the
   mobile SDKs use the Tasks API and you want identical landmark behavior on
   desktop and phone. The output shape `(frames, 21, 3)` does not change.
2. **Capture protocol first** (before recording anything): choose the first
   vocabulary (start with 5–8 signs, including one question marker such as `?`),
   fix camera distance/orientation/background, and aim for 15–20 recordings per
   sign from at least two people. Upload raw recordings to
   `gs://<bucket>/raw-recordings/` and fill the manifest. Get recorded signers'
   permission before keeping their video.
3. `ML-M2` → `ML-M3` → `ML-M4` → `ML-M5` exactly as in section 4. Keep
   scikit-learn Random Forest. If a laptop is too slow or the team wants shared
   notebooks, run training in **Colab**, read the dataset from Cloud Storage,
   and download `classifier.pkl` plus its metadata back into the repo's
   `model/` folder.
4. Create the **known-good fixture** (`tests/fixtures/known_sample.json`) now;
   mobile and backend tests will reuse it.

**Parallel work during Phase 1**
- Backend: `BE-M1`, then `BE-M2`/`BE-M3`/`BE-M5` clients with the mock path on.
- Mobile: role selection, permission screens, and a **landmark-capture spike**
  that produces a `(frames, 21, 3)` array logged to the console. Compare it
  against the desktop fixture before wiring any network calls.

**Gate:** held-out accuracy ≥80% with per-label counts saved, `predict()` unit
tests pass, fixture exists. If accuracy is below 80%, collect more recordings
(section 4, M4); do not tune the model architecture yet.

#### Phase 2 — Services live (Day 1 evening → Day 2 morning)

1. `ML-M6` (`/predict` with validation) and `ML-M7` (container).
2. `BE-M4` (`/process`), `BE-M6` (`/transcribe`), `BE-M7` (Firestore client),
   `BE-M8` (`/session`), `BE-M9` (container).
3. Put `GEMINI_API_KEY` in **Secret Manager** and reference it from Cloud Run
   (section 6, step 8). Set `PUBLIC_BASE_URL` to the backend's HTTPS URL.
4. Deploy both services through Cloud Build triggers with **included-files
   filters** (`ml-service/**`, `backend/**`).
5. Run the full section 6.1 verification and publish the two base URLs to the
   mobile developer.

**Cloud Run settings worth setting now:** a **maximum instance count** (e.g. 3–5)
as a cost cap, and **minimum instances = 1** for `ml-service` on demo day so the
first request does not pay a cold start loading the model. Set min back to 0
afterwards.

**Gate:** every item in section 6.1 passes from a network outside the developer
machine.

#### Phase 3 — Mobile integration and demo (Day 2)

1. Follow the section 1.1 sequence: create room → capture → `/predict` →
   `/process` → play audio → post `signer` message → record → `/transcribe` →
   post `speaker` message → poll messages.
2. Run section 8 on a **physical device**. Capture the evidence listed in section 9.
3. **Demo safety net:** keep the fixture-based replay path and mock mode ready as
   a clearly labeled fallback if venue Wi-Fi or a provider fails. State
   honestly in the demo which parts were live.
4. Freeze a **demo build**: pinned Cloud Run revisions, pinned model version,
   known vocabulary list, and a one-page script that only uses trained signs.

**Gate:** all seven section 8 checks pass; handoff note saved.

#### Phase 4 — Hardening (week 1 after the demo)

- [ ] **Audio to Cloud Storage.** Write MP3s to `gs://<bucket>/audio/` and return
      a short-lived **signed URL** (or serve through the backend with a lifecycle
      rule deleting objects after ~1 day). The `audio_url` contract does not
      change. This fixes the multi-instance problem where `/tmp` audio written on
      one Cloud Run instance 404s on another.
- [ ] **Least-privilege IAM.** Replace the broad **Editor** role on the default
      compute service account with a dedicated service account per service, with
      only what each needs (for example Firestore user, Secret Manager accessor,
      Storage object user for the audio prefix). Confirm Speech-to-Text and
      Text-to-Speech still work after the change.
- [ ] **Remove the Gemini API key.** Move `gemini_client.py` to the Vertex AI
      Gemini client using Application Default Credentials (same function
      signature). Delete the AI Studio key and its secret.
- [ ] **Verify model and SDK names.** Section 5 M2 pins a specific Gemini model
      and the `google-generativeai` package. Before this phase, check Google's
      current model list and SDK guidance, then set `GEMINI_MODEL` and update the
      client accordingly. Never hard-code the model name again.
- [ ] **Abuse controls.** Request size limits (already specified), per-IP/room
      rate limiting at the app layer or in front of Cloud Run, and Cloud Run max
      instances. Keep the billing alerts from Phase 0.
- [ ] **CI that tests before deploying.** Add a `cloudbuild.yaml` per service that
      runs `pytest` (mock mode) before building the image; use **Artifact
      Registry** for images; keep the path-filtered triggers.
- [ ] **Observability.** Cloud Logging with request ID, endpoint, and duration
      (never keys, audio, frames, or full conversation text); a **Cloud
      Monitoring uptime check** on both `/health` URLs with an alert; **Error
      Reporting** enabled; log-based metric for 5xx rate.
- [ ] **Firestore rules and retention.** Decide how long messages live; add a TTL
      policy on `rooms` documents; make room IDs longer than 8 characters.
- [ ] **Privacy basics.** Written note on what is stored (messages, no video, no
      audio), retention, and consent for training recordings.

**Gate:** none of the "stop and resolve" conditions in section 0.3 are present;
service accounts are least-privilege; no API key exists anywhere.

#### Phase 5 — Model quality (weeks 2–5)

Only start this after Phase 4's data and consent basics are in place. Make one
change at a time and keep the section 0.2 definition of "model done".

1. **More data, more people.** Target 30+ recordings per sign across at least
   5 signers, different lighting/backgrounds/phones. Always evaluate on a
   **held-out signer** (leave-one-signer-out), because splitting random
   recordings from the same signers overstates accuracy.
2. **Better features before better models.** Normalize landmarks relative to the
   wrist and hand scale, add frame-to-frame velocity, and compare against the
   Random Forest baseline using the same split.
3. **Sequence model.** Try a small GRU/1D-CNN in **TensorFlow/Keras** on **Colab**.
   Keep it only if it beats the baseline on the held-out signer.
4. **Vocabulary expansion with an "unknown" class.** Add a `none/unknown` label
   and a confidence threshold so non-signs do not produce confident nonsense.
   Return top-k with confidences from `/predict` as an *additive* contract
   change (new optional fields; old clients keep working).
5. **Two-hand and continuous signing.** Only after isolated signs are reliable:
   extend to two hands (shape becomes `(frames, 2, 21, 3)`, a **versioned**
   contract change) and add a start/end detector for continuous capture.
6. **On-device option.** Export the best model to **TensorFlow Lite** and
   evaluate running it on the phone. Benefits: lower latency, works offline,
   fewer bytes sent. Keep `/predict` as a server fallback and compare accuracy
   of both paths on the same fixtures.
7. **Gemini quality.** Add few-shot examples and a regression list of
   keyword→sentence cases (`pay medicine ?` → a question). Track them in the
   Sheet and rerun after every prompt or model change.

**Gate:** ≥85% on a held-out signer with per-label results, no label below 70%,
and a documented false-positive rate on non-sign movements.

#### Phase 6 — Product (weeks 5–10)

1. **Identity and abuse protection.** Add **Firebase Authentication** (start with
   anonymous sign-in), **Firebase App Check** on the mobile app, and Cloud Run
   authentication (the backend verifies Firebase ID tokens; stop allowing
   unauthenticated invocations on `backend`).
2. **Realtime timeline.** Replace polling with Firestore realtime listeners via
   the Firebase mobile SDK, protected by Firestore security rules that only let
   a user read/write rooms they joined. Keep `GET /messages` as a fallback.
3. **Languages.** Make Speech-to-Text and Text-to-Speech language codes
   configurable per session (for example `hi-IN` alongside `en-IN`), and have
   Gemini output the sentence in the session language. Test each language
   with native speakers before listing it as supported.
4. **Release engineering.** **Firebase App Distribution** for testers,
   **Crashlytics** for crashes, and privacy-safe analytics (screen, error and
   latency events only; no text, audio, or video).
5. **Accessibility and UX.** Large-text/high-contrast modes, haptic cues, visual
   confirmation of what was heard/signed, and an easy "that was wrong, retry"
   control. Collect that correction signal (with consent) as new training data.
6. **Pilot.** Test with real Deaf/hard-of-hearing users and interpreters,
   paying attention to *their* priorities (vocabulary, speed, error recovery).
   Record findings in the Sheet and let them drive the Phase 5/6 backlog.

**Gate:** pilot complete, top three findings addressed or scheduled.

### R.4 What to deliberately postpone

Do not start these before the Phase 3 gate: LSTM/transformer experiments,
two-hand support, on-device TFLite, Firebase Auth/App Check, multi-language,
or a fancy UI. Each depends on a stable contract and a trustworthy baseline.

---

## 0. How an agent must use this file

This is an execution runbook. Do not treat the code blocks as disconnected
examples: perform the actions in order, make the smallest change that satisfies
the current module, and leave evidence that the module works before moving on.

### 0.1 Required execution loop

At the start of the work, the agent must:

1. Inspect the repository tree and identify whether `ml-service/`, `backend/`,
   `shared/`, `data/`, and any existing mobile folder already exist. Preserve
   unrelated work; do not replace an existing framework or create a starter
   template without an explicit decision.
2. Read the current dependency files, environment files, and tests. Create a
   redacted `.env.example` only if the project does not already have one. Never
   write a real API key, service-account JSON, token, or device credential to
   the repository.
3. Confirm the chosen mobile stack and the two service base URLs as project
   decisions. The API contract below is framework-agnostic, so Flutter,
   React Native, native Android, or native iOS can consume it.
4. Create a short implementation checklist using the modules below. Mark a
   module complete only after its stated test has actually passed; record the
   command, result, and any environment limitation next to the checklist.

For every module, perform this same six-step loop:

1. **State the outcome.** Identify the one user-visible or service-visible
   capability being added.
2. **Prepare only the required files.** Touch the files named by that module
   and the minimum import/package files required to run them. Do not refactor
   later modules early.
3. **Implement the smallest complete path.** Keep interfaces stable, make
   configuration explicit, and use a local fake for a paid/cloud dependency
   when the real credential or network is not available.
4. **Run the module test.** Prefer a deterministic unit or HTTP test. For
   camera, microphone, Firestore, or cloud tests, also run the real smoke test
   when the required device/credential exists.
5. **Fix before advancing.** A skipped test is not a passing test. If a test is
   blocked by a missing secret, ask one targeted setup question or record the
   exact blocked gate and continue only with independent work.
6. **Record evidence.** Save the output shape, URL, accuracy, sample count, or
   other acceptance evidence. Then check the module off in section 7.

### 0.2 Definition of “model done”

The classifier is not considered done when `classifier.pkl` merely exists. The
ML work is complete only when all of these are true:

- the dataset contains labeled, fixed-shape sequences and a repeatable build
  command;
- validation accuracy is at least 80% on a held-out recording set, with the
  number of samples and per-label counts recorded;
- the saved model can predict a known sample through the same `predict()` path
  used by the API;
- `/health` and `/predict` work in the local container and on the deployed
  service;
- a real mobile-shaped payload (multiple frames, 21 points per frame, and
  zero-filled missing-hand frames) receives a valid response; and
- the mobile app can use the returned keywords to obtain playable audio from
  the backend.

### 0.3 Order of work and stop conditions

Work in this order: shared contract -> ML environment and data -> ML model ->
ML API -> backend skeleton and cloud clients -> backend APIs -> containers and
deployment -> mobile client integration smoke test. Do not tune the model
before the landmark shape is fixed, and do not start mobile UI work against
unstable or undocumented endpoints.

Stop and resolve the issue before advancing if:

- a payload shape differs between the contract, test, and implementation;
- a model test uses a different preprocessing path from the production API;
- `audio_url` is a local filesystem path rather than a URL the phone can reach;
- a cloud credential is being requested by or embedded in the mobile app; or
- a deployment works only from one developer's machine and cannot be rebuilt
  from the repository.

---

## 1. Architecture note (why backend exists separately from the mobile app)

The mobile app captures camera frames or microphone audio, sends only the
required data over HTTPS, and renders the response. The preferred MVP flow is:

1. The phone runs MediaPipe Hands (or the platform equivalent) on camera
   frames and builds a sequence shaped `(frames, 21, 3)`.
2. The phone sends that sequence to `ml-service /predict` and receives a small
   keyword list.
3. The phone sends the keywords to `backend /process` and receives a natural
   sentence plus an **absolute, HTTPS, phone-reachable** `audio_url`.
4. The phone plays the MP3 and optionally writes the sentence to the shared
   session timeline.
5. For speaker-to-signer communication, the phone records 16 kHz mono
   LINEAR16/WAV audio, base64-encodes it, sends it to `backend /transcribe`,
   and displays the returned text.

The mobile app never holds any Google Cloud API key or credential. All Gemini,
Speech, Text-to-Speech, and Firestore credentials live only inside `backend`
and `ml-service`, running on Cloud Run. A key embedded in a mobile binary can
be extracted, so this is both a security requirement and an important
architecture explanation for the demo.

Native mobile HTTP calls do not need CORS headers (CORS is a browser-only
restriction). The services still need HTTPS, bounded request sizes, useful
HTTP status codes, and timeouts because phones move between networks.

### 1.1 Mobile app planning for the MVP

Keep this section as the handoff contract even if a separate team implements
the app. Before mobile coding starts, make these decisions and write them in
the mobile project README:

- **Framework and platforms:** choose one stack and one first demo platform;
  do not make the backend depend on the choice.
- **Roles:** `signer` captures a sign and receives spoken audio; `speaker`
  records speech and receives readable text. A later version may let one
  device switch roles.
- **Screens:** landing/role selection, camera capture, microphone/transcript,
  session join/create, and a conversation timeline with clear loading/error
  states.
- **Permissions:** request camera only when entering sign mode and microphone
  only when entering speaker mode. Explain denial and provide a retry/settings
  path; never silently submit empty data.
- **Configuration:** keep `ML_BASE_URL` and `BACKEND_BASE_URL` in build-time
  environment configuration. They are URLs, not secrets. Do not ship
  `GEMINI_API_KEY`, Google credentials, or Firestore configuration in the app.
- **Networking:** use HTTPS, a finite request timeout, visible progress state,
  and a retry button. Retry safe reads such as `GET /messages`; do not blindly
  repeat a mutating request after an ambiguous timeout.

The mobile client should implement the following sequence before polishing UI:

1. Call `POST /session/create` and retain the returned `room_id` in screen
   state.
2. Capture a short landmark window, sample consistently, and call
   `POST {ML_BASE_URL}/predict`.
3. Render the returned keywords while processing, call
   `POST {BACKEND_BASE_URL}/process`, stream or download `audio_url`, and play
   the MP3.
4. Send the displayed sentence as a `signer` message if the product wants
   conversation history.
5. In speaker mode, record a 16 kHz mono clip, base64-encode the complete WAV,
   call `/transcribe`, and add the returned text as a `speaker` message.
6. Poll `/session/{room_id}/messages` during the MVP (for example every 1–2
   seconds while the timeline is visible); replace polling with a realtime
   transport only after the basic flow is reliable.

The first mobile acceptance test is a physical-device smoke test: grant
permissions, make one known sign, hear the generated sentence, speak one short
phrase, see its transcript, and see both messages in the same room. A simulator
may be used for layout work, but it is not evidence that camera, microphone,
audio playback, or landmark extraction work.

---

## 2. One-time Google Cloud project setup (do this before Module 0)

1. Go to `console.cloud.google.com` → create a new project (or pick an existing one) → note the **Project ID** (you'll need it everywhere below).
2. Confirm billing is linked (Billing → Link a billing account). New accounts get free trial credits — this is enough for a hackathon.
3. Go to **APIs & Services → Library** and enable each of these (search by name, click, click Enable):
   - Cloud Speech-to-Text API
   - Cloud Text-to-Speech API
   - Cloud Firestore API
   - Cloud Run Admin API
   - Cloud Build API
   - Artifact Registry API
   - Vertex AI API (only needed if you go the Vertex AI route for Gemini/model hosting — see note below)
4. **Firestore:** go to **Firestore** in the left sidebar → **Create Database** → choose **Native mode** → pick a region close to you (e.g. `asia-south1` for India) → Create.
5. **Gemini access — pick ONE:**
   - **Fast path (recommended for a 2-day build):** go to `aistudio.google.com` → Get API Key → copy it. You'll store this as `GEMINI_API_KEY` later. This uses the plain Gemini API, not Vertex AI.
   - **Stronger-for-judges path (more setup):** use Gemini through the Vertex AI SDK instead, authenticated via your Cloud project's service account (no separate API key). Only do this if Module backend-M2 finishes early and you have spare time — it's the same prompt logic, different client library (`vertexai` instead of `google.generativeai`).
6. You do **not** need to manually create a service account key file for Cloud Run — Cloud Run services run under a default service account automatically. You just need to grant that account the right permissions:
    - Go to **IAM & Admin → IAM**, find the service account named `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`.
    - Click the pencil icon → **Add Another Role** → add **Editor** (fastest for a hackathon; tighten later if this becomes a real product). This single role covers Firestore, Speech, and Text-to-Speech access.

> **Roadmap note:** Editor is acceptable for the demo only. Phase 4 replaces it with a dedicated, least-privilege service account per Cloud Run service.

### 2.1 Preflight checklist before writing service code

Perform this check once and keep the result in the local work log (not in
source control):

- [ ] Project ID, region, billing status, and enabled APIs are known.
- [ ] A Firestore Native-mode database exists in the chosen region.
- [ ] A Gemini route has been selected: AI Studio key for the fast path or
      Vertex AI/service-account authentication for the stronger path.
- [ ] Local development has either Application Default Credentials configured
      or a documented mock mode for each Google client.
- [ ] The mobile team knows that the model input is landmarks, not a video URL,
      and has confirmed whether on-device MediaPipe Hands is available in its
      selected framework.
- [ ] A physical test device, a short test WAV, and at least one labeled
      recording per planned sign are available. A single recording per label is
      enough to validate plumbing, but not enough to claim model quality.
- [ ] `.gitignore` excludes `.env`, credential files, generated audio, raw
      private recordings, `__pycache__/`, and local model-training artifacts that
      are not intentionally versioned.

If a required cloud account or device is not available, implement the local
fake and deterministic tests first. Do not weaken the API contract or commit a
credential just to make a test green.

---

## 3. Shared contract — Module 0 (do this first)

**File:** `shared/contracts.md`

```
POST /predict          (ml-service)
  in:  { "landmarks": [[[x,y,z], ... 21 points ...], ... frames ...] }
  rules: each frame has exactly 21 points; each point has numeric x, y, z;
         x/y are normalized to 0..1; missing-hand frames are all zeroes;
         the client sends no more than MAX_FRAMES (60) frames per request
  out: { "keywords": ["pay", "medicine", "?"] }

POST /process           (backend)
  in:  { "keywords": ["pay", "medicine", "?"] }
  out: { "sentence": "Can I pay for this medicine?", "audio_url": "https://.../audio/<id>.mp3" }

GET /audio/{file_id}.mp3  (backend)
  out: an audio/mpeg response that the mobile device can stream or download

POST /transcribe         (backend)
  in:  { "audio": "<base64 audio bytes>" }
  out: { "text": "Yes, that will be 200 rupees." }

POST /session/create          (backend)
  out: { "room_id": "abc123" }

POST /session/{room_id}/message     (backend)
  in:  { "sender": "signer" | "speaker", "text": "..." }
  out: { "status": "ok" }

GET /session/{room_id}/messages    (backend)
  out: { "messages": [ { "sender": "signer" | "speaker", "text": "...", "timestamp": "2026-01-01T00:00:00Z" }, ... ] }
```

**Test:** file exists with every example filled in exactly as above, including
the nested landmark shape, the absolute audio URL requirement, the sender
values, and the UTC timestamp format. No service code is written in this
module. Have the mobile developer review this file before implementing the
network layer; a contract change after mobile coding has started requires
updating the client, server, and contract test together.

---

## 4. ml-service modules (Builder-core)

### 4.0 ML execution strategy and mobile parity

The goal is a small, explainable baseline that the mobile client can reproduce.
Do not start with an LSTM or a large hosted vision model. First prove that the
same landmark representation can travel through the whole system.

For each sign label, capture several short recordings with the same camera
orientation and background as the demo. Keep recordings as separate samples;
do not split frames from one recording into both training and validation sets.
Record a manifest with label, recording filename, signer/session identifier,
and capture conditions so a later accuracy claim is reproducible. Do not commit
identifying or private recordings unless the team has permission.

Use these invariants in both the Python scripts and the mobile implementation:

- two stable hand slots (left/right), 21 landmarks per hand, followed by 68
  compact face landmarks: each frame is `(110, 3)`;
- x/y normalized by MediaPipe; preserve z rather than dropping it;
- a fixed sequence length of 60 frames, truncating a longer window and
  zero-padding a shorter window;
- all-zero 21-point blocks for missing hands and an all-zero 68-point block for
  a missing face; and
- the same frame sampling rate or an explicitly documented resampling rule.

The agent should create a tiny known-good fixture (for example, a saved array
from one permitted recording) and use it in unit and HTTP tests. This catches
the common failure where the desktop extraction script and the mobile camera
pipeline produce different nesting, frame counts, or coordinate ranges.

### M1 — Environment + webcam sanity check

**Tool:** MediaPipe Tasks Hand Landmarker + Face Landmarker, OpenCV.

**Install:**
```
python -m pip install mediapipe opencv-contrib-python numpy
```

Create or activate the project virtual environment before installing. On
Windows PowerShell, the typical sequence is
`py -m venv .venv`, then run `Activate.ps1` from the `.venv/Scripts` folder, then the install command; on
macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`. Keep
the chosen interpreter consistent for every later `pytest`, `uvicorn`, and
training command.

**What to write** (throwaway script, `scripts/00_sanity_check.py`):
```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = ...  # Tasks Hand Landmarker with num_hands=2; Face Landmarker runs alongside it

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    count = len(results.multi_hand_landmarks[0].landmark) if results.multi_hand_landmarks else 0
    print(f"landmarks detected: {count}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

> **Roadmap note:** `mp.solutions.hands` is MediaPipe's legacy interface. The implemented script uses Tasks Hand Landmarker plus Face Landmarker; its frame output is `(110, 3)`.

**How it works:** the Tasks Hand Landmarker returns up to two 21-point hands and the Face Landmarker must return 478 source points. M1 reports 68 selected points. M2 uses `FACE_FEATURE_VERSION="compact-v1"`, `FACE_LANDMARK_INDICES`, and `FACE_FEATURE_GROUPS`: concatenate the exact curated RIGHT_EYE, LEFT_EYE, RIGHT_EYEBROW, LEFT_EYEBROW, LIPS_OUTER, LIPS_INNER, NOSE indices documented in `shared/contracts.md`, preserving within-group order. Output order is left-hand, right-hand, compact face; missing blocks are zero-filled.

**Local asset requirement:** keep the hand asset at `ml-service/models/hand_landmarker.task` and place a compatible local `face_landmarker.task` at a path supplied through `--face-model`. The implementation intentionally does not download model assets or commit a face model automatically.

**Action:** first run this with a laptop webcam to prove the library and camera
permissions work. If the worker has no camera, add a deterministic image/video
fallback for CI rather than declaring the module passed. Always release the
camera and close the MediaPipe context in a `finally` block in the production
version.

**Test:** with one hand visible, console reports one detected hand; with two,
it reports two. Face detection reports `478` points when a face is visible and
`0` otherwise. With no hands, it reports zero hands without crashing. Also run one no-camera or prerecorded-fixture path so
automated builds do not depend on an interactive desktop device.

---

### M2 — Landmark extraction from a recorded video

**Files:** `scripts/extract_landmarks_module.py` (reusable function) and
`scripts/01_extract_landmarks.py` (command-line wrapper).

**Tool:** MediaPipe Tasks Hand + Face Landmarkers, now reading from a file instead of a live camera.

Put `extract_landmarks(video_path)` in the importable module and make the
numbered script import it, parse one input path, save the `.npy`, and print the
shape. This avoids the broken import pattern where M3 tries to import a
function from a filename that Python cannot import as a normal module. Validate
that the input opens, release the capture and MediaPipe resources in all exit
paths, and return an empty `(0, 110, 3)` array for a valid video with no frames
rather than crashing.

```python
import cv2
import mediapipe as mp
import numpy as np
import sys

mp_hands = mp.solutions.hands

def extract_landmarks(video_path: str) -> np.ndarray:
    hands = ...  # Tasks Hand Landmarker, num_hands=2; Face Landmarker runs alongside it
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            frame_points = [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark]
        else:
            frame_points = [[0.0, 0.0, 0.0]] * 21   # no hand detected this frame — pad with zeros
        sequence.append(frame_points)
    cap.release()
    return np.array(sequence)   # shape: (num_frames, 110, 3)

if __name__ == "__main__":
    arr = extract_landmarks(sys.argv[1])
    out_path = sys.argv[1].replace(".mp4", ".npy")
    np.save(out_path, arr)
    print(f"saved {out_path} with shape {arr.shape}")
```

**How to run:** `python scripts/01_extract_landmarks.py data/raw_videos/pay/pay_01.mp4 --model models/hand_landmarker.task --face-model models/face_landmarker.task`

**Test:** run on one sample recording. Confirm the printed shape is
`(num_frames, 110, 3)` where `num_frames` is roughly
`video_length_seconds × fps`. Also run a fixture with no detected hand and
confirm its frame entries are exactly 21 zero points, matching what the mobile
client will send.

---

### M3 — Dataset builder

**File:** `scripts/02_build_dataset.py`

**Tool:** NumPy only — this is pure data wrangling, no new API.

```python
import os
import numpy as np
from scripts.extract_landmarks_module import extract_landmarks  # reuse M2's function

RAW_DIR = "data/raw_videos"
MAX_FRAMES = 60   # fixed length every sequence gets padded/truncated to

def pad_or_truncate(seq, max_len=MAX_FRAMES):
    if len(seq) >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - len(seq), 110, 3))
    return np.concatenate([seq, pad], axis=0)

X, y = [], []
for sign_label in os.listdir(RAW_DIR):
    sign_folder = os.path.join(RAW_DIR, sign_label)
    for video_file in os.listdir(sign_folder):
        seq = extract_landmarks(os.path.join(sign_folder, video_file))
        X.append(pad_or_truncate(seq))
        y.append(sign_label)

X = np.array(X)   # shape: (num_samples, MAX_FRAMES, 110, 3)
y = np.array(y)
np.savez("model/dataset.npz", X=X, y=y)
print(f"dataset built: X={X.shape}, y={y.shape}")
```

**Why pad/truncate:** every sign recording is a different length in frames, but the classifier (Module M4) needs every input to be the exact same size. Fixing all sequences to `MAX_FRAMES` frames solves that.

**Action:** run the builder from the `ml-service/` directory or make all paths
explicitly relative to that directory. Sort labels and filenames so two runs
produce the same ordering, ignore non-video files, create `model/` if needed,
and fail with a clear message when a label folder is empty. Save the fixed
`MAX_FRAMES` value and label list beside the dataset (or in a small metadata
file) so inference cannot silently use a different preprocessing contract. Make
`scripts/` an importable package (add its minimal `__init__.py`) and use one
documented invocation style so the M2 import works both locally and in CI.

Each frame contains 21 left-hand points, 21 right-hand points, and 68 compact face
points; missing hand or face blocks remain zero-filled. Dataset construction
requires both local hand and face model assets (the face model is never
downloaded).

Folder and archive metadata contain `face_feature_version`, `face_landmark_indices`, `face_feature_groups`, `face_points=68`, and `face_source_points=478`. Rebuild old 520-point datasets. The Colab notebook retains its previous representation until updated separately.

**Test:** `len(X) == len(y)`, `X.shape[1:] == (60, 110, 3)`, and `set(y)` exactly
matches the folder names under `data/raw_videos/`. Print per-label counts and
retain that output as the dataset evidence for M4.

#### M3 WLASL archive mode

The workspace also contains a WLASL archive outside the cloned repository:
`E:\Desktop\Google Hackathon\archive (1)`. Do not copy its videos into
`ml-service/data/raw_videos`. Use the benchmark metadata directly so the
official signer-independent train/validation/test split is preserved.

The builder accepts:

```powershell
python scripts/02_build_dataset.py `
  --archive-root "E:\Desktop\Google Hackathon\archive (1)" `
  --benchmark 100 `
  --subset train `
  --max-samples 20 `
  --output ".\model\wlasl100-train.npz" `
  --model ".\models\hand_landmarker.task" `
  --face-model "C:\path\to\face_landmarker.task"
```

`--benchmark` may be `100`, `300`, `1000`, `2000`, or a benchmark JSON path.
The loader reads `wlasl_class_list.txt`, the selected `nslt_*.json`,
`WLASL_v0.3.json`, and `videos/`. It skips annotations whose video file is
missing, preserves `video_id`, `gloss`, `subset`, `signer_id`, and the
1-based inclusive benchmark frame range, and remaps sparse source class IDs to
contiguous dataset IDs. The output includes `dataset.manifest.json` without
absolute paths.

The extractor uses the benchmark action frame range before the existing
60-frame padding/truncation step. It does not copy or modify source videos.
Start with `--benchmark 100 --max-samples 20`, inspect the manifest, then scale
to the full benchmark after the face model and extraction throughput are
verified. The WLASL videos are ASL; do not claim Indian Sign Language accuracy
from this dataset.

For a Colab run over the archive, use
`notebooks/signbridge_wlasl_12k_colab.ipynb`. It is optimized for the full ~11,980 video
dataset with native T4 GPU acceleration, fast video seeking (`cv2.CAP_PROP_POS_FRAMES`),
frame downscaling, and safe crash/disconnect resuming (`RESUME=True`). It uploads/unzips
the archive, downloads local MediaPipe `.task` assets, maps benchmark class IDs to gloss words,
filters missing videos, crops frame ranges, and writes sharded NPZ files with the
`(batch, 60, 110, 3)` feature contract. It reports progress and ETA every 500 frames.

---

### M4 — Train the classifier

**File:** `scripts/03_train_classifier.py`

**Tool:** scikit-learn (start simple — a Random Forest on flattened landmark sequences trains in seconds and is easy to debug; only move to an LSTM/neural net if accuracy is poor and you have spare time).

**Install:** `pip install scikit-learn`

```python
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = np.load("model/dataset.npz", allow_pickle=True)
X, y = data["X"], data["y"]

X_flat = X.reshape(len(X), -1)   # flatten (frames, 21, 3) into one long vector per sample

X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_val, y_val)
print(f"validation accuracy: {accuracy:.2%}")

with open("model/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
```

**Action:** split by recording/sample, never by individual frames. Before
`stratify=y`, verify that every label has enough recordings for both train and
validation; otherwise stop and collect more data instead of weakening the
split. Print total samples, per-label counts, the random seed, validation
accuracy, and a confusion matrix. Save the classifier together with metadata
(`MAX_FRAMES`, coordinate format, labels, and training date/code version) so a
future model cannot be loaded with the wrong input shape.

**Test:** printed validation accuracy is ≥ 80%, the held-out set contains at
least one recording for every label, and `model/classifier.pkl` exists. Repeat
the training command once with the same seed and confirm the metric and label
set are stable.

**If below 80%:** do not proceed to M5. Record 15-20 repetitions per sign instead of fewer, keep hand position consistent in frame, then rerun M3 and M4.

---

### M5 — Inference wrapper

**File:** `service/inference.py`

```python
import pickle
from pathlib import Path
import numpy as np

MAX_FRAMES = 60
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "classifier.pkl"

with MODEL_PATH.open("rb") as f:
    _clf = pickle.load(f)

def predict(landmarks: list) -> list[str]:
    arr = np.array(landmarks)           # shape: (frames, 21, 3)
    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise ValueError("landmarks must have shape (frames, 21, 3)")
    if len(arr) >= MAX_FRAMES:
        arr = arr[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - len(arr), 21, 3))
        arr = np.concatenate([arr, pad], axis=0)
    flat = arr.reshape(1, -1)
    pred = _clf.predict(flat)
    return [pred[0]]
```

**Test** (`tests/test_predict.py`): load one known sample from `dataset.npz`, call `predict()` on its raw (un-flattened) landmark array, assert the returned label matches that sample's true label.
```python
import numpy as np
from service.inference import predict

def test_predict_known_sample():
    data = np.load("model/dataset.npz", allow_pickle=True)
    sample, true_label = data["X"][0], data["y"][0]
    assert predict(sample.tolist())[0] == true_label
```
Run: `pytest tests/test_predict.py`

Add boundary tests for a sequence shorter than 60 frames, exactly 60 frames,
longer than 60 frames, an all-zero missing-hand sequence, and an invalid shape.
The short and long cases must use the same pad/truncate helper as dataset
building; do not duplicate subtly different preprocessing in the API layer.
Import the model path relative to `service/inference.py` (not the caller's
current working directory) and fail with a clear startup error if the trained
artifact is missing.

---

### M6 — `/predict` API endpoint

**File:** `service/app.py`

**Tool:** FastAPI + Uvicorn.

**Install:** `pip install fastapi uvicorn pydantic`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from service.inference import predict

app = FastAPI()

class LandmarkInput(BaseModel):
    landmarks: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(payload: LandmarkInput):
    keywords = predict(payload.landmarks)
    return {"keywords": keywords}
```

Add request validation at this boundary so malformed mobile payloads receive a
useful `422` instead of a model traceback. Validate the nested shape, numeric
coordinates, frame limit, and a reasonable maximum JSON body size. Return a
stable error shape and keep `/health` cheap; health checks must not run a model
prediction or call a cloud service.

**Run locally:** `uvicorn service.app:app --reload --port 8080`

**Test:**
```
curl -X POST localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/known_sample.json
```
Confirm response is `{"keywords": [...]}` with HTTP 200. Also send an empty
payload, a frame with 20 or 22 points, and 61 frames; each must be rejected
without taking down the process. Use FastAPI's `TestClient` for this test in
CI and one real curl request for the local smoke test.

---

### M7 — Containerize and deploy

**File:** `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Create `requirements.txt` with pinned, compatible runtime dependencies and a
`.dockerignore` that excludes `.venv`, tests' local recordings, secrets, and
unneeded caches. Build and run the image locally before using Cloud Run:

```text
docker build -t signbridge-ml ./ml-service
docker run --rm -p 8080:8080 signbridge-ml
```

Deployment itself is covered in section 6 (GitHub → Cloud Run via Console) —
both `ml-service` and `backend` deploy the same way. The container must start
from a clean checkout with only its declared files and environment variables;
it must not depend on a developer's current working directory or untracked
model file.

**Test (after deploying via section 6):** hit the live URL's `/predict` with the same payload from M6 — confirm identical response. Save this URL; `backend` will never call it directly (the mobile app calls `ml-service` directly for the sign→keyword step, then sends keywords to `backend` for the rest) — confirm this matches how your mobile app team expects to call it, and adjust the contract in `shared/contracts.md` if not.

---

## 5. backend modules (Builder-integration)

### 5.0 Backend execution strategy

Build the backend as a thin orchestration layer: validate input, call one
service client, persist or expose the result, and return the contract shape.
Keep Gemini, Text-to-Speech, Speech-to-Text, and Firestore behind small modules
with the same function signatures in real and fake implementations. Use
`MOCK_EXTERNAL_SERVICES=1` (or the project's equivalent) for deterministic
local and CI tests; the mock must be explicit and must never be enabled in the
deployed demo by accident.

For every endpoint, test four paths:

1. valid request and expected response;
2. missing or malformed input (`4xx`, stable JSON error);
3. downstream empty/timeout/failure (`5xx` or a documented graceful response);
4. a real smoke request with the cloud client enabled.

Use timeouts and structured logs that contain request IDs, endpoint names, and
durations, but never log API keys, raw audio, camera frames, or full private
conversation text. Keep audio files in a dedicated temporary directory and
return a URL served by the backend; returning `/tmp/foo.mp3` as a response is
not a mobile integration.

### M1 — Skeleton + health check

**File:** `main.py`

```python
from fastapi import FastAPI
from routes import process, transcribe, session

app = FastAPI()
app.include_router(process.router)
app.include_router(transcribe.router)
app.include_router(session.router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Action:** create the package directories and `__init__.py` files required by
the imports before starting Uvicorn. Keep router imports free of network calls
at module import time so a health check can start in mock mode.

**Test:** `uvicorn main:app --reload --port 8080` then
`curl localhost:8080/health` → `{"status": "ok"}`. Add a `TestClient` health
test so the same check runs without a long-lived development server.

---

### M2 — Gemini client

**File:** `services/gemini_client.py`

**Tool:** `google-generativeai` (Gemini API via the AI Studio key from section 2, step 5).

**Install:** `pip install google-generativeai`

**Action:** make the model name configurable with `GEMINI_MODEL`, load the
client lazily or behind a factory, and provide a clearly enabled mock path for
tests that do not have a key. Validate that the keyword list is short and
contains strings before putting it into the prompt. Ask for one sentence only,
strip markdown/code fences from the result if the provider returns them, and
raise a useful error for an empty response. The API route should translate a
provider failure into a controlled `502`, not expose the provider traceback.

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-2.0-flash")

def keywords_to_sentence(keywords: list[str]) -> str:
    prompt = (
        "Turn this keyword sequence into one natural, grammatically correct "
        f"spoken sentence. Keywords: {keywords}. "
        "Output only the sentence, nothing else."
    )
    response = _model.generate_content(prompt)
    return response.text.strip()
```

**Set the key locally for testing:** `export GEMINI_API_KEY=your_key_here`
(Linux/Mac), or set it in a local untracked `.env` and load with
`python-dotenv`. In Windows PowerShell the equivalent is
`$env:GEMINI_API_KEY = "<local-key>"`. Never paste the key into this guide,
mobile configuration, a test fixture, or a commit.

**Test:** call `keywords_to_sentence(["pay", "medicine", "?"])`, assert the returned string is non-empty and contains "pay" or "medicine" in some form.

Run the same test with `MOCK_EXTERNAL_SERVICES=1` first, then run one real
provider test only when the credential is available. Record the selected model
name and latency, but not the secret or full request payload.

> **Roadmap note:** the model name and SDK in the snippet above are examples. Before pinning, check Google's current model list and SDK guidance and set `GEMINI_MODEL` accordingly. Phase 4 moves this client to Vertex AI with Application Default Credentials so no API key is needed.

*(Vertex AI alternative, only if you chose that path in section 2: swap this file's contents for the `vertexai.generative_models` client — same function signature, so nothing else in the codebase changes.)*

---

### M3 — Text-to-Speech client

**File:** `services/tts_client.py`

**Tool:** `google-cloud-texttospeech`.

**Install:** `pip install google-cloud-texttospeech`

**Action:** keep client construction out of import-time health checks when
possible, select the voice and language through configuration, and provide a
fake MP3 fixture for local endpoint tests. The fake must be recognizable as
test data; it must not make the mobile client skip its audio playback check.
The real path must return non-empty MP3 bytes and a content type that mobile
audio libraries understand.

```python
from google.cloud import texttospeech

_client = texttospeech.TextToSpeechClient()

def text_to_speech(sentence: str) -> bytes:
    input_text = texttospeech.SynthesisInput(text=sentence)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = _client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content   # raw MP3 bytes
```

**Note:** `_client = texttospeech.TextToSpeechClient()` authenticates automatically using the Cloud Run service account from section 2 (no key file needed) — this only works once deployed, or locally if you've run `gcloud auth application-default login`.

**Test:** call with a sample sentence, confirm the returned bytes are non-empty (`len(audio_bytes) > 0`) and, if you save them to a `.mp3` file, that they actually play.

Also verify that a mock-mode `/process` response points to a URL which returns
`audio/mpeg`, not a local path. Run the real Text-to-Speech call once before
deployment and record its latency and selected voice.

---

### M4 — `/process` endpoint

**File:** `routes/process.py`

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from services.gemini_client import keywords_to_sentence
from services.tts_client import text_to_speech
from pathlib import Path
import os
import uuid

router = APIRouter()
AUDIO_DIR = Path("/tmp/signbridge-audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class ProcessInput(BaseModel):
    keywords: list[str]

@router.post("/process")
def process(payload: ProcessInput, request: Request):
    sentence = keywords_to_sentence(payload.keywords)
    audio_bytes = text_to_speech(sentence)
    file_id = uuid.uuid4().hex
    filename = AUDIO_DIR / f"{file_id}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    public_base_url = os.getenv("PUBLIC_BASE_URL", str(request.base_url)).rstrip("/")
    audio_url = public_base_url + f"/audio/{file_id}.mp3"
    return {"sentence": sentence, "audio_url": audio_url}

@router.get("/audio/{file_id}.mp3")
def audio(file_id: str):
    filename = AUDIO_DIR / f"{file_id}.mp3"
    if not filename.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(filename, media_type="audio/mpeg")
```

Import `HTTPException` in the implementation and reject empty or excessively
long keyword lists. Set `PUBLIC_BASE_URL` to the deployed backend's HTTPS URL
so the response is not accidentally built from an internal or HTTP base URL.
For the MVP, `/tmp` is acceptable only because the audio
is short-lived and served by the same Cloud Run instance; it is not durable
storage. The production upgrade is Cloud Storage plus a signed URL, with the
same response contract.

**Test:** `curl -X POST localhost:8080/process -H "Content-Type: application/json" -d '{"keywords": ["pay","medicine","?"]}'` — confirm both `sentence` and an absolute `audio_url` come back non-empty. Then `curl` the returned URL, confirm HTTP 200, `Content-Type: audio/mpeg`, non-empty bytes, and actual playback. Test the missing-file URL and an empty keyword list as well.

---

### M5 — Speech-to-Text client

**File:** `services/stt_client.py`

**Tool:** `google-cloud-speech`.

**Install:** `pip install google-cloud-speech`

**Action:** agree on the recording format with the mobile developer before
testing the client: mono, 16,000 Hz, LINEAR16 PCM in a valid WAV container.
Inspect a test file's header (sample rate, channels, sample width, duration)
before sending it. Keep a mock transcript implementation for endpoint tests,
but run one real clip through the cloud client before deployment. Bound the
maximum audio duration/byte size so a phone cannot accidentally submit an
unlimited recording.

```python
from google.cloud import speech

_client = speech.SpeechClient()

def speech_to_text(audio_bytes: bytes) -> str:
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-IN",
    )
    response = _client.recognize(config=config, audio=audio)
    if not response.results:
        return ""
    return response.results[0].alternatives[0].transcript
```

**Important:** the mobile app must record audio as 16kHz mono LINEAR16 (raw PCM WAV) to match `sample_rate_hertz=16000` above — mismatched encoding is the most common Speech-to-Text bug. Confirm this with whoever builds the mobile app.

**Test:** record yourself saying "yes that's fine" as a 16kHz WAV file, read its bytes, call `speech_to_text()`, assert the returned text reasonably matches.

Repeat the test with silence and with an empty/invalid file. Silence should
produce a documented empty transcript rather than a server crash; invalid
encoding should produce a controlled client error. Do not log or commit the
recorded speech.

---

### M6 — `/transcribe` endpoint

**File:** `routes/transcribe.py`

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import binascii
from services.stt_client import speech_to_text

router = APIRouter()

class TranscribeInput(BaseModel):
    audio: str   # base64-encoded audio bytes

@router.post("/transcribe")
def transcribe(payload: TranscribeInput):
    try:
        audio_bytes = base64.b64decode(payload.audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="audio must be valid base64") from exc
    text = speech_to_text(audio_bytes)
    return {"text": text}
```

Import `HTTPException`, reject empty and oversized decoded audio, and translate
downstream Speech-to-Text failures into a stable `502` response. Do not return
the original audio in the response.

**Test:** base64-encode the same test WAV from M5, POST it, confirm the correct
text comes back through the actual HTTP endpoint (not just the raw function).
Also test invalid base64, empty audio, an oversized body, silence, and mock
mode. Verify the route's response can be inserted into the mobile session
timeline without a timestamp or sender ambiguity.

---

### M7 — Firestore client

**File:** `services/firestore_client.py`

**Tool:** `google-cloud-firestore`.

**Install:** `pip install google-cloud-firestore`

```python
from google.cloud import firestore
import uuid
from datetime import datetime, timezone

_db = firestore.Client()

def create_room() -> str:
    room_id = str(uuid.uuid4())[:8]
    _db.collection("rooms").document(room_id).set({"created_at": datetime.now(timezone.utc)})
    return room_id

def add_message(room_id: str, sender: str, text: str) -> None:
    _db.collection("rooms").document(room_id).collection("messages").add({
        "sender": sender,
        "text": text,
        "timestamp": datetime.now(timezone.utc),
    })

def get_messages(room_id: str) -> list[dict]:
    docs = (
        _db.collection("rooms").document(room_id)
        .collection("messages").order_by("timestamp").stream()
    )
    return [d.to_dict() for d in docs]
```

**Test:** `create_room()` → `add_message(room_id, "signer", "hello")` → `get_messages(room_id)` → assert the message you added is present with matching sender/text.

Run this first against a fake repository or the Firestore emulator so unit
tests do not require production data. Then run one real Firestore smoke test
with Application Default Credentials. Convert Firestore timestamp objects to
UTC ISO-8601 strings at the API boundary; the mobile app should never need to
understand a provider-specific timestamp type. Verify room IDs are not
guessable enough for a production product, and document that authentication is
still a hackathon hardening task if the endpoints remain public.

---

### M8 — `/session` endpoints

**File:** `routes/session.py`

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from services.firestore_client import create_room, add_message, get_messages

router = APIRouter()

@router.post("/session/create")
def session_create():
    return {"room_id": create_room()}

class MessageInput(BaseModel):
    sender: Literal["signer", "speaker"]
    text: str = Field(min_length=1, max_length=2000)

@router.post("/session/{room_id}/message")
def session_message(room_id: str, payload: MessageInput):
    add_message(room_id, payload.sender, payload.text)
    return {"status": "ok"}

@router.get("/session/{room_id}/messages")
def session_messages(room_id: str):
    return {"messages": get_messages(room_id)}
```

Reject unknown sender values, blank text, oversized text, and messages for a
missing room with stable `4xx` responses. Serialize every returned timestamp
to the ISO-8601 format in `shared/contracts.md`, and return messages in oldest
first order so the mobile timeline is deterministic.

**Test:** POST `/session/create` → grab `room_id` → POST a message from each
sender to `/session/{room_id}/message` → GET `/session/{room_id}/messages` →
assert both messages appear in order via the actual HTTP endpoints. Repeat the
test with mock Firestore in CI and real Firestore before deployment.

---

### M9 — Containerize

**File:** `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Deployment covered in section 6 below.

---

## 6. Deployment — GitHub repo → Google Cloud Console (Cloud Run)

You have one GitHub repo containing both `ml-service/` and `backend/` as subfolders. Set up **two separate Cloud Run services**, each pointing at the same repo but built from a different subfolder.

**For each service (repeat once for `ml-service`, once for `backend`):**

1. Push your repo to GitHub first (`git push origin main`) — Cloud Build needs to see it.
2. In Google Cloud Console, go to **Cloud Run** → **Create Service**.
3. Choose **"Continuously deploy from a repository"** → click **Set up with Cloud Build**.
4. Click **Connect Repository** → authenticate GitHub if prompted → select your repo → select branch `main`.
5. **Build configuration:** choose **Dockerfile**. Under **Source** / **Build context directory**, set it to `/backend` or `/ml-service` (whichever service you're setting up) — this tells Cloud Build which subfolder's `Dockerfile` to use, so one repo can drive two independent services.
6. Set **Service name** (e.g. `signbridge-backend` / `signbridge-ml`), pick your region (match your Firestore region, e.g. `asia-south1`).
7. Under **Authentication**, select **Allow unauthenticated invocations** (needed so your mobile app can call it without a Google auth token — fine for a hackathon demo, tighten later for production).
8. Under **Variables & Secrets** (only needed for `backend`): add the
   `GEMINI_API_KEY` secret and the non-secret `PUBLIC_BASE_URL` set to the
   backend's HTTPS service URL. Prefer **Secrets** over plain **Variables** —
   click **Reference a Secret**, create one via **Secret Manager** (Console:
   **Security → Secret Manager → Create Secret**), paste your key there, then
   reference it here.
9. Click **Create**. Cloud Build will pull the repo, build the Dockerfile from your specified subfolder, and deploy. You'll get a public HTTPS URL — this is what the mobile app calls, and what module tests in sections 4/5 run their final "deployed" test against.
10. From now on, every `git push` to `main` auto-triggers a rebuild and redeploy of both services. To avoid rebuilding `ml-service` every time you only touched `backend` (and vice versa), edit the trigger: **Cloud Build → Triggers → (your trigger) → Edit → Included files filter** → set to `backend/**` or `ml-service/**` respectively.

### 6.1 Deployment verification and mobile handoff

Do not hand the URLs to the mobile team immediately after clicking **Create**.
For each Cloud Run revision:

1. Open `/health` and confirm HTTP 200.
2. Send the known landmark fixture to the deployed ML URL and compare the
   keyword response with the local container.
3. Send the keywords to the deployed backend URL, fetch the returned
   `audio_url` from a network outside the developer machine, and verify it is
   HTTPS, HTTP 200, `audio/mpeg`, and playable.
4. Run `/transcribe` with the permitted test WAV and verify the response.
5. Create a room, write one message from each sender, and verify the ordered
   ISO timestamps.
6. Inspect Cloud Run logs for startup errors, provider errors, excessive
   latency, or accidental secret/audio logging. Delete failed revisions only
   after saving the useful error evidence.

Give the mobile team a small, non-secret configuration note containing:

```text
ML_BASE_URL=https://<ml-service-host>
BACKEND_BASE_URL=https://<backend-service-host>
```

The mobile client should append endpoint paths itself and avoid double slashes.
Use separate development and production values, and never hard-code a
developer laptop URL into a released build. If the mobile app is tested on a
physical device against a local service, bind the service to a reachable local
interface and document the device's network address; do not treat that as the
deployed smoke test.

Cloud Run's local filesystem is ephemeral and instance-local. The `/tmp` audio
route is a demo shortcut only: a later production version must move audio to
Cloud Storage or another durable object store and keep the same `audio_url`
contract. Public unauthenticated endpoints are also a hackathon choice; add
authentication, rate limiting, and abuse controls before real users or private
conversation data are allowed.

---

## 7. Progress checklist

**shared**
- [ ] M0 — `contracts.md` written and reviewed by the mobile developer
- [ ] Preflight — project, region, APIs, credentials/mock mode, device, and
      data permissions recorded

**ml-service**
- [ ] M1 — two-hand + face webcam/MediaPipe sanity check plus no-camera fixture path (face asset/device check pending)
- [x] M2 — reusable two-hand + face landmark extraction module and CLI (unit tests; real face extraction pending local face asset)
- [x] M3 — deterministic `(N, 60, 520, 3)` dataset builder with shape/count evidence (unit tests; no training-quality claim)
- [ ] M4 — classifier trained, held-out accuracy ≥80%, metadata saved
- [ ] M5 — inference wrapper and boundary unit tests passing
- [ ] M6 — validated `/predict` endpoint working locally
- [ ] M7 — clean container builds and live `/predict` matches locally

**backend**
- [ ] M1 — `/health` working locally and in mock mode
- [ ] M2 — Gemini client tested with mock and real provider paths
- [ ] M3 — TTS client tested and playable MP3 fixture available
- [ ] M4 — `/process` returns a phone-reachable audio URL
- [ ] M5 — STT client tested against a format-verified WAV
- [ ] M6 — `/transcribe` validates base64 and works through HTTP
- [ ] M7 — Firestore client tested with fake/emulator and real smoke path
- [ ] M8 — `/session` endpoints validate sender/text and serialize timestamps
- [ ] M9 — deployed via Cloud Run, all endpoints re-verified live

**mobile readiness**
- [ ] Base URLs are configured per build environment; no secrets are present
- [ ] Camera and microphone permissions have denial/retry states
- [ ] On-device landmark output matches `(frames, 21, 3)` and MAX_FRAMES=60
- [ ] Signer flow receives keywords, sentence, and playable MP3
- [ ] Speaker flow sends 16 kHz mono LINEAR16/WAV and displays transcript
- [ ] Session flow creates a room, sends both sender types, and polls messages
- [ ] Physical-device smoke test passes on the selected demo platform
- [ ] Offline, timeout, invalid-payload, and service-error states are visible

---

**post-MVP (see Roadmap phases 4–6)**
- [ ] Phase 4 — audio on Cloud Storage with signed URLs
- [ ] Phase 4 — least-privilege service accounts; Editor role removed
- [ ] Phase 4 — Gemini on Vertex AI (ADC); no API key remains
- [ ] Phase 4 — rate limits, max instances, budget alerts, uptime checks
- [ ] Phase 4 — CI runs tests before deploy; privacy/retention note written
- [ ] Phase 5 — held-out *signer* accuracy ≥85%; unknown-class threshold in place
- [ ] Phase 6 — Firebase Auth + App Check; backend verifies ID tokens
- [ ] Phase 6 — realtime timeline, per-session languages, Crashlytics, pilot findings logged

---

## 8. Final integration test

Run this in order using the deployed URLs and a physical device where noted:

1. Create a session and retain `room_id`.
2. Capture a short landmark sample for a known sign with the mobile pipeline →
   POST to live `ml-service /predict` → confirm HTTP 200 and an expected label.
3. Take the returned keywords → POST to live `backend /process` → confirm the
   sentence is sensible, the URL is absolute HTTPS, and the downloaded audio
   actually plays on the phone.
4. Send the sentence as a `signer` message; poll messages and confirm it is
   visible with a UTC ISO timestamp.
5. Record a short 16 kHz mono WAV on the phone → base64-encode → POST to
   `/transcribe` → confirm the text is reasonable and add it as a `speaker`
   message.
6. Poll the room from the second client (or a second device) and confirm both
   sides see the same ordered timeline.
7. Repeat once with denied permissions, no hand detected, a network timeout,
   invalid base64, and a service error. Confirm the app shows recovery actions
   rather than spinning forever or silently losing a message.

Only once all seven checks pass is the backend/ML half ready for the mobile app
demo. Save the exact service revisions, model version/accuracy, test device
and OS, and any known limitations in the project handoff note.

## 9. What the agent should report after completing the build

The completion report should be factual and reproducible:

- files created or changed, grouped by shared, ML, backend, and mobile handoff;
- commands actually run and their results (including Python, Docker, pytest,
  curl, and deployed URL checks);
- dataset size, labels, preprocessing shape, model version, and held-out
  accuracy;
- local and deployed service URLs, with secrets omitted;
- mobile platform/device smoke-test result and any permission/network caveat;
- failed or blocked checks, with the one concrete next action needed; and
- the next safe improvement, normally durable audio storage, authentication,
  rate limiting, or a larger diverse dataset—not an unverified model claim.

Never report “model complete” based only on a training metric or a local curl
response. Report the full vertical slice and clearly separate real evidence
from mocked or unavailable integrations.

---

## 10. Google tooling map (MVP choice → upgrade path)

| Need | MVP choice | Upgrade path | Why it fits SignBridge |
|---|---|---|---|
| Hand landmarks | MediaPipe Hand Landmarker (desktop Python, mobile Tasks SDK) | same | Same 21-point output on desktop and phone keeps training and inference consistent |
| Mobile app | Android-first (Kotlin) or Flutter if the team knows it | Flutter for cross-platform once MediaPipe plugin is proven | MediaPipe's first-party mobile support is strongest on native Android/iOS; Flutter needs a plugin or platform channel |
| Model training | local venv + scikit-learn | Colab → TensorFlow/Keras → TFLite | Cheap, shareable notebooks; clean path to on-device inference |
| Dataset/recordings | private Cloud Storage bucket + manifest in Sheets | Storage lifecycle rules, per-signer consent field | Keeps video out of git and shared drives |
| Sentence generation | Gemini (AI Studio key) | Gemini on Vertex AI with ADC | Removes the API key; same prompt and function signature |
| Speech → text | Cloud Speech-to-Text, `en-IN`, LINEAR16 16 kHz | configurable language per session | Already planned; just parametrize |
| Text → speech | Cloud Text-to-Speech, `en-IN`, MP3 | configurable voice/language per session | Already planned |
| Audio delivery | `/tmp` on Cloud Run (demo only) | Cloud Storage + signed URLs | Fixes cross-instance 404s; contract unchanged |
| Conversation store | Firestore, polling via backend | Firestore realtime listeners + security rules | Removes polling and backend hop for reads |
| Hosting | Cloud Run (two services) | same, with dedicated service accounts and max/min instances | Scales to zero; cheap for demos |
| Secrets | Secret Manager | same; then no API keys at all | Keeps credentials out of images and repo |
| CI/CD | Cloud Build trigger per service with path filters | `cloudbuild.yaml` with a test step; Artifact Registry | Prevents deploying a build that fails tests |
| Auth / abuse | public endpoints (hackathon only) | Firebase Auth + App Check + rate limits | Needed before real users or private conversations |
| Monitoring | Cloud Run logs | Cloud Logging, Monitoring uptime checks, Error Reporting, Crashlytics | Early warning for provider failures and app crashes |
| Tracking/test log | Google Sheets | Looker Studio dashboard over the Sheet (optional) | Zero-setup evidence trail |

---

## 11. Open decisions and risks

### 11.1 Decisions to record before Phase 1

| ID | Decision | Recommended default | Owner |
|---|---|---|---|
| D1 | Mobile stack and first demo platform | Android (Kotlin) unless the mobile dev is already productive in Flutter | Mobile dev |
| D2 | Initial sign vocabulary (5–8 signs) and the demo script that uses only those signs | Choose signs for the target scenario (e.g. a pharmacy or shop conversation) | Product + ML |
| D3 | Sign language/variant being captured and who the signers are | Record who and under what permission in the manifest | Product |
| D4 | Gemini route for the demo | AI Studio key (fast); migrate to Vertex AI in Phase 4 | Backend |
| D5 | Demo fallback policy (fixture replay, mock mode) and how it will be disclosed | Allowed, but labeled | Whole team |

### 11.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Accuracy is high on the team's own recordings but fails on other people | High | High | Evaluate on a held-out signer (Phase 5); show vocabulary limits honestly |
| Desktop and mobile landmark pipelines disagree on shape, rate, or ranges | Medium | High | Shared fixture and parity check in Phase 1 before any network wiring |
| Cold start of `ml-service` on demo day | Medium | Medium | Min instance = 1 for the demo |
| `/tmp` audio 404s across Cloud Run instances | Medium | Medium | Max instances low for the demo; Cloud Storage in Phase 4 |
| Public endpoints abused or run up cost | Medium | Medium | Budget alerts, max instances, request limits, then Auth + App Check |
| Gemini model/SDK names or limits change | Medium | Low–Medium | Config-driven `GEMINI_MODEL`; verify against Google docs in Phase 4 |
| Venue network or provider outage at demo | Medium | High | Labeled fixture/mock fallback and a recorded backup demo |
| Privacy of recorded signers and conversations | Medium | High | Consent in manifest, private bucket, no raw audio/video in logs or git |

---

## 12. Revision notes

- Added the **Roadmap** section (R.1–R.4), sections 10–12, and a post-MVP
  checklist in section 7.
- Left sections 0–9 intact except for three short "Roadmap note" callouts where
  the roadmap changes earlier advice (Editor role, Gemini model/SDK, legacy
  MediaPipe API).
- Nothing in the roadmap changes the API contract in section 3. Future changes
  (top-k confidences, two-hand shape, auth headers) are marked as additive or
  versioned.
