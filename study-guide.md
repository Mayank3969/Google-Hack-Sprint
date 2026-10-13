# SignBridge study guide

> A presentation-style guide to the working local architecture, contracts,
> formulas, setup, tests, and limits. Read this beside
> [`shared/contracts.md`](shared/contracts.md),
> [`ml-service/isl-contract.md`](ml-service/isl-contract.md), and the service
> READMEs when making an integration change.

---

## Page 1 — The product in one sentence

SignBridge helps a signer and a speaking conversation partner communicate. A
camera capture becomes a sequence of isolated sign candidates; the ML service
returns a result for every candidate; the backend turns a **complete and
reviewable** sequence into bilingual text and ordered audio; microphone speech
is transcribed back into the same conversation room.

The key product boundary is deliberate: the ML service recognizes isolated
words and abstains when evidence is weak. It does not invent a sentence or
silently fill an uncertain slot. The backend owns language generation, Hindi
translation, text-to-speech, speech-to-text, and conversation persistence.

Current local tests use deterministic model/provider doubles. They prove
plumbing and rejection behavior, not sign recognition accuracy, translation
quality, or production availability.

---

## Page 2 — The user journey

The intended live journey is:

1. Open the frontend in explicit live mode and create or join a room.
2. Enable camera access and capture one recording containing one or more sign
   candidates.
3. Stop the capture. The client sends the complete segmented capture to
   `POST /v2/predict/batch`.
4. Review the indexed ML results. Every slot must be recognized before the
   phrase can cross the sentence boundary.
5. Send the complete ordered result to `POST /process`.
6. Display the English and Hindi subtitles and let the user play the declared
   audio order.
7. Save the generated sentence as a `signer` message.
8. Record a short 16 kHz mono PCM16 WAV, call `/transcribe`, let the user review
   the transcript, and save it as a `speaker` message.

An uncertain, invalid, missing, overflow, or failed segment means **repeat the
whole recording**. Identified words may remain visible for review, but a
partial phrase is never sent for sentence generation.

---

## Page 3 — System architecture

```text
  Camera/video + microphone
              │
              ▼
  Frontend (Vite/React/TypeScript)
      │ landmarks only                 │ WAV/text/session requests
      ▼                                 ▼
  ML service                         Backend
  FastAPI + Keras                   FastAPI + provider adapters
      │                                 │
      │ ordered statuses                 ├─ Gemini sentence/translation
      └──────────────► frontend ────────┼─ Cloud TTS
                                        ├─ Cloud STT
                                        └─ Firestore
```

The frontend is an untrusted public client. Its build-time environment values
are endpoint URLs only. Provider credentials, Firestore access, Gemini keys,
and service-account files belong on the backend/deployment side and never in
the browser.

The ML service has no dependency on the backend conversation store. The backend
does not orchestrate model calls. This separation makes each boundary
testable, limits private data movement, and lets the client decide when the
user has reviewed a result.

---

## Page 4 — Ownership by service

### Frontend

The frontend owns permissions, capture lifecycle, landmark serialization,
batch request correlation, review state, playback controls, microphone WAV
creation, and recoverable error messages. It must preserve a draft after a
recoverable failure and must not retry an uncertain write automatically.

### ML service

The ML service owns raw landmark validation, missing-data quality measurement,
frozen preprocessing, model inference, confidence/margin policy, per-slot
status, batch ordering, and readiness reporting. It never calls Gemini/TTS/STT
and never writes a conversation message.

### Backend

The backend owns the sentence boundary, bilingual response validation, audio
storage/URLs, provider deadlines and error mapping, transcription validation,
room lifecycle, and UTC message serialization. It accepts an all-or-nothing
capture result and returns `record_again` when a phrase is incomplete.

---

## Page 5 — Capture data: 110 points per frame

Every frame sent to the v2 ML boundary has shape `(110, 3)`:

| Block | Points | Coordinates |
|---|---:|---:|
| left hand | 21 | 63 |
| right hand | 21 | 63 |
| compact face | 68 | 204 |
| **total** | **110** | **330** |

The coordinates are XYZ values in the frozen extractor order. Missing hands and
face blocks are zero-filled by the extraction boundary. JSON clients use
`null` for missing individual coordinates when needed; non-standard JSON `NaN`
is not a portable wire value. The client sends landmarks, never video bytes or
a video URL, to the ML service.

The compact face selection is versioned as `compact-v1` in the shared contract.
Its exact source groups are right eye, left eye, right eyebrow, left eyebrow,
outer lips, inner lips, and nose. A change to ordering is a model/data contract
change, not a cosmetic refactor.

---

## Page 6 — Fixed-rate sampling and segmentation

The reference extractor targets 15 FPS. Motion thresholds are measured in
frames, so irregular browser timing must be adapted before segmentation. The
segmenter finds candidate signs using hand motion, pause confirmation, and
tracking-loss rules. It keeps pre-roll and post-roll frames so onset and held
poses are not discarded.

The important limits are:

| Limit | Value | Meaning |
|---|---:|---|
| recording | 30 seconds | at most 450 raw frames |
| candidates | 10 | maximum batch slots |
| candidate model input | 60 positions | temporal resampling, not capture truncation |
| minimum candidate | 6 frames | short jitter is rejected |
| maximum candidate | 60 retained frames | overflow requests a repeat |

An overflow or mid-sign tracking loss is a failed candidate. Static signs are
hard for a motion boundary detector; a manual commit or a vocabulary-specific
capture cue is still needed if static poses are central to the product.

---

## Page 7 — The batch contract

`POST /v2/predict/batch` accepts:

```json
{
  "capture_id": "capture-2026-01",
  "segments": [
    {
      "segment_index": 1,
      "segment_id": "seg-1",
      "landmarks": [[[0.1, 0.2, 0.0]]]
    }
  ]
}
```

The example is abbreviated; a real frame has 110 points. There are 1–10
segments, unique indices from 0–9, optional unique IDs, and at most 450 total
raw frames. Gaps are allowed so a client can identify a failed slot for review.

The service returns one result per submitted segment, sorted by
`segment_index`, regardless of request order:

```json
{
  "capture_id": "capture-2026-01",
  "complete": true,
  "keywords": ["pay"],
  "results": [],
  "integration_ready": false,
  "warnings": []
}
```

The actual result carries status, keyword, confidence, top-k diagnostics,
quality, model version, and preprocessing version.

---

## Page 8 — Batch ordering and all-or-nothing semantics

The server sorts by index before preparing results. Valid slots are stacked for
one model call; invalid or low-quality slots remain in their original indexed
positions. Repeated predicted labels are legitimate: “hello hello” is not
deduplicated by the ML service.

`complete` means every submitted slot is accepted under the current policy. If
one slot is uncertain, invalid, insufficient, tracking-lost, or overflowed,
the response has `complete: false` and an empty `keywords` list. The frontend
must keep the result list for diagnostics/review and repeat the full recording.

Two independent gates matter:

```text
safe to form a phrase = complete AND every result.status == recognized
release evidence       = integration_ready
```

The current service deliberately reports `integration_ready: false` while
official runtime reevaluation, threshold calibration, and real multi-sign
evidence remain open. A successful HTTP 200 is therefore not a quality claim.

---

## Page 9 — Frozen model preprocessing

For a raw segment `X` with shape `(T, 110, 3)`, flatten each frame to 330
coordinates. Select 60 positions across the full segment:

```text
indices = np.linspace(0, T - 1, 60).astype(int)
```

For each coordinate `j`, compute a temporal mean over observed selected values
and center the observed values:

```text
mean_j = nanmean(X_selected[:, j])
N[t, j] = X_selected[t, j] - mean_j
N[t, j] = 0 for missing values
```

Then calculate first differences:

```text
V[0] = N[0] - N[0] = 0
V[t] = N[t] - N[t-1]
```

The model tensor is the concatenation `[N, V]`, shape `(60, 660)`, float32.
The browser does not calculate velocity and must not add wrist normalization,
timestamp division, interpolation, or masks as extra model channels without a
new versioned training contract.

---

## Page 10 — Quality, confidence, and abstention

Quality is measured from the original observation mask before temporal
centering. Useful values include coordinate coverage, left/right/face coverage,
valid frames, hand-valid frames, and `has_signal`.

The current development policy is intentionally provisional:

```text
minimum hand-valid frames = 3
minimum coordinate coverage = 0.05
minimum top-1 confidence = 0.50
minimum top-1 margin = 0.05
```

For top-1 score `p1` and runner-up score `p2`, the margin is:

```text
margin = p1 - p2
```

A word is accepted only when quality passes, `p1 >= threshold`, and
`p1 - p2 >= margin_threshold`. Softmax scores are not calibrated correctness
probabilities: an unknown sign can still score highly. All-zero, face-only,
static-informationless, and otherwise unusable buffers must not become fake
words.

---

## Page 11 — Result statuses and recovery

The ML status vocabulary is:

| Status | Meaning | Client action |
|---|---|---|
| `recognized` | accepted keyword and evidence | keep for review |
| `uncertain` | confidence or margin too low | repeat whole recording |
| `insufficient_data` | missing/too little/static evidence | improve capture and repeat |
| `tracking_lost` | detector loss invalidated candidate | repeat whole recording |
| `invalid` | malformed candidate payload | fix capture/request and repeat |
| `overflow` / `partial` | segmentation candidate outside safe boundary | repeat whole recording |

The client may display recognized words while showing which slot needs work.
It must not send the recognized prefix or suffix as a sentence. This is a
product decision, not merely an error-handling detail: sentence generation can
invent intent when a word is missing.

Provider failures use stable HTTP 502 messages. Model-load/runtime failures use
HTTP 503. Validation and body-limit failures use controlled 400/413/422
responses. Internal paths, credentials, raw frames, and provider stack traces
must not cross the public boundary.

---

## Page 12 — Crossing into the backend

The backend accepts either legacy/manual keywords or the reviewed capture
result envelope:

```json
{
  "capture_id": "capture-2026-01",
  "complete": true,
  "integration_ready": false,
  "results": [
    {
      "segment_index": 0,
      "status": "recognized",
      "keyword": "pay",
      "confidence": 0.91,
      "coverage": 0.96,
      "top_k": [{"label": "pay", "confidence": 0.91}]
    }
  ]
}
```

The frontend adapter may flatten ML `quality.coordinate_coverage` to
`coverage`; the backend also understands the full quality field. The backend
sorts by `segment_index` and applies its threshold to both confidence and
coverage. Duplicate candidates are not silently allowed to turn an incomplete
batch into a phrase. If the full batch is not safe, `/process` returns:

```json
{"status":"record_again","record_again":true}
```

No Gemini call, TTS call, or audio file should be produced for that result.

---

## Page 13 — Gemini sentence and Hindi output

Gemini receives validated ordered keywords as data, not instructions. The
backend asks for a strict bilingual decision:

```text
can_complete_sentence: boolean
hindi_keywords: bounded non-empty list
english_sentence: bounded Latin-text sentence
hindi_sentence: bounded Devanagari sentence
```

The provider may explicitly return `can_complete_sentence: false`. The backend
maps that decision to `record_again` and skips TTS. Responses with extra keys,
markdown wrappers, duplicate JSON keys, missing language text, oversized
strings, or invalid types are rejected.

Local mock mode is deterministic and covers representative phrases such as
`pay medicine ?`, `hello`, `need water`, `doctor tomorrow ?`, and
`emergency hospital`. These fixtures demonstrate serialization and error
boundaries. They do not demonstrate that a model understands a real signer.

---

## Page 14 — Locale propagation and bilingual playback

The backend owns language selection. English and Hindi TTS settings are
configured independently, for example:

```text
TTS_EN_LANGUAGE_CODE=en-IN
TTS_HI_LANGUAGE_CODE=hi-IN
TTS_EN_VOICE_NAME=...
TTS_HI_VOICE_NAME=...
TTS_VOICE_GENDER=NEUTRAL
```

The successful response labels every output explicitly:

```json
{
  "subtitles": [
    {"language":"en-IN","text":"Can I pay for this medicine?"},
    {"language":"hi-IN","text":"क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"}
  ],
  "audio_segments": [
    {"language":"en-IN","audio_url":"https://backend/audio/a.mp3"},
    {"language":"hi-IN","audio_url":"https://backend/audio/b.mp3"}
  ],
  "playback_order":["en-IN","hi-IN"],
  "selected_language":"en-IN",
  "selected_sentence":"Can I pay for this medicine?",
  "selected_audio_url":"https://backend/audio/a.mp3"
}
```

The request may include `locale: "en-IN"` or `locale: "hi-IN"`. The backend keeps
both labelled subtitles/audio segments, but places the selected language first
in `playback_order` and returns matching selected metadata. The client plays the
declared order and waits for the current segment’s `ended` event; it does not
guess from array position or replace backend locale data.

---

## Page 15 — Audio URL and storage contract

`POST /process` returns an absolute HTTP(S) `audio_url`, plus labelled
`audio_segments`. `GET /audio/{file_id}.mp3` returns non-empty bytes with
`Content-Type: audio/mpeg`. Unknown IDs return 404.

The local implementation writes two MP3 fixtures to a configured ephemeral
directory. It cleans already-written files when a later write fails. A deployed
multi-instance service needs durable shared object storage or signed URLs; a
local filesystem path is not a production storage contract.

Before paid/provider work, the backend validates `PUBLIC_BASE_URL`. Deployed
mode requires an explicit HTTPS base URL. Credentials, query strings, fragments,
control characters, and malformed hostnames are rejected. Local development can
use `http://127.0.0.1:8081` with HTTPS enforcement explicitly disabled.

Audio URLs are data returned by a service and are validated by the frontend as
absolute HTTP(S) URLs without credentials. Missing or relative URLs are not
silently repaired.

---

## Page 16 — Transcription and session boundaries

`POST /transcribe` accepts base64-encoded WAV bytes. The decoded audio must be
mono, 16,000 Hz, 16-bit PCM, within the bounded duration and request-size
limits. Invalid base64 and invalid WAV data are controlled client errors;
provider failures become 502. Silence may return `{"text":""}` and is not
automatically saved as a conversation message.

The backend does not write a session message inside `/transcribe`. The client
reviews or edits the returned transcript, then explicitly posts:

```json
{"sender":"speaker","text":"Yes, that is fine."}
```

Rooms are created by `POST /session/create`. Messages are read oldest first and
timestamps are normalized to UTC ISO-8601 strings ending in `Z`. Only `signer`
and `speaker` senders are accepted; blank, oversized, and unknown-room writes
are rejected without changing existing history.

---

## Page 17 — Configuration that is safe by default

### Frontend public variables

```text
VITE_DEMO_MODE=true|false
VITE_BACKEND_BASE_URL=https://backend.example
VITE_ML_BASE_URL=https://ml.example
VITE_SITE_URL=https://app.example
```

Demo mode must be explicit. Live mode requires both service URLs. Browser-bundled
values contain no secrets. Production endpoints use HTTPS; local HTTP is only
accepted for localhost-style development addresses.

### ML variables

`SIGNBRIDGE_ISL_MODEL_PATH`, label/manifest paths, provisional quality
thresholds, CORS origins, and `PORT` select the serving runtime. Health is a
cheap liveness check; readiness loads/verifies the Keras artifact and reports
the separate integration gate.

### Backend variables

`MOCK_EXTERNAL_SERVICES=1` selects deterministic local provider doubles.
`PUBLIC_BASE_URL`, `REQUIRE_HTTPS_PUBLIC_BASE_URL`, provider language codes,
timeouts, CORS origins, and audio directory are deployment settings. Invalid
timeouts and flags fail closed before provider initialization.

---

## Page 18 — Local setup

Use Python 3.13 and the root virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Run the frontend separately:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

For a local live service pair, use two terminals from the repository root:

```powershell
# ML, port 8080
.venv\Scripts\python.exe -m uvicorn service.app:app --app-dir ml-service --host 127.0.0.1 --port 8080

# Backend, port 8081
$env:MOCK_EXTERNAL_SERVICES = "1"
$env:PUBLIC_BASE_URL = "http://127.0.0.1:8081"
$env:REQUIRE_HTTPS_PUBLIC_BASE_URL = "0"
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8081
```

Set frontend URLs to those loopback endpoints. A physical phone needs a
LAN-reachable address and appropriate cleartext/HTTPS and firewall settings.

---

## Page 19 — The test pyramid

The repository has three useful layers:

1. **Unit tests** cover preprocessing formulas, quality policy, model loading,
   provider configuration, WAV validation, URL validation, and Firestore
   serialization.
2. **Service HTTP tests** use FastAPI `TestClient` to check status codes, body
   limits, health/readiness, batch ordering, provider error mapping, audio
   retrieval, transcription, and sessions.
3. **Root integration tests** connect both FastAPI apps in one process with
   deterministic doubles. They prove that one complete capture can cross the
   ML, backend, TTS, STT, and session boundaries, and that an uncertain batch
   stops at the sentence boundary.

Optional real-artifact or real-video checks are marked separately. They prove
execution with the available artifact/fixture, not recognition quality. Live
Gemini, TTS, STT, Firestore emulator, camera, microphone, browser, Docker, and
Cloud Run checks require explicit environment or hardware setup.

---

## Page 20 — What the root integration checks prove

`tests/test_agreed_flow.py` uses an ordered model double and exercises the
agreed path:

```text
out-of-order batch
  → sorted [pay, medicine, ?]
  → one prepared model call
  → backend process
  → en-GB / hi-IN labelled audio
  → signer message
  → WAV transcribe
  → speaker message
  → oldest-first room read
```

The same file checks that two uncertain slots produce `complete: false`, empty
ML keywords, backend `record_again`, and zero audio files. It also checks that
duplicate indices and aggregate frame overflow are 422 responses while health
stays available without model/provider initialization.

The adapter helper in this test is intentional documentation of the boundary:
the ML service reports quality under `quality.coordinate_coverage`, while the
backend DTO can receive a flattened `coverage`. It keeps the test focused on
the public contract instead of requiring either service to know the other’s
internal Python types.

---

## Page 21 — Health, readiness, and failure behavior

`GET /health` means process liveness. It should be cheap and return
`{"status":"ok"}` without loading a model, contacting Gemini, opening a
Firestore connection, or accessing a camera/microphone. A successful health
response does not mean a model is ready.

`GET /v2/readiness` is the ML model-load check. It may load and hash-check the
selected Keras artifact, then report `model_ready` and the separate
`integration_ready` evidence gate. Readiness failure is 503 and does not expose
filesystem or provider details.

Common public responses:

| Code | Boundary meaning |
|---:|---|
| 200 | accepted operation or explicit `record_again` decision |
| 400 | malformed encoding/request framing |
| 404 | unknown room/audio ID |
| 413 | body/audio/batch limit exceeded |
| 422 | semantic input validation failed |
| 502 | downstream provider operation failed |
| 503 | model/provider setup unavailable |

Every request has a bounded timeout. Automatic provider retries are disabled at
these adapters so the client can make an informed, explicit retry decision.

---

## Page 22 — Deployment shape

Deploy the frontend as a static Vite build and the ML and backend services as
separate HTTP services. Each service binds `0.0.0.0` to the platform `PORT`.
Configure exact frontend origins in backend and ML CORS settings. The backend’s
public base URL must be the HTTPS URL reachable by the browser/device, not a
container-local path.

Cloud deployment requires:

- a reviewed Keras model/label/manifest bundle;
- provider APIs and service-account IAM;
- secrets supplied by Secret Manager or runtime configuration;
- durable audio storage;
- authentication/rate limits and room authorization;
- structured privacy-safe logs and retention policy;
- a real frontend origin in CORS settings.

The checked-in Dockerfiles and READMEs document build commands, but a local
test pass does not prove that Docker, Cloud Run, DNS, IAM, secret mounts, or
browser reachability are configured correctly.

---

## Page 23 — Known limitations and honest claims

The current runtime remains evidence-pending. Specific limitations are:

- official compact-68 runtime reevaluation is pending;
- model confidence and margin thresholds are provisional, not calibrated
  unknown-sign probabilities;
- real signer/device/domain-shift evidence is limited and multi-sign exact
  accuracy is not established;
- static signs are difficult for motion-based segmentation;
- face asset availability can skip MediaPipe integration checks;
- local mock providers do not measure Gemini translation or Google speech
  quality;
- local MP3 storage is ephemeral and unsafe across scaled instances;
- authentication, authorization, retention, and rate limits are deployment
  work;
- camera/microphone permission, audible playback, browser autoplay, and device
  networking require physical acceptance checks;
- the legacy `/predict` endpoint remains for compatibility and is not the
  agreed multi-segment flow.

The honest claim from the normal suite is: **service contracts, boundaries,
validation, and deterministic local plumbing execute as designed.** It is not:
“the model recognizes signs accurately in production.”

---

## Page 24 — Release checklist

Before calling a build demo- or release-ready, verify:

### Contracts and code

- [ ] The frontend sends the full capture to `/v2/predict/batch` after Stop.
- [ ] Results are sorted by index and preserve repeated labels.
- [ ] Any non-recognized slot causes a whole-recording repeat.
- [ ] `complete` and `integration_ready` are not conflated.
- [ ] Backend locale labels agree with subtitle/audio content and playback order.
- [ ] Transcription never writes a message automatically.
- [ ] Session reads are UTC and oldest first.

### Automated checks

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_agreed_flow.py
.venv\Scripts\python.exe -m pytest -q
python -m compileall -q backend ml-service tests
```

### Evidence still required

- [ ] Official model/runtime parity and calibrated thresholds.
- [ ] Labeled signer/device/multi-sign evaluation.
- [ ] Physical camera, microphone, playback, and permission flows.
- [ ] Live provider and Firestore smoke checks with privacy-safe logs.
- [ ] Docker/Cloud Run deployment and HTTPS/CORS verification.
- [ ] Durable audio storage, authentication, retention, and rate limits.

When a check fails, preserve the capture/review state, report the stable public
error, and fix the narrowest boundary first. That discipline keeps a helpful
demo from becoming an unsupported production claim.
