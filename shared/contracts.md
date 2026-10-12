# SignBridge Shared API Contract

Status: **M0 shared contract**  
Audience: mobile app, `ml-service`, and `backend` developers  
Transport: HTTPS with JSON request/response bodies unless noted otherwise

This contract is the boundary between the mobile app and the two services. The
mobile app must never receive or store Gemini, Speech, Text-to-Speech,
Firestore, or other Google Cloud credentials.

## Service configuration

The mobile app receives two build-time, non-secret values:

```text
ML_BASE_URL=https://<ml-service-host>
BACKEND_BASE_URL=https://<backend-service-host>
```

The client appends endpoint paths itself and must avoid double slashes. All
production URLs must use HTTPS. Every request should have a finite timeout and
display a recoverable error state when the service is unavailable.

## Common conventions

- JSON uses UTF-8.
- Timestamps are UTC ISO-8601 strings, for example
  `2026-01-01T00:00:00Z`.
- Successful responses use HTTP 200 unless the endpoint explicitly creates a
  resource with HTTP 201.
- Validation failures use HTTP 400 or 422 and a JSON body containing `detail`.
- Downstream provider failures use HTTP 502 and must not expose credentials,
  provider stack traces, raw audio, or camera frames.
- Request IDs may be returned or logged for support, but private payloads must
  not be logged.

## 1. Sign landmarks to keywords

### `GET /health` — `ml-service`

Response, HTTP 200:

```json
{"status":"ok"}
```

The health check must not call an external provider or require a camera.

### `POST /predict` — `ml-service` (M1/M2 expanded feature contract)

Request:

```json
{
  "landmarks": [
    [[0.10, 0.20, 0.00], "... 109 more [x,y,z] points ..."]
  ]
}
```

Each frame is exactly 110 points: 21 left-hand points, 21 right-hand points,
then 68 selected face points using `FACE_FEATURE_VERSION="compact-v1"`.
Validate the model returns 478 source points, then concatenate these groups
in the listed order without sorting their indices:

```text
RIGHT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_EYE=[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
RIGHT_EYEBROW=[70,63,105,66,107]
LEFT_EYEBROW=[336,296,334,293,300]
LIPS_OUTER=[61,146,91,181,84,17,314,405,321,375]
LIPS_INNER=[78,95,88,178,87,14,317,402,318,324]
NOSE=[1,2,4,5,6,168]
```

A valid request must follow:

- `landmarks` is an array of 1–60 frames.
- Every frame contains exactly 110 points in that order.
- Every point contains numeric `[x, y, z]` values.
- MediaPipe-normalized `x` and `y` values should be in the range 0..1.
- `z` is preserved from the hand-landmark detector.
- Missing hands and missing face are represented by zero-filled blocks (21 and
  68 points respectively).
- The mobile client must not send video bytes or a video URL to this endpoint.

Canonical request shape:

```json
{
  "landmarks": [
    [[0.10,0.20,0.00], "... 109 more points ..."],
    "... more frames up to 60 ..."
  ]
}
```

Successful response, HTTP 200:

```json
{"keywords":["pay"]}
```

The current M4 model is an isolated-word classifier, so the list contains one
predicted word. The response remains a list for compatibility with the
backend sentence-generation contract and future sequence models. The mobile
app may show the word as an interim processing state but should not treat it as
the final natural-language sentence.

Invalid examples that must be rejected without crashing the service:

- missing `landmarks`;
- zero frames or more than 60 frames;
- a frame containing 20 or 22 points;
- a point containing two or four coordinates; and
- non-numeric coordinate values.

## 2. Keywords to sentence and audio

### `POST /process` — `backend`

Request:

```json
{"keywords":["pay","medicine","?"]}
```

The backend also accepts the segmented ML capture result directly. This is the
preferred handoff once the ML model returns confidence and coverage metadata:

```json
{
  "capture_id": "capture-123",
  "results": [
    {
      "segment_index": 0,
      "status": "recognized",
      "keyword": "pay",
      "confidence": 0.91,
      "coverage": 0.96,
      "top_k": [
        {"label": "pay", "confidence": 0.91},
        {"label": "medicine", "confidence": 0.82}
      ]
    },
    {
      "segment_index": 1,
      "status": "uncertain",
      "keyword": null,
      "confidence": 0.42,
      "coverage": 0.61
    }
  ]
}
```

`results` accepts at most 10 entries. A result may contain multiple `top_k`
 candidates, not only its highest-scoring label. The backend applies the
inclusive `SIGNBRIDGE_RECOGNITION_THRESHOLD` (default `0.50`) to the result's
`confidence`, result `coverage`, and each candidate's `confidence`. Candidates
below the threshold are ignored. Candidates at or above the threshold are
taken in `segment_index` then `top_k` order, and duplicate labels are removed
case-insensitively while keeping the first occurrence. The final list is still
bounded by the Gemini keyword limit. If `top_k` is absent, the result's
`keyword` is treated as its single candidate. `uncertain` entries are ignored.
Segment indices must be unique, confidence and coverage must be between 0 and 1,
and at least one recognized keyword or candidate is required. The legacy
`keywords` form remains supported during migration, but the two forms must not
be mixed.

Rules:

- `keywords` must be a non-empty array of strings.
- The backend limits the number and length of keywords to a documented safe
  maximum.
- The backend generates one natural sentence and synthesizes MP3 audio.

Successful response, HTTP 200:

```json
{
  "sentence":"Can I pay for this medicine?",
  "audio_url":"https://<backend-service-host>/audio/<id>.mp3"
}
```

`audio_url` must be an absolute HTTPS URL reachable from the mobile device. A
local path such as `/tmp/example.mp3` is not a valid response. The mobile app
may stream or download the URL and should expect `Content-Type: audio/mpeg`.

### BE-M9 bilingual text-generation result

Before BE-M11 wires the bilingual result into the public `/process` response,
the provider-neutral Gemini stage first decides whether the recognized words
are sufficient to form a proper sentence. For accepted input, the validated
Gemini object is:

```json
{
  "can_complete_sentence": true,
  "hindi_keywords": ["भुगतान", "दवा", "?"],
  "english_sentence": "Can I pay for this medicine?",
  "hindi_sentence": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"
}
```

For incomplete or unrecognized input, Gemini must return exactly:

```json
{"can_complete_sentence": false}
```

Rules:

- `hindi_keywords` is a non-empty ordered array of strings, with the same
  maximum item count and per-item bound as the input keywords;
- Hindi keyword text and `hindi_sentence` contain Devanagari letters; standalone
  punctuation such as `?` may be preserved;
- `english_sentence` contains Latin letters and both sentences are non-empty
  and bounded strings;
- `can_complete_sentence` is a strict JSON boolean; when it is `true`, the
  structured result contains exactly the four keys shown above, and when it is
  `false`, it contains only that decision key;
- there is no markdown, explanation, or provider metadata; and
- the Hindi sentence preserves the English sentence's intent, question/statement
  form, negation, quantities, and named entities.

When the decision is `false`, BE-M11 returns HTTP 200 with
`{"status":"record_again","record_again":true}` and does not call TTS or
publish audio. When it is `true`, this is a backend text-generation result, not
the final mobile response. BE-M10 adds labelled English/Hindi speech segments
and BE-M11 adds these values to the `/process` subtitle/audio response while
preserving the existing fields.

### BE-M11 bilingual `/process` response

When BE-M11 is enabled, the existing `sentence` and `audio_url` fields remain
the English sentence and English MP3 URL. The response adds the Hindi text,
labelled subtitles, two independently retrievable audio segments, and their
playback order:

```json
{
  "sentence": "Can I pay for this medicine?",
  "audio_url": "https://backend.example/audio/<english-id>.mp3",
  "hindi_keywords": ["भुगतान", "दवा", "?"],
  "hindi_sentence": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
  "subtitles": [
    {"language": "en-IN", "text": "Can I pay for this medicine?"},
    {"language": "hi-IN", "text": "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?"}
  ],
  "audio_segments": [
    {"language": "en-IN", "audio_url": "https://backend.example/audio/<english-id>.mp3"},
    {"language": "hi-IN", "audio_url": "https://backend.example/audio/<hindi-id>.mp3"}
  ],
  "playback_order": ["en-IN", "hi-IN"]
}
```

The mobile client displays the subtitles and plays the audio segments in the
declared order, waiting for English playback to finish before Hindi starts.
Both segment URLs must return non-empty `audio/mpeg` bytes. If bilingual text,
either TTS segment, or either audio write fails, the backend returns a
controlled error and does not publish a partial response or orphaned audio
file.

If `/process` returns `{"status":"record_again","record_again":true}`, the
mobile client must discard the incomplete attempt, prompt the user to capture
the sign/phrase again, and must not call or play any audio segment.

### `GET /audio/{file_id}.mp3` — `backend`

Successful response, HTTP 200:

- response body: non-empty MP3 bytes;
- `Content-Type: audio/mpeg`.

Unknown or expired audio IDs return HTTP 404. Demo storage may be ephemeral,
but a durable object store should preserve this URL contract before production
use.

## 3. Speech to text

### `POST /transcribe` — `backend`

Request:

```json
{"audio":"<base64-encoded WAV bytes>"}
```

The decoded audio must be mono, 16,000 Hz, LINEAR16 PCM in a valid WAV
container. The mobile app must enforce a maximum recording duration and request
size. The service rejects invalid base64, empty audio, oversized audio, or an
unsupported audio format with a controlled client error.

Successful response, HTTP 200:

```json
{"text":"Yes, that will be 200 rupees."}
```

Silence may return an empty string. The mobile app should display that as a
no-speech result rather than treating it as a successful conversation message.

## 4. Conversation sessions

### `POST /session/create` — `backend`

Request body: none.

Successful response, HTTP 200:

```json
{"room_id":"abc123"}
```

The mobile app retains `room_id` for the active conversation and does not put
credentials or private user data in the room ID.

### `POST /session/{room_id}/message` — `backend`

Request:

```json
{"sender":"signer","text":"Can I pay for this medicine?"}
```

`sender` must be exactly `signer` or `speaker`. `text` must be non-empty and
within the service's documented maximum length.

Successful response, HTTP 200:

```json
{"status":"ok"}
```

Unknown rooms, invalid sender values, blank text, and oversized text return a
controlled 4xx response.

### `GET /session/{room_id}/messages` — `backend`

Successful response, HTTP 200:

```json
{
  "messages":[
    {
      "sender":"signer",
      "text":"Can I pay for this medicine?",
      "timestamp":"2026-01-01T00:00:00Z"
    },
    {
      "sender":"speaker",
      "text":"Yes, that will be 200 rupees.",
      "timestamp":"2026-01-01T00:00:02Z"
    }
  ]
}
```

Messages are returned oldest first. Provider-specific timestamp objects must
be converted to UTC ISO-8601 strings before leaving the backend.

## 5. Mobile vertical-slice acceptance test

The contract is ready for mobile implementation when these checks pass against
the deployed services:

1. Create a session and retain `room_id`.
2. Capture a known sign, produce `(frames, 110, 3)` landmarks, and call
   `ml-service /predict`.
3. Send the returned keywords to `backend /process`.
4. Fetch and play the returned `audio_url` on a physical device.
5. Post the generated sentence as a `signer` message.
6. Record a 16 kHz mono LINEAR16 WAV, call `/transcribe`, and post its text as a
   `speaker` message.
7. Fetch `/session/{room_id}/messages` and confirm both messages appear in
   timestamp order.
8. Repeat the flow with denied permissions, no detected hand, a timeout, an
   invalid payload, and invalid base64. The app must show a recoverable error
   instead of hanging or silently losing the action.

This file defines the shared interface only. Service implementation begins
after this module's contract review and validation.
