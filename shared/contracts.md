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

### `POST /predict` — `ml-service`

Request:

```json
{
  "landmarks": [
    [
      [0.10, 0.20, 0.00],
      [0.11, 0.21, -0.01]
    ]
  ]
}
```

The abbreviated example above shows two points only for readability. A valid
request must follow these rules:

- `landmarks` is an array of 1–60 frames.
- Every frame contains exactly 21 points.
- Every point contains numeric `[x, y, z]` values.
- MediaPipe-normalized `x` and `y` values should be in the range 0..1.
- `z` is preserved from the hand-landmark detector.
- A frame with no detected hand is represented by 21 points of
  `[0.0, 0.0, 0.0]`.
- The mobile client must not send video bytes or a video URL to this endpoint.

Canonical request shape:

```json
{
  "landmarks": [
    [[0.10,0.20,0.00], [0.11,0.21,-0.01], "... 19 more points ..."],
    "... more frames up to 60 ..."
  ]
}
```

Successful response, HTTP 200:

```json
{"keywords":["pay","medicine","?"]}
```

The keyword list is ordered and intended for the backend sentence generator.
The mobile app may show it as an interim processing state but should not treat
it as the final natural-language sentence.

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
2. Capture a known sign, produce `(frames, 21, 3)` landmarks, and call
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
