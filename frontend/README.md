# SignBridge frontend

Responsive Vite + React + TypeScript implementation of the supplied SignBridge desktop/mobile workspace and settings export. The warm paper, forest, terracotta, editorial typography, two-panel conversation layout, segmented input modes, and trust/settings cards follow the exported `DESIGN.md`; the HTML is implemented as React controls, not embedded. Newsreader and Plus Jakarta Sans are self-hosted build assets.

## Run locally

Requires Node.js 20.19+ (Node 22 LTS recommended) and npm.

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

The example explicitly enables demo mode. Create a room, send a message as either role, and enter comma-separated keywords to try the full local conversation flow. Demo rooms live only in this tab's memory, cannot be joined from another tab/device, and disappear on reload. A demo join requires a room created earlier in the same tab. Demo phrases join keywords verbatim; they are not AI predictions. Browser speech playback is optional and user initiated. Recording works locally; demo transcription is explicitly unavailable.

## Public configuration

| Variable | Meaning |
| --- | --- |
| `VITE_DEMO_MODE` | Required exact `true` or `false`; no implicit fallback. |
| `VITE_BACKEND_BASE_URL` | Required in live mode; absolute public HTTP(S) backend base URL. |
| `VITE_ML_BASE_URL` | Required in live mode; absolute public HTTP(S) updated 110-point ML service URL for health and explicit prediction. |

All `VITE_` values are bundled into the browser and **must not contain secrets**. `.env*` is ignored except `.env.example`. Credentials, query strings, fragments, and non-HTTP(S) endpoint schemes are rejected. There is no localhost fallback, automatic demo fallback, or runtime mode switch. Endpoint paths may have a prefix. Explicit HTTP localhost URLs can be configured for local development; production deployments should configure HTTPS services. Editing variables requires restarting Vite or rebuilding production.

Missing/invalid configuration shows a setup screen instead of sending requests. Live mode errors remain errors and never substitute demo output.

## Service contract

All conversation, language, and transcription requests go to the backend; ML receives its health probe and explicitly submitted landmark captures.

| Request | Payload / expected response |
| --- | --- |
| `GET /health` | Health JSON; any successful HTTP result indicates reachability, not model readiness. |
| `POST /predict` (ML) | `{landmarks: frames}`; 1–60 frames, each 110 numeric `[x,y,z]` points. Returns `{keywords: string[]}` only; no confidence is invented. |
| `POST /session/create` | No request body; `{room_id: string}`. |
| `GET /session/{encoded-room-id}/messages` | `{messages: [{sender: "signer" \| "speaker", text: string, timestamp: string}]}`. |
| `POST /session/{encoded-room-id}/message` | `{sender, text}`; nonempty text, maximum 2,000 characters. |
| `POST /process` | Manual UI sends `{keywords: string[]}`; 1–20 nonempty strings, maximum 64 characters each. Adapter also accepts typed `{capture_id, results}`. Accepts legacy `{sentence, audio_url}`, complete bilingual success (below), or HTTP 200 `{status: "record_again", record_again: true}`. |
| `POST /transcribe` | `{audio: "plain-base64-PCM16-WAV"}`; expects `{text: string}`. |

Contract source: `origin/backend:backend/routes/process.py` and `origin/backend:shared/contracts.md` (BE-M11); the workspace's shared contract may still document the older response. Live success requires a nonempty sentence and absolute HTTP(S) audio URL without credentials. Relative/missing/null audio URLs are rejected, as specified by the public backend contract. Demo alone uses browser voice without a server audio URL. Backend CORS must permit the deployed frontend origin, GET/POST/OPTIONS and Content-Type. Audio hosting must be accessible over HTTPS for an HTTPS app. No authentication scheme is added by the frontend.

Bilingual success adds `hindi_sentence`, nonempty `hindi_keywords`, `subtitles: [{language, text}]`, `audio_segments: [{language, audio_url}]`, and `playback_order`. Both `en-IN` and `hi-IN` must appear exactly once in each labelled list/order; English aliases and subtitle text must agree with the sentence fields. Partial, contradictory, or malformed results are rejected. Both subtitles are displayed with language attributes. The Play audio button follows the declared order (not array position), advancing only on the current segment's `ended` event. Stop/replay never repeats `POST /process`. Either sentence can be copied to an editable signer draft.

`record_again` clears prior audio/subtitles and offers **Recapture signs**, which clears attempted keywords and resets the capture UI without another POST or automatic camera permission. Enable camera again to start a fresh capture. **Re-enter keywords** instead focuses/selects the preserved manual input. Transcription HTTP 400/413/422 errors offer record-again guidance; HTTP 404 room errors direct users to Create/Join; HTTP 502 errors explain provider unavailability. Backend validation bodies (including private `input` fields) are never displayed. Empty transcripts preserve the existing draft. Failures preserve keywords, room input, and draft.

The existing segmented capture-input type remains available at the adapter boundary. The current isolated-word ML model does not return that metadata: camera/video flow deliberately uses the legacy `{keywords}` process payload after user review, without fabricated confidence, coverage, or segmentation.

## Camera/video setup and flow

**The existing local `ml-service/` folder uses the old 21-point layout. Set `VITE_ML_BASE_URL` to your teammate's updated 110-point ML service with its trained classifier loaded.** This frontend does not change or train the ML service. A healthy old endpoint is not a compatible endpoint. Authoritative layout/handedness were checked against `git show origin/backend:shared/contracts.md` and `git show origin/backend:ml-service/scripts/extract_landmarks_module.py`.

MediaPipe Tasks Vision is pinned to **0.10.22-rc.20250304** in package.json and package-lock.json. Runtime assets are served from the frontend origin, not a CDN. Install real official version-1 detection models and copy WASM from that exact installed npm package before enabling vision. These are landmark detectors, not the sign classifier. No placeholder/fake model is included.

Run from `frontend/` after `npm ci` (PowerShell):

```powershell
New-Item -ItemType Directory -Force public/models, public/mediapipe/wasm
Copy-Item node_modules/@mediapipe/tasks-vision/wasm/* public/mediapipe/wasm/ -Force
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile public/models/hand_landmarker.task
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile public/models/face_landmarker.task
npm run build
```

Provision these assets in deployment before `npm run build` too; generated WASM/model assets are ignored by git. Verify `/models/hand_landmarker.task`, `/models/face_landmarker.task` and the copied `/mediapipe/wasm/*` files return binary/JavaScript assets rather than an SPA HTML fallback. The face model must produce **478** source points. Check applicable MediaPipe model licenses when distributing. Setup downloads happen only when the developer runs those commands; the UI does not automatically download remote models.

1. Choose live mode explicitly in `.env.local`; demo hides camera/video controls and performs no model loading or prediction.
2. Press **Enable camera** (HTTPS/localhost required) or choose a supported local video. This loads local detectors and opens an unmirrored preview; it does not capture landmarks or contact cloud services yet.
3. Press **Capture sign**, perform one isolated word, then **Finish capture**, or wait for the 4.5-second wall limit. Sampling is at most 15 fps, capped at 60 frames. Slow devices may produce fewer frames. Local video capture starts at the beginning and stops at its end or these limits.
4. Capture releases tracks, both detectors, video source and local object URL. Press **Predict sign** to send landmarks only; no video bytes or video URL leaves the browser. Review the returned word(s), then **Append reviewed words**. Repeat to collect isolated words; edit/remove words in the comma-separated input (1–20 words, 64 characters each).
5. Press **Generate sentence** to call `/process` with reviewed keywords. Bilingual playback and editable drafts remain user-operated.

Each feature frame is left hand 21, right hand 21, then face compact-v1 68 points in exact contract group order (right eye, left eye, right eyebrow, left eyebrow, outer lips, inner lips, nose). Detector handedness maps directly to slots, with detector-index fallback for absent/unknown labels, matching training. Coordinates are never mirrored, clipped, sorted, centered or normalized again; z is preserved. Missing blocks are zeros. A capture with no hands in any frame is rejected locally. There is no automatic segmentation or continuous-sentence recognition.

Cancel/reset, input-mode changes, settings navigation, hidden tabs, unmount and detection/setup failures release owned resources. Late camera permission/model initialization results are closed rather than attached after cancellation. Pending prediction is aborted/ignored on teardown. A denied permission offers browser settings/local-video recovery; missing models/WASM point back to asset setup; network/CORS errors require endpoint/origin checks; 422 indicates incompatible or rejected landmarks; 503 indicates the ML host must load its trained model. Retries are explicit. The backend and ML server must both allow frontend-origin CORS for POST/OPTIONS and Content-Type.

Timeline sync uses full server snapshots (never appends each poll), retains genuinely repeated messages, permits only one in-flight poll per active loop, pauses when the page is hidden, and cancels/ignores results when the room, route, or pause state changes. It retries failed reads on the next 2-second cycle. Writes are never automatically retried after errors/timeouts because the server may already have accepted them. Inspect the timeline before manually retrying an uncertain write.

## Audio and privacy

- Microphone access is opt-in through Start microphone and needs HTTPS (or browser-trusted localhost), AudioWorklet, and Web Audio support.
- A worklet mixes input channels to mono and caps samples at 10 seconds. A wall-clock timer also stops capture. The input sample rate is honored: windowed-sinc low-pass resampling produces actual 16 kHz audio, encoded with RIFF/WAVE PCM16 headers and sent as base64 without a data-URL prefix. This does not rename WebM as WAV.
- Stop releases tracks and closes the AudioContext before transcription. Cancel, navigation, late permission results, failed setup, and unmount also clean up microphone resources. Transcription results are editable before posting and may be discarded. Cancellation cannot retract a request already received by a service.
- Generated and recorded audio uses user-operated playback. New generation/recording, mode/route changes, room changes, record-again recovery, and unmount discard the generated playlist; listeners are removed, media is paused and its source released. Late generation results are ignored after navigation/mode changes. The last recording is held as an in-memory object URL and released on replacement/unmount. Nothing autoplays before a user starts playback.
- Demo playback uses browser speech synthesis, preferring an installed local voice if available. A browser/OS may use its own online speech service; the frontend does not claim that all browser voices are offline.
- Live messages/keywords/audio leave the browser for the configured backend. Storage, provider usage, retention, and room authorization are backend deployment concerns. Knowing a room ID is not a frontend authentication mechanism.
- Camera/video extraction is opt-in. Only explicit prediction uploads landmarks (including face features) to the configured ML host. No confidence scores, analytics, or external font calls are added.

## Checks

```sh
npm test
npm run typecheck
npm run build
npm run preview
```

Unit/UI coverage (Vitest + React Testing Library/jsdom): legacy and bilingual success, typed capture forwarding, record-again recovery, malformed/unsafe/inconsistent payloads, server-declared Hindi-first playback order, replay without duplicate POST, playlist cleanup/stale results, editable bilingual drafts, error/silence input preservation, safe error guidance, and isolated demo behavior. Existing coverage also checks explicit config and URL rejection, API routes/input limits, serial/visibility-aware polling, actual 48/44.1 kHz conversion, anti-alias attenuation, PCM16 headers/clipping and base64, late microphone permission cleanup, worklet failure cleanup, and the 10-second WAV cap. UI tests simulate media events and microphone capture; hardware permissions, browser autoplay policies, audible playback, visual breakpoints, and deployed services still require browser/device acceptance testing. There is no lint script configured.

Vision tests check exact feature order/handedness/missing blocks, source-face count, finite coordinates/frame bounds, handless blocking, capture caps and resource cleanup, late permission results, partial detector failure, explicit capture → predict → review → process, 503 retry, late prediction cancellation, demo isolation and actionable record-again recovery. These use mocked detectors/media/network, not a physical browser verification. Before a demo, test real camera permission denial/allow, model loading and missing assets, hand/face detection, local codecs, cancellation/navigation and camera-indicator shutdown, 422/503/CORS, a known trained word, review/append and actual bilingual playback against the updated service.

Browser acceptance: at mobile and desktop widths, create/join a real shared room from two tabs in live mode, send from both roles, hide/resume the tab, pause/resume sync, generate a sentence and explicitly play audio, deny/allow the mic, stop and cancel recordings, let the 10-second cap expire, navigate during permission/transcription, and confirm the browser microphone indicator turns off. Confirm returned transcript is editable and only posted after Send.

## Vercel

Import the repository and set **Root Directory: `frontend`**, framework Vite, install command `npm ci`, build command `npm run build`, output directory `dist`. Configure the three public variables for the deployment environment before building. `vercel.json` provides SPA rewrites for `/settings` and other client paths; `/assets` stays static. Vercel serves public worklet and font assets with the build. Redeploy after environment edits. Keep cloud credentials in backend/ML service configuration, never in this app.

This deploys only the frontend. Backend/ML deployment and CORS must be configured separately. A successful static build does not verify real microphone hardware, cloud speech, model availability, or backend authorization.
