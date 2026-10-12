# SignBridge mobile playback handoff

The backend owns Gemini, Google Cloud Text-to-Speech, and audio URL creation.
The mobile client never receives provider credentials.

## `/process` playback contract

The client should read the response fields as follows:

- `sentence` remains the English subtitle for existing clients.
- `hindi_sentence` is the Hindi subtitle.
- `subtitles` contains labelled entries for `en-IN` and `hi-IN`.
- `audio_segments` contains independently retrievable `audio/mpeg` URLs.
- `playback_order` is authoritative and currently must be `["en-IN", "hi-IN"]`.
- `audio_url` is preserved as an alias for the English segment URL.

Play the segment whose language is first in `playback_order`, wait for playback
completion, then play the next segment. Do not merge the two audio payloads and
do not infer order from array position when `playback_order` is present.

## Error and connectivity behavior

Treat HTTP 502 from `/process` as a recoverable synthesis failure. Do not play
or cache a partial result. Each audio URL should be fetched over the configured
backend base URL and should return non-empty `audio/mpeg` bytes.

When the recognized words are not sufficient to form a proper sentence, `/process`
returns HTTP 200:

```json
{"status":"record_again","record_again":true}
```

Treat this as a normal capture retry: discard the incomplete result, prompt the
user to sign/capture again, and do not call TTS or fetch any audio URL. This is
different from a provider/server failure (`502`).

The backend M13 gate verifies this contract in mock mode and can repeat the
HTTP checks against a live backend with:

```powershell
python -m backend.verification_gate --live --base-url https://<backend-url>
```

The gate reports only check names and timings; it never prints credentials,
prompts, or generated text.
