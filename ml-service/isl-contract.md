# Keras ISL v2 boundary — Person B

**INTEGRATION_WARNING:** `isl-preprocessing-v2` is now the integrated frozen
Person A contract. `integration_ready` remains false until official runtime
reevaluation, calibrated quality thresholds, and real signer/multi-sign evidence
pass. A successful HTTP response is not a recognition-quality claim.

## Raw data and missingness

- Raw shape `(T,110,3)`; assumed left 21, right 21, compact face 68, XYZ order.
- **Exact face source indices, handedness convention and mirroring are unverified
  against the actual compact training CSV.** Do not infer them from shape alone.
- Finite numbers and JSON `null` are accepted. Numeric strings, booleans and
  infinities are rejected. NaN is supported internally for offline NumPy data;
  JSON clients must send `null`, not nonstandard JSON `NaN`.
- Optional `observed_mask` is a strictly boolean array with the same shape.
  False coordinates become missing before mean-centering. Masks are not appended
  to the model tensor. A mask does not make null/non-finite coordinates valid.
- Without a mask, an entire zero-filled hand or face block is treated as missing.
  Individual zero coordinates inside an observed block are valid observations.
- All-zero or face-only buffers cannot become accepted words. One valid hand
  without face/the other hand can be processed.
- Perfectly static buffers whose temporal variation is erased by v2 centering
  are rejected rather than classified from an informationless tensor. This
  model/pause heuristic cannot reliably support purely static signs.

## Transformation

1. Flatten raw coordinates in their received order: `(T,330)`.
2. Select indices `linspace(0,T-1,60).astype(int)` across the **whole** buffer.
3. Compute each coordinate's temporal mean over observed selected values.
4. Subtract that mean; missing positions become zero.
5. Difference adjacent normalized frames; the first difference is zero.
6. Concatenate all 330 positions, then all 330 differences → `(60,660)` float32.

There is no interpolation, wrist-anchor normalization, timestamp division or
missing-transition velocity suppression in the compatible transform. Adding any
of those requires Person A's new version and validation/retraining decision.

## Single prediction

`POST /v2/predict?top_k=3`

Fields: `segment_index` (strict integer 0–9), optional `segment_id` (≤128 chars),
`landmarks`, optional `observed_mask`. Up to 450 raw frames; `top_k` is 1–10.

Result fields:

```json
{
  "segment_index": 0,
  "segment_id": "capture-a-sign-0",
  "status": "uncertain",
  "keyword": null,
  "class_id": null,
  "confidence": 0.4,
  "top_k": [],
  "quality": {},
  "error": "prediction confidence or margin is too low",
  "model_version": "ISL_1DCNN_Transformer_v2:<artifact-hash-prefix>",
  "preprocessing_version": "isl-preprocessing-v2",
  "integration_ready": false,
  "warnings": ["OFFICIAL_COMPACT68_RUNTIME_REEVALUATION_PENDING"]
}
```

Example abbreviates `quality`, `top_k` and warnings. Actual top-k items contain
`class_id`, `label`, `confidence`. `quality` contains raw/valid/hand-valid frame
counts, coordinate coverage, per-hand/face coverage and `has_signal`.

`recognized` returns a word; `uncertain`, `insufficient_data`, `tracking_lost`,
and `invalid` return null keywords. Model scores are not calibrated probabilities
of correctness. Unknown signs can still produce high softmax scores.

## Batch prediction

`POST /v2/predict/batch?top_k=3`

Envelope: `{capture_id: string, segments: [single-segment fields, ...]}`.

- 1–10 segments; unique indices and non-null IDs; 450 total raw frames.
- Results sorted by `segment_index`. Gaps are permitted for selective repeats.
- Duplicate predicted labels are preserved; repeated signs are legitimate.
- Valid buffers are evaluated in one model call. Invalid/uncertain slots stay
  in their original index positions.
- Response: `{capture_id, complete, keywords, results, integration_ready, warnings}`.
- `complete` means every submitted slot is recognized under the configured
  provisional policy. **It does not mean the system is integration-ready.**
- `keywords` is empty if any slot is not recognized. Clients use the per-slot
  results to request repeats and must not send a partial phrase downstream.
- Both `complete` and reviewed `integration_ready` are needed before automatic
  speech in the integrated app. Current output is for development inspection.
- One JSON response after the model batch; this is not NDJSON/live streaming.
  Progressive server results would need a separately agreed protocol.

## Errors, limits and lifecycle

- Invalid envelope, duplicate IDs/indices or aggregate frame overflow: HTTP 422.
- Malformed individual landmark buffer: HTTP 200 with that slot `invalid`.
- Oversized body: HTTP 413, including chunked uploads; malformed length: 400.
- Model load or runtime failure: HTTP 503 without internal exception details.
- Body limits: v2 8 MiB; legacy 512 KiB. Limits apply before JSON parsing.
- No persistent side effects, model retraining, session writes, or provider calls.
- `/health` is process liveness; `/v2/readiness` distinguishes model load readiness
  from the pending Person A/real-capture evidence.

## Segmentation reference boundary

The Python reference expects fixed-rate 15 FPS arrays. It returns ordered
`SignSegment` objects with inclusive start/end frames and raw frames. Default
limits: 30 seconds, 10 candidates, 60 retained sign frames plus pause lookahead.
`overflow`, `partial`, `tracking_lost` candidates must be repeated, never split
silently into extra words or passed to the model by the replay tool.

Adaptive motion thresholds fall back to fixed thresholds without sufficient
quiet calibration data. Face-only motion, detector disappearance and recovery
are not sign boundaries. Brief pauses cancel if motion resumes. Long internal
holds and purely static signs cannot be disambiguated by this heuristic.
Irregular browser timing needs a fixed-rate adapter during frontend integration.
