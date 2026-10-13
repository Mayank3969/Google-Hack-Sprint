# SignBridge ML service guide

This guide describes the complete ML-side pipeline: video/camera landmarks,
missing-data handling, fixed-rate sampling, pause segmentation, Keras inference,
quality rejection, batch ordering, and the tests/evidence that determine what is
safe to claim.

## 1. End-to-end map

```text
camera/video
  ↓
MediaPipe Hand Landmarker + Face Landmarker
  ↓
one raw frame: (110, 3)
  = hand_0 21 points + hand_1 21 points + compact face 68 points
  ↓
fixed-rate sampling (15 FPS target)
  ↓
motion diagnostics + pause segmenter
  ↓
one raw buffer per candidate sign
  ↓
quality/missing-data adapter
  ↓
frozen preprocessing: (T,110,3) → (60,660)
  ↓
Keras 1D-CNN + Transformer, 50-way softmax
  ↓
ordered {keyword, confidence, top_k, quality, status}
  ↓
frontend keeps only a complete ordered keyword list for backend sentence/TTS
```

The ML service does not generate a sentence, call Gemini, synthesize audio, or
write session history. It returns isolated sign results. The browser/integration
layer decides when to send the final ordered keywords downstream.

## 2. Landmark representation

Each decoded frame is converted to exactly 110 points with XYZ coordinates:

| Block | Points | Flattened values |
|---|---:|---:|
| `hand_0` | 21 | 63 |
| `hand_1` | 21 | 63 |
| compact face | 68 | 204 |
| **Total** | **110** | **330** |

The frozen compact face order is the 68 source indices in
`service/isl_preprocessing.py`, grouped as right eye, left eye, eyebrows, outer
and inner lips, and nose. The extractor writes the blocks in this order. In the
training CSV, `hand_0` and `hand_1` are the two named dataset slots; the serving
extractor uses MediaPipe handedness categories to place anatomical left/right
hands into those slots. A CSS camera mirror must not silently swap the semantic
slots or alter the training coordinate convention.

When a hand or face is not detected, the extractor writes an all-zero block.
When a CSV/NumPy source contains missing values, it uses NaN internally. JSON
clients should use `null` or an explicit boolean `observed_mask`; JSON NaN is not
portable. Individual zero coordinates inside an otherwise observed block are
valid coordinates. An entire zero block is treated as absent when no mask is
provided.

## 3. MediaPipe extraction

`scripts/extract_landmarks_module.py` opens a video, creates Tasks Hand and Face
Landmarkers in VIDEO mode, and supplies strictly increasing millisecond
timestamps. It first uses OpenCV position metadata; if that is missing or
non-increasing, it derives timestamps from the finite positive video FPS. Native
resources are closed with `ExitStack`.

For every frame:

1. RGB conversion is performed.
2. Up to two hands are detected.
3. handedness labels choose slot 0/1; missing labels fall back to detector order
   and should be treated as a data-quality warning.
4. 21 XYZ hand points are copied per slot.
5. One 478-point face result is reduced to the frozen 68 indices.
6. Missing blocks remain zeros.

The extractor does not normalize, resample, pad, or calculate velocity. Those
belong to later stages. The checked-in face task asset is ignored by Git and is
required for real extraction.

## 4. Fixed-rate sampling

The browser should sample landmarks at approximately 15 FPS. A video fixture may
be 25 FPS, so `resample_fixed_rate()` maps target timestamps `0, 1/15, 2/15…`
to nearest decoded source frames, removes duplicate selections, preserves
chronological order, and includes the final source frame.

Why fixed rate matters: movement thresholds and the pause duration are measured
in frames. At 15 FPS, one frame is about 66.7 ms and eight frames are about
533 ms. If the browser sends irregular frame timing, motion scores are not
comparable. Use timestamps/frame scheduling to make the input cadence stable.

The model's 60 frames are different from the capture rate. The complete sign
buffer is later resampled to 60 model positions. A raw multi-sign recording may
contain more than 60 frames; it must be segmented first rather than truncated
as one phrase.

## 5. Motion measurement

Motion is used only for segmentation and diagnostics. It is not appended to the
model input by this module.

For each region (left hand, right hand, face):

1. Find points observed in both consecutive frames.
2. Compute 2D XY displacement with Euclidean distance:

   ```text
   displacement_i = sqrt((x_i[t]-x_i[t-1])² + (y_i[t]-y_i[t-1])²)
   ```

3. Use the 75th percentile, with half the maximum as a fingertip safeguard.
4. Divide by the current region's XY span; the minimum scale is 0.03 to avoid
   huge values from a one-point region.

`motion_components()` returns:

```text
left_hand
right_hand
hand = max(left_hand, right_hand)
face
overall = max(hand, 0.35 × face)
```

Hands dominate because head movement should not start/end a sign. Face motion is
still reported for diagnostics and can contribute at 35% when hands are moving.
If no common observed points exist, that region's motion is zero; the segmenter
separately tracks detector loss so missingness is not incorrectly treated as a
quiet pause.

## 6. Adaptive pause calibration

The segmenter starts with fixed fallback thresholds:

```text
pause threshold: 0.04
active threshold: 0.08
```

During the first 12 samples it collects only valid low-motion hand scores. If at
least six finite scores are available, it calculates:

```text
median = median(scores)
MAD    = median(abs(scores - median))
noise  = median + 3 × MAD
pause  = clamp(max(2.5 × noise, 0.04), 0.04, 0.08)
active = min(max(1.5 × pause, pause + 0.02), 0.12)
```

The separation between active and pause is hysteresis. It prevents a score near
the boundary from rapidly toggling the state. With fewer than six usable quiet
observations, the fixed thresholds remain active. This is a magnitude-adaptive
threshold, not an adaptive pause duration: pause confirmation remains a count of
eight low-motion frames by default (about 0.53 s at 15 FPS).

## 7. Pause segmenter state machine

States:

```text
IDLE → CALIBRATING → WAITING_FOR_SIGN → ACTIVE_SIGN
                                      ↘ PAUSE_CANDIDATE
PAUSE_CANDIDATE → ACTIVE_SIGN (movement resumes)
PAUSE_CANDIDATE → WAITING_FOR_SIGN (eight quiet frames)
ACTIVE_SIGN/PAUSE_CANDIDATE → FINISHED (Stop)
```

Default configuration:

| Setting | Value | Meaning |
|---|---:|---|
| target rate | 15 FPS | fixed-rate input |
| calibration | 12 frames | about 0.8 s |
| onset | 2 active frames | avoids one-frame jitter |
| pre-roll | 3 frames | preserves sign onset |
| pause confirmation | 8 frames | about 0.53 s |
| post-roll | 3 frames | preserves the held end pose |
| minimum candidate | 6 frames | rejects accidental taps |
| maximum candidate | 60 frames | four seconds at 15 FPS |
| max recording | 30 s | 450 raw frames |
| max segments | 10 | bounded batch |
| tracking grace | 4 frames | transient detector loss |

Only hand-valid motion can start a sign. A face-only movement does not create a
candidate. While active, every fixed-rate frame is retained; we do **not** keep
only high-motion frames because static hand configurations can carry the sign.

When a candidate is quiet for eight frames, the last three quiet frames are kept
as post-roll and the remainder is committed. A short pause that resumes before
confirmation returns to `ACTIVE_SIGN`. A segment over 60 frames is marked
`overflow`, not silently split into two words. A terminal tracking-loss tail may
be downgraded to `ready` when Stop is pressed after a usable sign; mid-sign loss
remains `tracking_lost`.

Static signs are intrinsically ambiguous to a motion-boundary detector. The UI
should provide an explicit ready cue or a manual sign commit fallback if the
selected vocabulary contains static signs.

## 8. Frozen model preprocessing

`service/isl_preprocessing.py` is the single source of feature math. It accepts
raw `(T,110,3)` or flattened `(T,330)` values and returns `(60,660)` float32.

For a raw sequence:

1. Validate numeric finite/infinite rules.
2. Select `np.linspace(0, T-1, 60).astype(int)` frame indices. Duplicates are
   intentional for short buffers.
3. For each of the 330 coordinate features, compute `nanmean` over the selected
   frames.
4. Subtract the mean from observed values; map missing values to zero.
5. Compute first differences:

   ```text
   velocity[0] = normalized[0] - normalized[0] = 0
   velocity[t] = normalized[t] - normalized[t-1]
   ```

6. Concatenate all 330 positions then all 330 velocities: `(60,660)`.

This model was trained with velocity features. The browser does not need to
calculate velocity; the server owns this transform. Do not add masks, change to
wrist normalization, interpolate, divide velocity by timestamps, or gate
velocity across missing transitions without retraining/parity evidence.

## 9. Keras model and labels

The selected runtime is Keras 3 with TensorFlow CPU. The serialized custom
layers are:

1. training-only landmark jitter/frame masking;
2. depthwise + pointwise Conv1D tokenizer;
3. two Transformer encoder blocks;
4. global average/max pooling;
5. dense 50-way softmax.

The loader verifies model and label SHA-256 hashes, input `(None,60,660)`, output
`(None,50)`, exactly 50 IDs `0…49`, and normalized finite probabilities. It uses
the real `.keras` weights and caches the loaded model. Two artifacts exist;
official validation comparison must freeze which one is served. The historical
91.67%/88.90 macro-F1 report is provenance, not a fresh file-specific score.

## 10. Quality and abstention

Before inference, quality is calculated from original observation masks:

- `coordinate_coverage`: observed coordinate fraction;
- left/right/face coverage;
- valid frames;
- hand-valid frames;
- `has_signal`.

Default development policy:

```text
minimum hand-valid frames: 3
minimum coordinate coverage: 0.05
minimum top-1 confidence: 0.50
minimum top-1 minus top-2 margin: 0.05
```

These thresholds are not calibrated unknown-sign probabilities. A 50-way
softmax always has a top class, so the service rejects all-missing/face-only/
low-coverage/static-informationless buffers externally. A low confidence or
margin returns `uncertain`, and the UI should request a repeat. It must not send
partial keywords to sentence generation.

Statuses:

```text
recognized, uncertain, insufficient_data, tracking_lost, invalid, overflow, partial
```

## 11. HTTP and batch behavior

`POST /v2/predict` accepts one raw segment. `POST /v2/predict/batch` accepts 1–10
segments with unique indices and at most 450 raw frames total. Valid segments are
stacked into `(N,60,660)` and evaluated in one Keras call. Results are sorted by
`segment_index`; invalid slots remain present. `keywords` is returned only when
every slot is recognized; otherwise it is empty.

This preserves the six possible ordered two-video combinations and repeated
words. The client should use `capture_id` and `segment_id` to discard stale
responses. Body limits and JSON validation happen before expensive inference.

## 12. Missing-data T6 matrix

The diagnostic applies clean, missing-left, missing-right, missing-face, short
and long internal gaps, missing leading/trailing frames, multiple blocks, and all
landmarks absent. It reports status, coverage, confidence, and label stability.
The all-absent condition must be rejected. Clean/missing-modality accuracy on
these three downloaded clips is not a training-set robustness metric; the frozen
model still needs a labeled official validation/test matrix for a retraining
decision.

## 13. Real-video T9 evidence

`scripts/evaluate_capture.py` discovers labels from fixture filenames, extracts
real 110-point MediaPipe sequences, samples at 15 FPS, reports hand/face/overall
motion and adaptive thresholds, runs single-buffer Keras predictions, runs the
missing-data matrix, and creates all six ordered two-video combinations:

```text
Good Morning → Monday
Good Morning → happy
Monday → Good Morning
Monday → happy
happy → Good Morning
happy → Monday
```

The report is diagnostic. In the current run, landmark extraction and all model
calls work, but only 1/3 filename-labelled clips is top-1 correct and 0/6 pairs
are exact ordered sequences. That is evidence of dataset/device/signer domain
shift and segmentation limitations, not a reason to force labels. The output is
ignored by Git under `model/isl/video_evidence/`.

## 14. Testing map

| Layer | Tests |
|---|---|
| unit | preprocessing, hashes, custom-model loading, quality, motion, calibration, segment states |
| integration | FastAPI single/batch, body limits, real Uvicorn, real Keras artifacts |
| derived | fixed-rate 25→15 sampling, missing-data matrix, motion component reports, replay sequence ordering |
| real video | opt-in `SIGNBRIDGE_RUN_REAL_VIDEO=1`; three MP4s through MediaPipe + Keras + six pairs |
| release | official compact68 runtime reevaluation, calibrated thresholds, signer/device evidence |

The normal suite uses real weights but generated inputs where appropriate. It is
not an accuracy claim. The opt-in fixture test and the saved capture report are
the evidence for the actual videos. Run from `ml-service`:

```powershell
python -m pytest tests -q -rs
$env:SIGNBRIDGE_RUN_REAL_VIDEO="1"
python -m pytest tests/test_capture_evaluation.py -m real_video -q -s
python -m scripts.evaluate_capture --video-dir ..\tests\Videos --face-model models\face_landmarker.task --output model\isl\video_evidence\capture_report.json
```

## 15. Honest current status

Completed: frozen Person A preprocessing is integrated, real MediaPipe extraction
works, both Keras artifacts load, per-video and six-pair diagnostics run, missing
data is exercised, motion/calibration are reported, and tests pass.

Pending: official runtime reevaluation on the exact compact68 test split,
checkpoint selection, calibrated thresholds, signer/device robustness, and a
successful real multi-sign accuracy result. Until those pass, keep the API's
`integration_ready: false` and use repeat/fallback behavior rather than claiming
continuous sentence translation.
