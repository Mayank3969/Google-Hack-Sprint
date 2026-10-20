# SignBridge

**Two-way communication between sign-language users and the hearing public. Offline. Landmark-only. No video stored.**

> We don't fix the person. We fix the conversation.

![Status](https://img.shields.io/badge/status-prototype-orange)
![Model](https://img.shields.io/badge/model-1D%20CNN%20%2B%20Transformer-blue)
![Accuracy](https://img.shields.io/badge/test%20accuracy-91.67%25-brightgreen)
![Signs](https://img.shields.io/badge/ISL%20signs-50-informational)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What SignBridge Does](#what-signbridge-does)
3. [How It Works](#how-it-works)
4. [Model Architecture](#model-architecture)
5. [Why This Architecture](#why-this-architecture)
6. [Results](#results)
7. [Handling Wrong Predictions](#handling-wrong-predictions)
8. [Privacy and Offline Design](#privacy-and-offline-design)
9. [Deployment Plans](#deployment-plans)
10. [Tech Stack](#tech-stack)
11. [Project Structure](#project-structure)
12. [Getting Started](#getting-started)
13. [Limitations](#limitations)
14. [Roadmap](#roadmap)
15. [Credits](#credits)

---

## The Problem

India has roughly:

| Group | Approx. population |
|---|---|
| People with hearing impairment | ~50 lakh |
| People with speech impairment | ~20 lakh |
| Certified sign language interpreters | **339** |

That is about one interpreter for every 15,000+ people who may need one. Interpreters are concentrated in a few cities. A farmer at a block office, a patient in a district hospital, or a customer at a shop usually has no interpreter, and often cannot read or write either. Typing on a phone is not an answer for them.

The result: people with hearing or speech impairments cannot access basic public services without a family member translating.

> Figures are approximate and drawn from national disability statistics and interpreter registries. Verify against the latest official source before citing.

---

## What SignBridge Does

SignBridge closes the conversation loop in **both directions**.

**Direction 1: Sign to Speech**
A signer performs an Indian Sign Language (ISL) word in front of a camera. SignBridge recognises it and speaks it aloud in the chosen language.

**Direction 2: Speech to Picture and Text**
The hearing person (an officer, doctor, shopkeeper) replies by voice. SignBridge converts that to text and matching pictures the signer can understand without reading.

Most sign-language projects stop at one direction. A conversation needs both.

---

## How It Works

```
                      SIGNER SIDE
 ┌────────┐   ┌────────────┐   ┌───────────────┐   ┌─────────────────┐
 │ Camera │──▶│ MediaPipe  │──▶│ Preprocessing │──▶│ 1D CNN +        │
 │        │   │ Landmarks  │   │ (normalise,   │   │ Transformer     │
 └────────┘   └────────────┘   │  motion feats)│   │ classifier      │
                               └───────────────┘   └────────┬────────┘
                                                            │ Top-3 predictions
                                                            ▼
                                            ┌───────────────────────────┐
                                            │ Signer confirms / taps    │
                                            │ 2nd or 3rd option         │
                                            └────────────┬──────────────┘
                                                         ▼
                                            ┌───────────────────────────┐
                                            │ Word → Text → Speech      │
                                            │ (Hindi / English / other) │
                                            └───────────────────────────┘

                    HEARING PERSON SIDE
 ┌────────────┐   ┌──────────────┐   ┌──────────────────────────────┐
 │ Voice      │──▶│ Speech-to-   │──▶│ Text + locally stored        │
 │ (officer)  │   │ Text         │   │ pictures, editable by region │
 └────────────┘   └──────────────┘   └──────────────────────────────┘
```

### Step by step

1. **Capture.** The camera streams frames. Nothing is recorded or uploaded.
2. **Landmark extraction.** MediaPipe converts each frame into 2D landmark coordinates for hands, eyes, eyebrows, nose and lips. Raw pixels are discarded immediately. *Why:* landmarks are tiny, lighting-invariant, background-invariant and carry no identity information.
3. **Windowing.** A 60-frame window holds one sign. Shorter signs are masked, not zero-padded blindly (see [Masking](#masking)).
4. **Preprocessing.** Coordinates are normalised and motion features are added (see [Preprocessing](#preprocessing)).
5. **Classification.** The 1D CNN + Transformer outputs probabilities over 50 ISL words.
6. **Confirmation.** The top guess is auto-selected. The user can tap the 2nd or 3rd guess to correct it.
7. **Speech output.** The confirmed word is spoken. One sign can map to Hindi, English or any other configured language.
8. **Reverse loop.** The hearing person's speech is transcribed and shown as text plus pictures stored locally.

---

## Model Architecture

The classifier is adapted from the 1st-place solution of the Kaggle *Google: Isolated Sign Language Recognition* competition (ASL, 250 signs) by [hoyso48](https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with). We retrained it on Indian Sign Language.

### Preprocessing

| Step | What | Why |
|---|---|---|
| Landmark selection | Hands, eyes, eyebrows, nose, lips | Sign meaning lives in hands plus face. Dropping body/other face points cuts noise and model size. |
| Anchor normalisation | Subtract nose landmark position, divide by landmark standard deviation | Makes the model independent of where the signer stands and how far they are from the camera. |
| Motion features | Frame difference at lag 1 and lag 2 (`x[t+1]-x[t]`, `x[t+2]-x[t]`) | Signs are movements, not poses. Explicit velocity helps the model. Lags beyond 2 gave no extra gain in the reference work. |
| NaN handling | Missing landmarks set to 0 | Hands leave the frame often. The model must tolerate that. |
| Final features | `[position, lag-1 motion, lag-2 motion]` concatenated per frame | One vector per frame, fed to the network. |

### Network

```
Input  (T frames × F features)
   │
Masking  ← ignores padded frames
   │
Dense (stem) → BatchNorm
   │
┌──────────────────────────────┐
│ Conv1D Block ×3              │  depthwise Conv1D, causal padding,
│ Transformer Block ×1         │  BatchNorm, Swish, drop-path
└──────────────────────────────┘
   │
┌──────────────────────────────┐
│ Conv1D Block ×3              │
│ Transformer Block ×1         │
└──────────────────────────────┘
   │
Dense (expand ×2)
   │
Global Average Pooling (mask-aware)
   │
Late Dropout (high rate)
   │
Dense → 50-class logits
```

### Key components

**Conv1D Block**
Depthwise 1D convolution with a large kernel, causal padding, BatchNorm and a residual connection with drop-path. It captures *local* temporal patterns: the short motion that forms part of a sign.

**Transformer Block**
Multi-head self-attention plus a feed-forward layer. It captures *global* relationships across the whole sign, such as how the start of a gesture relates to its end.

**Masking**
Signs have different lengths. Padded frames are masked so they never influence the output. Causal padding keeps frame positions aligned, so one mask works through every convolution.

**Late Dropout and Global Average Pooling**
Dropout is applied only after pooling and only after the model has begun learning. This avoids hurting early training while still regularising the classifier head.

### Regularisation

The model trains from scratch on a small dataset, so overfitting is the main risk.

| Technique | Role |
|---|---|
| Drop-path (stochastic depth) | Randomly skips residual blocks during training |
| High dropout after pooling | Forces redundancy in learned features |
| Adversarial Weight Perturbation | Pushes the model toward flatter, more robust minima |
| Temporal augmentation | Random resampling and masking |
| Spatial augmentation | Horizontal flip, random affine (scale, shift, rotate, shear), random cutout |

### Training recipe

| Item | Setting |
|---|---|
| Optimiser | RAdam with Lookahead |
| Schedule | Cosine decay |
| Loss | Categorical cross-entropy (optional label smoothing) |
| Seeds | Multiple seeds can be ensembled |

---

## Why This Architecture

**Why landmarks instead of raw video?**
Raw video models need huge datasets and GPUs. Landmarks give a compact, privacy-safe representation that runs on a phone CPU. The model file is **5.16 MB**.

**Why a 1D CNN before a Transformer?**
Consecutive frames of a sign are strongly correlated. A 1D CNN exploits that cheaply and learns local motion patterns. It acts as a trainable tokeniser, turning raw frames into compact motion tokens that the Transformer can reason over. In the reference competition, a pure 1D CNN already beat a pure Transformer, and the hybrid beat both.

**Why not a larger model?**
Latency and size. Measured inference is **63 ms** per sign window, which is fast enough for real-time use on modest hardware. A bigger model would add little on 50 classes and hurt offline deployment.

**Why mask instead of pad with zeros?**
Training with padding but inferring without it causes a train-test mismatch and drops accuracy. Masking keeps both consistent and lets short signs run faster at inference.

**When does this design struggle?**
Signs that differ mainly in fine finger shape or facial expression, and signs performed with a hand outside the camera frame. See [Limitations](#limitations).

---

## Results

Evaluated on a held-out test set. Model: `ISL_1DCNN_Transformer_v2`.

| Metric | Value |
|---|---|
| Test accuracy | **91.67%** |
| Test loss | 0.3045 |
| Macro F1 | 0.8890 |
| Weighted F1 | 0.9213 |
| Classes | 50 ISL words |
| Test samples | 192 |
| Model size | 5.16 MB |
| Inference latency | ~63 ms |

**Reading these numbers**

- Accuracy of 91.67% means about 9 in 10 signs are recognised correctly.
- Macro F1 (0.889) is lower than weighted F1 (0.921). So a few classes perform noticeably worse than the average. Per-class analysis is a planned next step.
- The test set is small (192 samples across 50 classes, roughly 4 per class), so each class estimate is noisy. Treat these as a strong prototype result, not a field-validated one.

---

## Handling Wrong Predictions

No classifier is perfect, and a wrong word in a hospital matters. SignBridge does not hide the error. It makes correction one tap.

1. The top prediction is auto-selected and spoken only after the signer sees it.
2. The 2nd and 3rd most likely words are shown as alternatives.
3. The signer taps the right one if the top guess is wrong.

At ~92% top-1 accuracy, top-3 coverage is higher, so most errors are fixable in a single tap.

---

## Privacy and Offline Design

- **Only landmarks are processed.** No images or video are stored or transmitted.
- **No user data leaves the device.**
- **Fully offline operation** is the design target. Recognition, word-to-speech and the picture library all run locally.
- **Local, editable picture and language packs.** Each region or office can adjust pictures and languages for its own context.

*Why it matters:* these users are often in sensitive settings (health, government, legal). Sending their face and hands to a cloud server is a barrier to trust and to use in low-connectivity areas.

---

## Deployment Plans

**Plan 1: Kiosks and devices in public offices**
Hospitals, government offices and the public sector. Aimed at users who cannot read or write, where an interpreter is rarely available.

**Plan 2: Offline mobile-first app**
For everyday street use: shops, transport, college and workplaces.

**Beyond India**
The pipeline is language-agnostic. Retrained on another sign language dataset, the same system can serve other countries.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Landmark extraction | MediaPipe |
| Sign model | TensorFlow / Keras, 1D CNN + Transformer |
| Training | Kaggle / Colab GPUs |
| Dataset | INCLUDE (ISL), 50-word landmark subset |
| Grammar cleanup (optional, online) | Gemini API |
| Text to speech | Google Cloud TTS |
| Speech to text | Google Cloud Speech-to-Text |
| Backend and deployment | Google Cloud Run, continuous deployment from GitHub |
| Prototype UI | Streamlit |
| Target client | Native mobile app |

---

## Project Structure

```
signbridge/
├── data/                 # landmark datasets (not committed)
├── notebooks/            # training and evaluation notebooks
├── src/
│   ├── preprocess.py     # normalisation + motion features
│   ├── model.py          # 1D CNN + Transformer
│   ├── train.py          # training loop, AWP, augmentation
│   ├── infer.py          # real-time inference
│   └── speech/           # TTS and STT wrappers
├── app/
│   └── streamlit_app.py  # demo prototype
├── models/               # trained weights
├── assets/               # pictures, language packs
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/<your-username>/signbridge.git
cd signbridge
pip install -r requirements.txt

# Run the demo
streamlit run app/streamlit_app.py
```

**Train from scratch**

```bash
python src/train.py --config configs/isl50.yaml
```

---

## Limitations

- **50 words only.** This is a proof of concept. Sentence-level and open-vocabulary signing are not supported.
- **Isolated signs.** Each sign is classified separately. Continuous signing is future work.
- **Small test set.** 192 samples. Real-world accuracy will vary with lighting, camera angle and signer.
- **Hand-out-of-frame cases** reduce accuracy, as with all landmark-based methods.
- **Not a replacement for interpreters.** SignBridge supports basic, high-frequency interactions where no interpreter is available.


---

## Credits

- Dataset: INCLUDE, Indian Sign Language.
- Landmarks: MediaPipe by Google.

**Team:** Mayank Dhapodkar, Shlok Dwivedi, Ujwal Nandanwar, Kirtikumar Mishra
