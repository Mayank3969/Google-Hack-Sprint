# Person B — actual verification record

**INTEGRATION_WARNING:** these checks establish code behavior and real-artifact
execution. They do not establish live ISL accuracy, Person A feature parity,
calibrated confidence, or a verified Docker/Cloud Run deployment. Follow
`PERSON_B_HANDOFF.md` before clearing the API's pending-integration flags.

## Final results

| Check | Actual result |
|---|---|
| Full ML suite, pinned serving runtime | **145 passed, 1 skipped** |
| Existing root vertical-slice + two-process legacy HTTP regression | **3 passed** |
| Both committed `.keras` artifacts | Loaded and ran real weights; correct `(N,60,660) → (N,50)` shape; batched/individual outputs agree |
| Real Uvicorn process over loopback | Health, model readiness, mixed-batch inference and empty-input rejection passed |
| Real downloaded video pipeline | MediaPipe extraction: 3/3 videos; single-video top-1: **1/3**; two-video ordered combinations: **0/6 exact**; report saved locally under ignored `model/isl/video_evidence/` |
| Release-evidence CLI | **Blocked, exit 2**, correctly lists five missing Person A evidence files |
| Docker build | **Not run**: Docker command unavailable on this host |
| Live-camera / annotated ISL recognition | **Not run**: data/frontend capture not available |
| New checkpoint accuracy comparison | **Not run**: no independent prepared validation/test tensors |
| Git whitespace check | `git diff --check` passed; Git printed line-ending conversion notices |

The one normal-suite skip is the opt-in real-video test; the dedicated real-video
run was executed separately and passed. No release checks were skipped or
replaced with fake successes.

## Environment and exact commands

Host: Windows, Python 3.13.5. Initial checks used the existing local TensorFlow
2.21.0 / Keras 3.15.1 runtime. Final checks used a temporary virtual environment
with system-site-package access and these explicitly installed serving pins:

- `tensorflow-cpu==2.21.0`
- `keras==3.13.2`
- `numpy==2.3.5`
- `fastapi==0.115.14` (`starlette==0.46.2` resolved)
- `uvicorn==0.35.0`

The verification environment inherited pytest 9.1.1 and httpx 0.28.1. The new
developer requirements retain the repository's pytest 8.4.2 pin; a clean Linux
container remains a separate packaging check. An inherited Gradio dependency
conflict was reported in the temporary environment; Gradio was not used and
the workspace/global Python environment was not modified by that install.

Full ML check, working directory `D:\omnirush\hack-sprint\ml-service`:

```powershell
& "C:\Users\Mayank\AppData\Local\Temp\omnirush\isl-b-runtime\Scripts\python.exe" -m pytest tests -q -rs
```

Existing regression check, working directory repository root:

```powershell
python -m pytest tests/test_vertical_slice.py tests/test_http_integration.py -q
```

That existing regression uses mocked cloud providers and a synthetic legacy
PyTorch checkpoint. It verifies compatibility, not cloud availability or ISL.

Real-artifact smoke and pending release check, working directory `ml-service`:

```powershell
python -m scripts.evaluate_isl_models --smoke
& "C:\Users\Mayank\AppData\Local\Temp\omnirush\isl-b-runtime\Scripts\python.exe" -m scripts.check_isl_release
```

For another machine, create/activate a virtual environment, install
`requirements-isl-dev.txt`, and use `python` instead of the local absolute path.
Use the existing full developer requirements for legacy/extraction tests too.

## Incremental checks performed during implementation

Each group was run after its implementation/change, then included in the final
suite. Counts below are historical checkpoints; later regression additions
increase the final count.

| Increment | Focused evidence |
|---|---|
| B1 model loading | `test_isl_model.py`: initially 8 passed; both real artifacts tested |
| B3 preprocessing | Preprocessing + model checks: 25 passed; manual expected resampling/difference values, null/mask cases |
| B4–B6 orchestration | Inference policy tests: 11 passed after correcting a test's missing-value encoding |
| B5/B8 endpoints | Keras batch/inference + legacy HTTP/inference: 45 passed |
| B9 reference segmenter | 13 passed after calibration, fixed-rate and terminal-loss regressions |
| B10 replay | Replay + segmentation + inference: 24 passed |
| B7/B12 evidence tools | Evidence + replay: 6 passed; missing release evidence explicitly fails closed |
| B11 actual runtime | Pinned-runtime real-process + model + batch checks: 22 passed |
| Final input-type hardening | Preprocessing + batch: 31 passed; mixed boolean/string JSON rejected; JSON null accepted |

The initial root-directory ML test invocation failed module resolution; tests
were rerun from `ml-service`. `pytest.ini` now includes the ML root in pythonpath.
A temporary dependency install timed out after downloading TensorFlow; its
partial state was inspected, installation completed, and the pinned-runtime
suite passed afterward.

## Regressions fixed while building

- `top_k=1` could otherwise bypass top-1/top-2 margin rejection.
- Face-only/no-hand data could otherwise pass a global coverage threshold.
- Perfectly static v2 tensors could otherwise force a known-word guess.
- Mixed JSON booleans could silently coerce to floats in NumPy.
- Missing detector blocks could otherwise be counted as measured zeros before
  normalization; the serving adapter now preserves missingness semantics.
- Sparse fingertip motion could be lost by a percentile-only motion score.
- Calibration could discard early signing or count non-consecutive onset frames.
- Overflow/tracking recovery could silently create additional words or disappear.
- A partial recognized keyword list could otherwise be sent downstream while
  an uncertain slot remained unresolved.

## Evidence still required

`models/person_a_evidence/` must receive independent `golden.npz`,
`provenance.json`, `selection.json`, `calibration.json`, and
`live_validation.json`. No placeholder versions are supplied. The release
checker exits 2 until those exist and match. All v2 responses currently keep
`integration_ready: false`, with explicit warning codes.
