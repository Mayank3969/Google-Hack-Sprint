"""Real loopback Uvicorn/M5 integration with explicitly mocked cloud providers.

Run with the project interpreter: .venv/Scripts/python.exe -m pytest
tests/test_http_integration.py -q. The synthetic checkpoint has controlled
weights solely to verify loading/inference/wiring, never recognition quality.
"""

from __future__ import annotations

import base64
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import importlib
from io import BytesIO
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import wave

import httpx
import numpy as np
import pytest
import torch

from backend.services.stt_client import MOCK_TRANSCRIPT


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_SECONDS = 30
pytestmark = pytest.mark.integration


def _synthetic_checkpoint(path: Path) -> Path:
    trainer = importlib.import_module("scripts.04_train_bigru")
    config = trainer.TrainConfig(hidden_size=4, layers=1, dropout=0.0)
    model = trainer.build_model(2, config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Exercise the real BiGRU path but make its output independent of random
        # initialization and input content. These are NOT trained sign weights.
        model.classifier[-1].bias.copy_(torch.tensor([4.0, -4.0]))
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_size": config.input_size,
                "hidden_size": config.hidden_size,
                "layers": config.layers,
                "dropout": config.dropout,
            },
            "label_names": ["pay", "medicine"],
            "feature_shape": trainer.FEATURE_SHAPE,
            "max_frames": trainer.MAX_FRAMES,
            "preprocessing_version": trainer.PREPROCESSING_VERSION,
        },
        path,
    )
    return path


def _reserve_loopback_port():
    reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    reserved.bind(("127.0.0.1", 0))
    return reserved


def _stop_owned_process(process):
    """Reap only the Popen instance this fixture created, including on failure."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_for_health(process, client, log_path):
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_result = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"Uvicorn exited with {process.returncode}: "
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            )
        try:
            response = client.get("/health", timeout=0.5)
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
            last_result = f"HTTP {response.status_code}: {response.text}"
        except (httpx.TransportError, ValueError) as error:
            last_result = repr(error)
        time.sleep(0.05)
    pytest.fail(
        f"Uvicorn not healthy within {STARTUP_TIMEOUT_SECONDS}s: {last_result}; "
        f"{log_path.read_text(encoding='utf-8', errors='replace')}"
    )


@pytest.fixture
def live_services(tmp_path):
    checkpoint = _synthetic_checkpoint(tmp_path / "synthetic_bigru.pt")
    audio_dir = tmp_path / "audio"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(REPO_ROOT), str(REPO_ROOT / "ml-service"))),
            "SIGNBRIDGE_MODEL_PATH": str(checkpoint),
            "SIGNBRIDGE_DEVICE": "cpu",
            "MOCK_EXTERNAL_SERVICES": "1",
            "AUDIO_DIR": str(audio_dir),
            "PUBLIC_BASE_URL": "",
            "REQUIRE_HTTPS_PUBLIC_BASE_URL": "0",
            "GEMINI_API_KEY": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "TTS_LANGUAGE_CODE": "en-IN",
            "TTS_VOICE_NAME": "",
            "TTS_VOICE_GENDER": "NEUTRAL",
            "TTS_TIMEOUT_SECONDS": "10",
            "STT_LANGUAGE_CODE": "en-IN",
            "STT_TIMEOUT_SECONDS": "10",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    # Hold both reservations until their respective launches so the selected
    # ports are distinct. Uvicorn binds immediately after each socket closes.
    with ExitStack() as resources:
        reservations = [resources.enter_context(_reserve_loopback_port()) for _ in range(2)]
        ports = [reserved.getsockname()[1] for reserved in reservations]
        services = []
        for name, target, reserved, port in zip(
            ("ml", "backend"),
            ("service.app:app", "backend.main:app"),
            reservations,
            ports,
        ):
            log_path = tmp_path / f"{name}-uvicorn.log"
            log = resources.enter_context(log_path.open("wb"))
            reserved.close()
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", target,
                    "--host", "127.0.0.1", "--port", str(port),
                    "--workers", "1", "--loop", "asyncio", "--http", "h11",
                    "--no-access-log", "--log-level", "warning",
                    "--timeout-graceful-shutdown", "2",
                ],
                cwd=REPO_ROOT,
                env=environment.copy(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            resources.callback(_stop_owned_process, process)
            client = resources.enter_context(
                httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False)
            )
            services.append((process, client, log_path))
        for process, client, log_path in services:
            _wait_for_health(process, client, log_path)
        yield services[0][1], services[1][1], audio_dir
    # ExitStack's finally path closes clients, terminates/reaps owned children,
    # and closes files/sockets even when startup or an assertion fails.


def _wav_base64(*, silent=False):
    stream = BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((b"\x00\x00" if silent else b"\x01\x00") * 160)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_two_process_http_vertical_slice(live_services):
    ml, backend, audio_dir = live_services
    for client in (ml, backend):
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

    # Nonzero short sequence plus right-padding exercises M4/M5 temporal
    # normalization in the actual ML child, without any fake engine injection.
    landmarks = np.zeros((5, 110, 3), dtype=np.float32)
    landmarks[:3, 0, 0] = [0.1, 0.25, 0.5]
    predicted = ml.post("/predict", json={"landmarks": landmarks.tolist()})
    assert predicted.status_code == 200, predicted.text
    assert predicted.json() == {"keywords": ["pay"]}

    # The HTTP client forwards the ML result; backend has no ML dependency.
    processed = backend.post("/process", json=predicted.json())
    assert processed.status_code == 200, processed.text
    body = processed.json()
    assert body["sentence"] == "Pay."
    audio_url = urlsplit(body["audio_url"])
    assert audio_url.scheme == "http"
    assert audio_url.hostname == "127.0.0.1"
    assert audio_url.port == backend.base_url.port
    assert audio_url.path.startswith("/audio/") and audio_url.path.endswith(".mp3")
    # Fetch the absolute URL exactly as the mobile app receives it.
    audio = backend.get(body["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/mpeg")
    assert audio.content.startswith(b"ID3")
    assert b"SignBridge mock audio" in audio.content
    assert audio.content == (audio_dir / Path(audio_url.path).name).read_bytes()

    created = backend.post("/session/create")
    assert created.status_code == 200, created.text
    room = created.json()["room_id"]
    assert isinstance(room, str) and room
    empty = backend.get(f"/session/{room}/messages")
    assert empty.status_code == 200
    assert empty.json() == {"messages": []}
    started = datetime.now(timezone.utc)
    signed = backend.post(
        f"/session/{room}/message", json={"sender": "signer", "text": body["sentence"]}
    )
    assert signed.status_code == 200
    assert signed.json() == {"status": "ok"}

    transcribed = backend.post("/transcribe", json={"audio": _wav_base64()})
    assert transcribed.status_code == 200, transcribed.text
    assert transcribed.json() == {"text": MOCK_TRANSCRIPT}
    spoken = backend.post(
        f"/session/{room}/message",
        json={"sender": "speaker", "text": transcribed.json()["text"]},
    )
    assert spoken.status_code == 200
    assert spoken.json() == {"status": "ok"}
    response = backend.get(f"/session/{room}/messages")
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [(message["sender"], message["text"]) for message in messages] == [
        ("signer", body["sentence"]),
        ("speaker", transcribed.json()["text"]),
    ]
    timestamps = [datetime.fromisoformat(message["timestamp"]) for message in messages]
    assert all(message["timestamp"].endswith("Z") for message in messages)
    assert all(timestamp.utcoffset() == timedelta(0) for timestamp in timestamps)
    assert timestamps == sorted(timestamps)
    assert all(started <= timestamp <= datetime.now(timezone.utc) for timestamp in timestamps)

    # Rejected input must leave the conversation intact, and services available.
    for payload in ({}, {"landmarks": []}, {"landmarks": np.zeros((61, 110, 3)).tolist()}):
        rejected = ml.post("/predict", json=payload)
        assert rejected.status_code == 422
        assert "detail" in rejected.json()
    invalid_keywords = backend.post("/process", json={"keywords": []})
    assert invalid_keywords.status_code == 422
    invalid_audio = backend.post("/transcribe", json={"audio": "not-base64***"})
    assert invalid_audio.status_code == 400
    assert invalid_audio.json() == {"detail": "audio must be valid base64"}
    silence = backend.post("/transcribe", json={"audio": _wav_base64(silent=True)})
    assert silence.status_code == 200 and silence.json() == {"text": ""}
    rejected_message = backend.post(
        f"/session/{room}/message", json={"sender": "speaker", "text": silence.json()["text"]}
    )
    assert rejected_message.status_code == 422
    for missing in (
        backend.get("/session/missing-room/messages"),
        backend.post("/session/missing-room/message", json={"sender": "signer", "text": "hello"}),
    ):
        assert missing.status_code == 404
        assert missing.json() == {"detail": "conversation room not found"}
    missing_audio = backend.get("/audio/00000000000000000000000000000000.mp3")
    assert missing_audio.status_code == 404
    assert missing_audio.json() == {"detail": "audio not found"}
    after_errors = backend.get(f"/session/{room}/messages")
    assert after_errors.status_code == 200 and after_errors.json() == response.json()
    assert ml.post("/predict", json={"landmarks": landmarks.tolist()}).json() == predicted.json()
