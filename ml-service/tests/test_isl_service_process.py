"""B11/B12 real Uvicorn + real Keras weights over loopback HTTP.

INTEGRATION_WARNING: generated landmark motion is NOT a labeled sign. This
proves startup/request/response mechanics, not Person A parity or recognition.
No backend/cloud requests are made. Only this test's own process is terminated.
"""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest


@pytest.mark.real_model
def test_real_service_liveness_readiness_and_batch(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env["TF_NUM_INTRAOP_THREADS"] = "2"
    env["TF_NUM_INTEROP_THREADS"] = "2"
    log_path = tmp_path / "isl-service.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "service.app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=90) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        pytest.fail(log_path.read_text(encoding="utf-8"))
                    try:
                        if client.get("/health", timeout=.5).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        pytest.fail("service liveness timeout")
                    time.sleep(.1)
                readiness = client.get("/v2/readiness")
                assert readiness.status_code == 200, readiness.text
                assert readiness.json()["model_ready"]
                assert not readiness.json()["integration_ready"]
                raw = np.random.default_rng(93).random((12, 110, 3)).tolist()
                response = client.post("/v2/predict/batch", json={"capture_id": "loopback", "segments": [
                    {"segment_index": 1, "landmarks": np.zeros((12, 110, 3)).tolist()},
                    {"segment_index": 0, "landmarks": raw},
                ]})
                assert response.status_code == 200, response.text
                body = response.json()
                assert not body["complete"] and body["keywords"] == []
                assert [r["segment_index"] for r in body["results"]] == [0, 1]
                assert body["results"][1]["status"] == "insufficient_data"
                assert len(body["results"][0]["top_k"]) == 3
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
