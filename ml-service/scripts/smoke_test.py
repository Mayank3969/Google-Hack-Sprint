"""Small local smoke check for a running ML service.

Usage (with the service running locally or in Docker)::

    python scripts/smoke_test.py --base-url http://127.0.0.1:8080

The health check always runs.  A prediction check runs only when the configured
classifier artifact exists locally; this script never invents a model or a
prediction.  Set MODEL_PATH if the artifact is stored outside ``models/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(base_url: str, path: str, *, payload: dict | None = None) -> tuple[int, dict]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers)
    with urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    try:
        status, body = _request(args.base_url, "/health")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"FAIL /health: service is not reachable ({exc})")
        return 1
    if status != 200 or body.get("status") != "ok":
        print(f"FAIL /health: HTTP {status} {body}")
        return 1
    print("PASS /health")

    artifact = Path(
        os.environ.get(
            "MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "classifier.pkl"
        )
    )
    if not artifact.is_file():
        print(f"SKIP /predict: classifier artifact is absent ({artifact})")
        return 0

    landmarks = [[[0.0, 0.0, 0.0] for _ in range(21)]]
    try:
        status, body = _request(args.base_url, "/predict", payload={"landmarks": landmarks})
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"FAIL /predict: request failed ({exc})")
        return 1
    if status != 200 or not isinstance(body.get("keywords"), list):
        print(f"FAIL /predict: HTTP {status} {body}")
        return 1
    print("PASS /predict (artifact present; response shape verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
