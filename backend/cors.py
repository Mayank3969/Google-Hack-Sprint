"""Opt-in, exact-origin browser access for the React frontend."""

import ipaddress
import os
import re
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def allowed_origins(value: str) -> list[str]:
    """Validate HTTP(S) origins, failing closed on invalid configuration."""

    origins: list[str] = []
    for entry in value.split(","):
        origin = entry.strip()
        if not origin:
            continue
        try:
            parsed = urlsplit(origin)
            valid = (
                origin.startswith(("http://", "https://"))
                and bool(parsed.hostname)
                and not parsed.path
                and not any(char in origin for char in "*?#\\")
                and not any(char.isspace() for char in origin)
                and parsed.username is None
                and parsed.password is None
                and not parsed.netloc.endswith(":")
                and (parsed.port is None or 1 <= parsed.port <= 65535)
            )
            hostname = parsed.hostname or ""
            if ":" in hostname:
                ipaddress.IPv6Address(hostname)
                valid = valid and "%" not in hostname
            else:
                valid = valid and len(hostname) <= 253 and all(
                    re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                    for label in hostname.split(".")
                )
            if not valid:
                raise ValueError("invalid origin")
        except ValueError as exc:
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must contain exact HTTP(S) origins "
                "without wildcards, credentials, paths, queries, or fragments"
            ) from exc
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(app: FastAPI) -> None:
    origins = allowed_origins(os.getenv("CORS_ALLOWED_ORIGINS", ""))
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
