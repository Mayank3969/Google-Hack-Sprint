from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scripts import extract_landmarks_module as module


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, detected):
        self.detected = detected
        self.closed = False

    def process(self, _rgb):
        if not self.detected:
            return type("Result", (), {"multi_hand_landmarks": []})()
        points = [type("Point", (), {"x": i / 21, "y": 0.5, "z": -0.1})() for i in range(21)]
        hand = type("Hand", (), {"landmark": points})()
        return type("Result", (), {"multi_hand_landmarks": [hand]})()

    def close(self):
        self.closed = True


def test_extract_shape_zero_fills_and_closes_resources(monkeypatch, tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"fixture")
    capture = FakeCapture([np.ones((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)])
    detector = FakeDetector(detected=False)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(module, "_legacy_detector", lambda: detector)
    monkeypatch.setattr(module.cv2, "cvtColor", lambda frame, _: frame)

    result = module.extract_landmarks(video)

    assert result.shape == (2, 21, 3)
    assert result.dtype == np.float32
    assert np.count_nonzero(result) == 0
    assert capture.released and detector.closed


def test_missing_video_has_clear_error():
    with pytest.raises(FileNotFoundError, match="input video does not exist"):
        module.extract_landmarks("does-not-exist.mp4")


def test_unopenable_video_releases_capture(monkeypatch, tmp_path):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"fixture")
    capture = FakeCapture([], opened=False)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(RuntimeError, match="could not open input video"):
        module.extract_landmarks(video)
    assert capture.released


def test_numbered_wrapper_imports_from_repository_root():
    script = Path(__file__).resolve().parents[2] / "ml-service" / "scripts" / "01_extract_landmarks.py"
    spec = importlib.util.spec_from_file_location("m2_wrapper", script)
    wrapper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wrapper)
    assert callable(wrapper.main)
