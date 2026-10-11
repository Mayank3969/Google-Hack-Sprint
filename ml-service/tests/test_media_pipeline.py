"""Real decoder/Tasks/CLI smoke tests on generated non-personal video only."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.extract_landmarks_module import DEFAULT_MODEL_PATH, extract_landmarks


pytestmark = pytest.mark.integration
SERVICE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    assert DEFAULT_MODEL_PATH.is_file(), "checked-in Tasks model is missing"
    face_model = SERVICE_ROOT / "models" / "face_landmarker.task"
    if not face_model.is_file():
        pytest.skip("face_landmarker.task is not checked into this workspace")
    return cv2, face_model


def make_video(cv2, path, *, frames=12, codec="MJPG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 15.0, (96, 96))
    try:
        assert writer.isOpened(), f"test codec {codec} unavailable"
        for _ in range(frames):
            writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize("extension,codec", [(".avi", "MJPG"), (".mp4", "mp4v")])
def test_real_video_decoding_and_no_hand_landmarks(media, tmp_path, extension, codec):
    cv2, face_model = media
    video = tmp_path / ("blank" + extension)
    make_video(cv2, video, codec=codec)
    actual = extract_landmarks(video, face_model_path=face_model)
    assert actual.shape == (12, 110, 3)
    assert actual.dtype == np.float32
    assert not actual.any()


def run_cli(*args, cwd):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result


def test_real_cli_pipeline_and_repeatability(media, tmp_path):
    cv2, face_model = media
    raw = tmp_path / "raw"
    clip = raw / "synthetic_a" / "blank.avi"
    make_video(cv2, clip)
    make_video(cv2, raw / "synthetic_b" / "blank.avi", frames=65)

    extracted = tmp_path / "clip.npy"
    run_cli(SERVICE_ROOT / "scripts/01_extract_landmarks.py", clip, "--face-model", face_model, "--output", extracted,
            cwd=tmp_path)
    assert np.load(extracted, allow_pickle=False).shape == (12, 110, 3)
    run_cli("-m", "scripts.01_extract_landmarks", clip, "--face-model", face_model, "--output", extracted, "--overwrite",
            cwd=SERVICE_ROOT)

    output = tmp_path / "artifacts" / "dataset.npz"
    result = run_cli(SERVICE_ROOT / "scripts/02_build_dataset.py", "--raw-dir", raw,
                     "--output", output, "--face-model", face_model, cwd=tmp_path)
    assert "X=(2, 60, 110, 3), y=(2,)" in result.stdout
    with np.load(output, allow_pickle=False) as dataset:
        X, y = dataset["X"], dataset["y"]
    assert X.shape == (2, 60, 110, 3)
    assert not X.any()  # synthetic no-hand data is NOT a training/accuracy claim
    assert y.tolist() == ["synthetic_a", "synthetic_b"]
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["per_label_counts"] == {"synthetic_a": 1, "synthetic_b": 1}
    again = output.with_name("again.npz")
    run_cli("-m", "scripts.02_build_dataset", "--raw-dir", raw, "--output", again,
            "--face-model", face_model, cwd=SERVICE_ROOT)
    with np.load(again, allow_pickle=False) as rebuilt:
        np.testing.assert_array_equal(rebuilt["X"], X)
        np.testing.assert_array_equal(rebuilt["y"], y)


def test_real_corrupt_video_fails(media, tmp_path):
    _, face_model = media
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could not open input video"):
        extract_landmarks(video, face_model_path=face_model)
