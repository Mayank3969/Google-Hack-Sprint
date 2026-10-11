"""M2 tests for two hand slots plus a compact 68-point face block."""

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import extract_landmarks_module as extraction


def point(value=0.25):
    return NS(x=value, y=0.5, z=-0.1)


def hand(value):
    return [point(value) for _ in range(21)]


def face(value=0.75):
    return [point(value) for _ in range(478)]


@pytest.fixture
def files(tmp_path):
    video = tmp_path / "clip.mp4"
    hand_model = tmp_path / "hand.task"
    face_model = tmp_path / "face.task"
    for path in (video, hand_model, face_model):
        path.write_bytes(b"fixture")
    return video, hand_model, face_model


def fake_pipeline(monkeypatch, *, hand_results=None, face_results=None, positions=None, fps=25.0):
    hand_results = [NS(hand_landmarks=[], handedness=[])] if hand_results is None else hand_results
    face_results = [NS(face_landmarks=[])] * len(hand_results) if face_results is None else face_results
    positions = [i * 40.0 for i in range(len(hand_results))] if positions is None else positions
    state = NS(timestamps=[], hand_options=None, face_options=None, released=0, closed=0, reads=0)

    class Capture:
        def isOpened(self): return True
        def read(self):
            index = state.reads
            state.reads += 1
            if index >= len(hand_results): return False, None
            return True, np.ones((2, 2, 3), dtype=np.uint8)
        def get(self, prop):
            return positions[state.reads - 1] if prop == "position" else fps
        def release(self): state.released += 1

    class Landmarker:
        def __init__(self, kind): self.kind = kind
        def detect_for_video(self, image, timestamp):
            state.timestamps.append(timestamp) if self.kind == "hand" else None
            index = len(state.timestamps) - 1
            return (hand_results[index] if self.kind == "hand" else face_results[index])
        def close(self): state.closed += 1

    def create_hand(options):
        state.hand_options = options
        return Landmarker("hand")
    def create_face(options):
        state.face_options = options
        return Landmarker("face")
    cv2 = NS(VideoCapture=lambda path: Capture(), CAP_PROP_POS_MSEC="position", CAP_PROP_FPS="fps",
             COLOR_BGR2RGB="rgb", cvtColor=lambda frame, code: frame)
    mp = NS(Image=lambda **kwargs: NS(**kwargs), ImageFormat=NS(SRGB="srgb"))
    python = NS(BaseOptions=NS)
    vision = NS(
        HandLandmarkerOptions=NS, FaceLandmarkerOptions=NS,
        RunningMode=NS(VIDEO="video"),
        HandLandmarker=NS(create_from_options=create_hand),
        FaceLandmarker=NS(create_from_options=create_face),
    )
    monkeypatch.setattr(extraction, "_load_dependencies", lambda: (cv2, mp, python, vision))
    return state


def test_two_hands_and_face_are_stably_slotted(monkeypatch, files):
    hands = NS(hand_landmarks=[hand(2), hand(1)], handedness=[[NS(category_name="Right")], [NS(category_name="Left")]])
    state = fake_pipeline(monkeypatch, hand_results=[hands], face_results=[NS(face_landmarks=[face()])])
    actual = extraction.extract_landmarks(files[0], model_path=files[1], face_model_path=files[2])
    assert actual.shape == (1, 110, 3)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0, 0:21, 0], 1.0)  # left hand
    np.testing.assert_allclose(actual[0, 21:42, 0], 2.0)  # right hand
    np.testing.assert_allclose(actual[0, 42:, 0], 0.75)
    assert len(extraction.FACE_LANDMARK_INDICES) == 68
    assert state.hand_options.num_hands == 2
    assert state.released == 1 and state.closed == 2


def test_missing_hands_and_face_are_zero_blocks(monkeypatch, files):
    state = fake_pipeline(monkeypatch, hand_results=[NS(hand_landmarks=[], handedness=[])], face_results=[NS(face_landmarks=[])])
    actual = extraction.extract_landmarks(files[0], model_path=files[1], face_model_path=files[2])
    assert actual.shape == (1, 110, 3)
    assert not actual.any()
    assert state.released == 1 and state.closed == 2


def test_empty_video_shape_and_required_face_model(monkeypatch, files):
    state = fake_pipeline(monkeypatch, hand_results=[])
    actual = extraction.extract_landmarks(files[0], model_path=files[1], face_model_path=files[2])
    assert actual.shape == (0, 110, 3)
    assert state.closed == 0
    with pytest.raises(FileNotFoundError, match="face-landmarker model is required"):
        extraction.extract_landmarks(files[0], model_path=files[1])


@pytest.mark.parametrize("source_count", [1, 68, 468, 477, 479])
def test_invalid_face_shape_is_rejected(monkeypatch, files, source_count):
    state = fake_pipeline(monkeypatch, hand_results=[NS(hand_landmarks=[], handedness=[])], face_results=[NS(face_landmarks=[[point()] * source_count])])
    with pytest.raises(ValueError, match="478"):
        extraction.extract_landmarks(files[0], model_path=files[1], face_model_path=files[2])
    assert state.released == 1 and state.closed == 2


def test_curated_face_selection_order():
    expected = (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,
                263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,
                70,63,105,66,107,336,296,334,293,300,
                61,146,91,181,84,17,314,405,321,375,
                78,95,88,178,87,14,317,402,318,324,1,2,4,5,6,168)
    assert extraction.FACE_FEATURE_VERSION == "compact-v1"
    assert extraction.FACE_LANDMARK_INDICES == expected
    source = [point(i) for i in range(478)]
    output = extraction._frame_points(NS(hand_landmarks=[]), NS(face_landmarks=[source]), 0)
    np.testing.assert_array_equal(output[42:, 0], expected)
    assert output.shape == (110, 3)
