"""BE-M12 deterministic bilingual mock plumbing matrix.

These cases prove service wiring only. They do not claim translation quality or
model accuracy, and they must remain provider-free in local/CI mode.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
    from backend.routes import process as process_route
    from backend.services import gemini_client, tts_client
    from backend.fixtures.mock_audio import MOCK_MP3_BYTES_EN_IN, MOCK_MP3_BYTES_HI_IN
except ModuleNotFoundError:
    from main import app
    from routes import process as process_route
    from services import gemini_client, tts_client
    from fixtures.mock_audio import MOCK_MP3_BYTES_EN_IN, MOCK_MP3_BYTES_HI_IN


client = TestClient(app)


@pytest.fixture
def isolated_mock_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(process_route, "AUDIO_DIR", tmp_path)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("REQUIRE_HTTPS_PUBLIC_BASE_URL", "0")
    monkeypatch.setenv("MOCK_EXTERNAL_SERVICES", "1")

    def unexpected_gemini_setup(*args, **kwargs):
        pytest.fail("M12 mock mode must not initialize Gemini")

    def unexpected_tts_setup(*args, **kwargs):
        pytest.fail("M12 mock mode must not initialize Google TTS")

    monkeypatch.setattr(gemini_client, "_create_real_model", unexpected_gemini_setup)
    monkeypatch.setattr(tts_client, "_load_texttospeech", unexpected_tts_setup)
    monkeypatch.setattr(tts_client, "_create_real_client", unexpected_tts_setup)
    return tmp_path


@pytest.mark.parametrize(
    "keywords, english, hindi_keywords, hindi, is_question",
    [
        (
            ["pay", "medicine", "?"],
            "Can I pay for this medicine?",
            ["भुगतान", "दवा", "?"],
            "क्या मैं इस दवा के लिए भुगतान कर सकता हूँ?",
            True,
        ),
        (["hello"], "Hello.", ["नमस्ते"], "नमस्ते।", False),
        (
            ["need", "water"],
            "I need water.",
            ["ज़रूरत", "पानी"],
            "मुझे पानी चाहिए।",
            False,
        ),
        (
            ["doctor", "tomorrow", "?"],
            "Can I see the doctor tomorrow?",
            ["डॉक्टर", "कल", "?"],
            "क्या मैं कल डॉक्टर से मिल सकता हूँ?",
            True,
        ),
        (
            ["emergency", "hospital"],
            "It is an emergency; I need to go to the hospital.",
            ["आपातकाल", "अस्पताल"],
            "यह आपातकाल है; मुझे अस्पताल जाना है।",
            False,
        ),
    ],
    ids=["question", "greeting", "request", "scheduling-question", "urgent"],
)
def test_m12_mock_matrix_covers_bilingual_process_plumbing(
    isolated_mock_process: Path,
    keywords: list[str],
    english: str,
    hindi_keywords: list[str],
    hindi: str,
    is_question: bool,
) -> None:
    response = client.post("/process", json={"keywords": keywords})

    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == english
    assert body["hindi_keywords"] == hindi_keywords
    assert body["hindi_sentence"] == hindi
    assert body["sentence"].endswith("?") is is_question
    assert body["hindi_sentence"].endswith("?") is is_question
    assert body["playback_order"] == ["en-IN", "hi-IN"]
    assert body["subtitles"] == [
        {"language": "en-IN", "text": english},
        {"language": "hi-IN", "text": hindi},
    ]
    assert [segment["language"] for segment in body["audio_segments"]] == [
        "en-IN", "hi-IN"
    ]
    assert body["audio_url"] == body["audio_segments"][0]["audio_url"]

    english_path = "/" + body["audio_segments"][0]["audio_url"].split("/", 3)[-1]
    hindi_path = "/" + body["audio_segments"][1]["audio_url"].split("/", 3)[-1]
    english_audio = client.get(english_path)
    hindi_audio = client.get(hindi_path)

    assert english_audio.status_code == 200
    assert hindi_audio.status_code == 200
    assert english_audio.headers["content-type"].startswith("audio/mpeg")
    assert hindi_audio.headers["content-type"].startswith("audio/mpeg")
    assert english_audio.content == MOCK_MP3_BYTES_EN_IN
    assert hindi_audio.content == MOCK_MP3_BYTES_HI_IN
    assert english_audio.content
    assert hindi_audio.content
    assert len(list(isolated_mock_process.glob("*.mp3"))) == 2


def test_m12_mock_matrix_is_repeatable(isolated_mock_process: Path) -> None:
    payload = {"keywords": ["doctor", "tomorrow", "?"]}

    first = client.post("/process", json=payload).json()
    first_audio = client.get(
        "/" + first["audio_segments"][0]["audio_url"].split("/", 3)[-1]
    ).content
    second = client.post("/process", json=payload).json()
    second_audio = client.get(
        "/" + second["audio_segments"][0]["audio_url"].split("/", 3)[-1]
    ).content

    assert first["sentence"] == second["sentence"]
    assert first["hindi_keywords"] == second["hindi_keywords"]
    assert first["hindi_sentence"] == second["hindi_sentence"]
    assert first["subtitles"] == second["subtitles"]
    assert first["playback_order"] == second["playback_order"]
    assert first_audio == second_audio == MOCK_MP3_BYTES_EN_IN
