"""A small deterministic MP3 fixture for local Text-to-Speech tests.

The fixture contains an ID3 marker identifying it as SignBridge test audio and
two silent MPEG-1 Layer III frames. It is intentionally not a recording and
must never be used as evidence of a real Text-to-Speech response.
"""


def _syncsafe(value: int) -> bytes:
    """Encode a small ID3v2 size using four sync-safe bytes."""

    return bytes(
        (
            (value >> 21) & 0x7F,
            (value >> 14) & 0x7F,
            (value >> 7) & 0x7F,
            value & 0x7F,
        )
    )


def _build_fixture() -> bytes:
    marker = b"\x00SignBridge\x00SignBridge mock audio"
    comment_frame = (
        b"TXXX"
        + len(marker).to_bytes(4, "big")
        + b"\x00\x00"
        + marker
    )
    id3_tag = b"ID3\x04\x00\x00" + _syncsafe(len(comment_frame)) + comment_frame

    # MPEG-1 Layer III, 128 kbps, 44.1 kHz, joint stereo. A frame is 417
    # bytes at this bitrate/sample-rate combination; zeroed side/main data
    # gives a deterministic silent test payload.
    frame_header = bytes.fromhex("FF FB 90 64")
    frame = frame_header + bytes(417 - len(frame_header))
    return id3_tag + frame + frame


MOCK_MP3_BYTES = _build_fixture()
MOCK_MP3_MARKER = b"SignBridge mock audio"
