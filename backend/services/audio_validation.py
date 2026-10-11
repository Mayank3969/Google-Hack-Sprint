"""Validation for the bounded WAV format accepted by SignBridge STT.

The MVP accepts only little-endian RIFF/WAVE files containing uncompressed
PCM (format code 1), mono 16-bit samples at 16,000 Hz.  Compressed, floating
point, and WAVE_FORMAT_EXTENSIBLE audio are intentionally unsupported.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MAX_AUDIO_DURATION_SECONDS = 30
MAX_AUDIO_BYTES = 1_000_000


class AudioValidationError(ValueError):
    """Raised when an uploaded WAV is not safe and supported audio."""


@dataclass(frozen=True)
class ValidatedWav:
    """The checked WAV metadata and its bounded PCM sample payload."""

    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    pcm_frames: bytes


def validate_wav(audio_bytes: bytes) -> ValidatedWav:
    """Validate a bounded WAV and return its PCM frames.

    Parsing is performed against the supplied in-memory bytes only.  Chunk
    sizes are checked before each offset advance, and the RIFF size must cover
    the complete input so bytes after the container cannot be smuggled in.
    """

    if type(audio_bytes) is not bytes:
        raise AudioValidationError("audio must be bytes")
    if not audio_bytes:
        raise AudioValidationError("audio must not be empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise AudioValidationError("audio exceeds the maximum size")

    if len(audio_bytes) < 12:
        raise AudioValidationError("audio is not a complete RIFF/WAVE file")
    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise AudioValidationError("audio must use a RIFF/WAVE container")

    riff_size = struct.unpack_from("<I", audio_bytes, 4)[0]
    riff_end = riff_size + 8
    if riff_size < 4 or riff_end != len(audio_bytes):
        raise AudioValidationError("RIFF container length is invalid")

    fmt_range: tuple[int, int] | None = None
    data_range: tuple[int, int] | None = None
    offset = 12
    while offset < riff_end:
        if riff_end - offset < 8:
            raise AudioValidationError("RIFF chunk header is truncated")

        chunk_id = audio_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio_bytes, offset + 4)[0]
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > riff_end:
            raise AudioValidationError("RIFF chunk extends beyond the container")

        next_offset = payload_end + (chunk_size & 1)
        if next_offset > riff_end:
            raise AudioValidationError("RIFF chunk padding is truncated")

        if chunk_id == b"fmt ":
            if fmt_range is not None:
                raise AudioValidationError("WAV contains duplicate format chunks")
            fmt_range = (payload_start, payload_end)
        elif chunk_id == b"data":
            if data_range is not None:
                raise AudioValidationError("WAV contains duplicate data chunks")
            data_range = (payload_start, payload_end)

        offset = next_offset

    if offset != riff_end:
        raise AudioValidationError("RIFF container length is invalid")
    if fmt_range is None or data_range is None:
        raise AudioValidationError("WAV must contain one format and one data chunk")

    fmt_start, fmt_end = fmt_range
    data_start, data_end = data_range
    if fmt_end - fmt_start < 16:
        raise AudioValidationError("WAV format chunk is truncated")

    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = (
        struct.unpack_from("<HHIIHH", audio_bytes, fmt_start)
    )
    if audio_format != 1:
        raise AudioValidationError("WAV must contain uncompressed PCM format 1")
    if channels != CHANNELS:
        raise AudioValidationError("WAV must be mono")
    if sample_rate != SAMPLE_RATE_HZ:
        raise AudioValidationError("WAV sample rate is unsupported")
    if bits_per_sample != SAMPLE_WIDTH_BYTES * 8:
        raise AudioValidationError("WAV sample width is unsupported")

    expected_block_align = CHANNELS * SAMPLE_WIDTH_BYTES
    expected_byte_rate = SAMPLE_RATE_HZ * expected_block_align
    if block_align != expected_block_align:
        raise AudioValidationError("WAV block alignment is invalid")
    if byte_rate != expected_byte_rate:
        raise AudioValidationError("WAV byte rate is invalid")

    data_size = data_end - data_start
    if data_size == 0 or data_size % block_align:
        raise AudioValidationError("WAV PCM data is not frame-aligned")

    frame_count = data_size // block_align
    max_frame_count = MAX_AUDIO_DURATION_SECONDS * SAMPLE_RATE_HZ
    if frame_count == 0 or frame_count > max_frame_count:
        raise AudioValidationError("WAV duration is outside the supported range")

    return ValidatedWav(
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=SAMPLE_WIDTH_BYTES,
        frame_count=frame_count,
        duration_seconds=frame_count / SAMPLE_RATE_HZ,
        pcm_frames=audio_bytes[data_start:data_end],
    )
