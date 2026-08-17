"""Speech-to-Text (STT) integration using Sarvam AI REST API (saaras:v3, transcribe mode).

Transcribes audio byte payloads (WAV, MP3, WebM, AAC) into Indian English, Hindi,
or code-mixed text with detected language codes and latency tracking.
"""

from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import httpx

from pipeline.config import get_settings
from pipeline.schemas import AudioInput

SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
DEFAULT_STT_MODEL = "saaras:v3"
DEFAULT_LANGUAGE_CODE = "hi-IN"
DEFAULT_MODE = "transcribe"  # Strictly transcribe, not translate


def transcribe(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    model: str = DEFAULT_STT_MODEL,
    mode: str = DEFAULT_MODE,
    filename: str = "audio.wav",
    api_key: Optional[str] = None,
    timeout_s: float = 5.0,
) -> Tuple[str, str, float]:
    """Transcribe audio bytes to text using Sarvam AI STT REST API.

    Parameters
    ----------
    audio_bytes:
        Raw audio byte stream (e.g. WAV, MP3, WebM).
    language_code:
        Target BCP-47 / Indic language code (e.g. 'hi-IN', 'en-IN', 'bn-IN').
    model:
        Sarvam STT model identifier (default: 'saaras:v3').
    mode:
        Operation mode: 'transcribe' (default) or 'translate'.
    filename:
        Filename identifier for multipart upload.
    api_key:
        Sarvam API key (defaults to SARVAM_API_KEY from environment).
    timeout_s:
        Network request timeout in seconds.

    Returns
    -------
    Tuple[str, str, float]
        (transcript_text, detected_language_code, duration_ms)
    """
    t0 = time.perf_counter()
    settings = get_settings()
    key = api_key or settings.sarvam_api_key

    # Fallback if API key is not configured (mock for offline / test environments)
    if not key:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        # Return empty or placeholder indicator for mock audio
        return "", language_code, duration_ms

    headers = {
        "api-subscription-key": key,
    }

    files = {
        "file": (filename, audio_bytes, "audio/wav"),
    }
    data = {
        "model": model,
        "language_code": language_code,
        "mode": mode,
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                SARVAM_STT_ENDPOINT,
                headers=headers,
                data=data,
                files=files,
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code == 200:
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            detected_lang = res_json.get("language_code", language_code)
            return transcript, detected_lang, duration_ms
        else:
            return "", language_code, duration_ms
    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return "", language_code, duration_ms


async def atranscribe(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    model: str = DEFAULT_STT_MODEL,
    mode: str = DEFAULT_MODE,
    filename: str = "audio.wav",
    api_key: Optional[str] = None,
    timeout_s: float = 5.0,
) -> Tuple[str, str, float]:
    """Asynchronous variant of transcribe for FastAPI async request handlers."""
    t0 = time.perf_counter()
    settings = get_settings()
    key = api_key or settings.sarvam_api_key

    if not key:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return "", language_code, duration_ms

    headers = {
        "api-subscription-key": key,
    }
    files = {
        "file": (filename, audio_bytes, "audio/wav"),
    }
    data = {
        "model": model,
        "language_code": language_code,
        "mode": mode,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                SARVAM_STT_ENDPOINT,
                headers=headers,
                data=data,
                files=files,
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code == 200:
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            detected_lang = res_json.get("language_code", language_code)
            return transcript, detected_lang, duration_ms
        else:
            return "", language_code, duration_ms
    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return "", language_code, duration_ms


# ---------------------------------------------------------------------------
# Class-based STT interface
# ---------------------------------------------------------------------------

class BaseSTT(ABC):
    """Abstract base class for Speech-To-Text transcription."""

    @abstractmethod
    def transcribe(self, audio: AudioInput) -> Tuple[str, float]:
        """Transcribe audio input returning text and latency in milliseconds."""
        raise NotImplementedError

    @abstractmethod
    async def atranscribe(self, audio: AudioInput) -> Tuple[str, float]:
        """Asynchronously transcribe audio input."""
        raise NotImplementedError


class SarvamSTT(BaseSTT):
    """Sarvam AI Speech-To-Text transcription wrapper."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: Optional[str] = None,
        model: str = DEFAULT_STT_MODEL,
    ) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.sarvam_api_key
        self.language_code = language_code or self.settings.sarvam_language_code or DEFAULT_LANGUAGE_CODE
        self.model = model

    def transcribe(self, audio: AudioInput) -> Tuple[str, float]:
        """Synchronously transcribe AudioInput object."""
        raw_bytes = audio.audio_bytes or b""
        if not raw_bytes and audio.audio_base64:
            import base64
            raw_bytes = base64.b64decode(audio.audio_base64)

        lang = audio.language_code or self.language_code
        text, detected_lang, duration_ms = transcribe(
            raw_bytes,
            language_code=lang,
            model=self.model,
            api_key=self.api_key,
        )
        return text, duration_ms

    async def atranscribe(self, audio: AudioInput) -> Tuple[str, float]:
        """Asynchronously transcribe AudioInput object."""
        raw_bytes = audio.audio_bytes or b""
        if not raw_bytes and audio.audio_base64:
            import base64
            raw_bytes = base64.b64decode(audio.audio_base64)

        lang = audio.language_code or self.language_code
        text, detected_lang, duration_ms = await atranscribe(
            raw_bytes,
            language_code=lang,
            model=self.model,
            api_key=self.api_key,
        )
        return text, duration_ms
