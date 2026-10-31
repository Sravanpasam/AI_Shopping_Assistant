"""
transcription.py
----------------
Speech-to-text audio transcription using OpenAI Whisper.
Transcribes audio recordings with automatic language detection (supporting English and Arabic).
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional
import whisper

logger = logging.getLogger(__name__)

# Module-level model references
_whisper_model: Optional[whisper.Whisper] = None
_whisper_models: Dict[str, whisper.Whisper] = {}


def get_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Lazy singleton loader for the Whisper ASR model.
    Caches loaded models by model size.

    Args:
        model_size: Model size string ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        Loaded whisper.Whisper instance.
    """
    global _whisper_model, _whisper_models
    size = (model_size or "base").strip().lower()

    if size not in _whisper_models:
        logger.info("Loading Whisper ASR model ('%s')...", size)
        _whisper_models[size] = whisper.load_model(size)
        logger.info("Whisper model ('%s') loaded successfully.", size)

    _whisper_model = _whisper_models[size]
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, model_size: str = "base") -> Dict[str, Any]:
    """
    Transcribe raw audio bytes using OpenAI Whisper with automatic language detection.

    Args:
        audio_bytes: Raw bytes of the audio file (WAV, MP3, WebM, etc.).
        model_size: Whisper model size to use, defaults to 'base'.

    Returns:
        Dict with keys:
            - 'text': Transcribed text string
            - 'language': Detected language code (e.g. 'en', 'ar')
            - 'detected_language': Same detected language code
    """
    if not audio_bytes:
        return {"text": "", "language": "en", "detected_language": "en"}

    # Use mkstemp and immediately close the descriptor after writing to avoid Windows file lock
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(audio_bytes)

        model = get_whisper_model(model_size=model_size)
        logger.info("Transcribing audio file (%d bytes) with Whisper '%s'...", len(audio_bytes), model_size)

        result = model.transcribe(temp_path, language=None)
        transcribed_text = str(result.get("text", "")).strip()
        detected_lang = str(result.get("language", "en"))

        logger.info("Transcription completed (lang: %s): '%s'", detected_lang, transcribed_text[:50])
        return {
            "text": transcribed_text,
            "language": detected_lang,
            "detected_language": detected_lang,
        }
    except Exception as exc:
        logger.exception("Audio transcription failed: %s", exc)
        return {
            "text": "",
            "language": "en",
            "detected_language": "en",
            "error": str(exc),
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as rm_exc:
                logger.warning("Failed to remove temp audio file '%s': %s", temp_path, rm_exc)
