"""Local transcription for bounded, explicitly recorded voice questions."""
from __future__ import annotations

import os
from pathlib import Path
import threading

import numpy as np


def _decode(path):
    from faster_whisper.audio import decode_audio
    return decode_audio(str(path), sampling_rate=16000)


def _load_model():
    from faster_whisper import WhisperModel
    import ctranslate2
    from smartreplay.speech import _cuda_dlls

    _cuda_dlls()
    name = os.environ.get("SMARTREPLAY_VOICE_MODEL", "large-v3-turbo")
    gpu = ctranslate2.get_cuda_device_count() > 0
    try:
        return WhisperModel(name, device="cuda" if gpu else "cpu",
                            compute_type="float16" if gpu else "int8",
                            local_files_only=True, cpu_threads=4)
    except (RuntimeError, ValueError, OSError):
        if not gpu:
            raise
        return WhisperModel(name, device="cpu", compute_type="int8",
                            local_files_only=True, cpu_threads=4)


class VoiceTranscriber:
    """Reuse one local model; no microphone access and no implicit downloads."""
    def __init__(self, model_loader=None, audio_decoder=None):
        self._loader = model_loader or _load_model
        self._decode = audio_decoder or _decode
        self._model = None
        self._lock = threading.Lock()

    def warmup(self):
        with self._lock:
            if self._model is None:
                self._model = self._loader()

    def transcribe(self, path: Path) -> str:
        audio = np.asarray(self._decode(path), dtype=np.float32)
        if audio.ndim != 1 or not 4000 <= audio.size <= 16000 * 15.1:
            raise ValueError("Hold your voice shortcut while speaking a question of up to 15 seconds.")
        if not np.isfinite(audio).all() or np.sqrt(np.mean(audio.astype(np.float64) ** 2)) < .001:
            raise ValueError("I couldn't hear a question. Check your microphone and try again.")
        with self._lock:
            if self._model is None:
                self._model = self._loader()
            segments, _ = self._model.transcribe(audio, language="en", beam_size=1,
                vad_filter=True, condition_on_previous_text=False,
                without_timestamps=True, temperature=0)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not text:
            raise ValueError("I couldn't hear a question. Check your microphone and try again.")
        if len(text) > 800:
            raise ValueError("That question was too long. Try a shorter question.")
        return text
