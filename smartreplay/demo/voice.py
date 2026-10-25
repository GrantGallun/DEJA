"""Local transcription for bounded, explicitly recorded voice questions."""
from __future__ import annotations

import os
import ctypes
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
    if gpu and os.name == "nt":
        try:
            ctypes.WinDLL("cublas64_12.dll")
        except OSError:
            gpu = False  # GPU detection alone does not mean its inference DLLs exist.
    if not gpu:
        return _load_cpu_model()
    try:
        return WhisperModel(name, device="cuda",
                            compute_type="float16",
                            local_files_only=True, cpu_threads=4)
    except (RuntimeError, ValueError, OSError):
        return _load_cpu_model()


def _load_cpu_model():
    from faster_whisper import WhisperModel
    # Short English commands must stay interactive while the game is running.
    # The large GPU model can take over a minute on the CPU on demo hardware.
    return WhisperModel(os.environ.get("SMARTREPLAY_VOICE_CPU_MODEL", "base.en"),
                        device="cpu", compute_type="int8", local_files_only=True, cpu_threads=4)


class VoiceTranscriber:
    """Reuse one local model; no microphone access and no implicit downloads."""
    def __init__(self, model_loader=None, audio_decoder=None, fallback_loader=None):
        self._loader = model_loader or _load_model
        self._decode = audio_decoder or _decode
        self._model = None
        self._lock = threading.Lock()
        self._fallback_loader = fallback_loader or _load_cpu_model
        self._cpu_fallback_used = False

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
            try:
                text = self._transcribe_text(audio)
            except (RuntimeError, OSError) as exc:
                # Whisper's iterator loads CUDA libraries lazily, after warmup.
                # Recover the same spoken question instead of requiring another take.
                if self._cpu_fallback_used or not any(word in str(exc).lower()
                        for word in ("cuda", "cublas", "cudnn", ".dll", "library")):
                    raise
                self._model = self._fallback_loader()
                self._cpu_fallback_used = True
                text = self._transcribe_text(audio)
        if not text:
            raise ValueError("I couldn't hear a question. Check your microphone and try again.")
        if len(text) > 800:
            raise ValueError("That question was too long. Try a shorter question.")
        return text

    def _transcribe_text(self, audio):
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
            without_timestamps=True, temperature=0)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
