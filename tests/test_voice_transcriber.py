from types import SimpleNamespace

import numpy as np
import pytest

from smartreplay.demo.voice import VoiceTranscriber


def test_warmup_loads_once_without_capturing_or_decoding_audio():
    loads = []
    def forbidden(_):
        pytest.fail("Warmup must not capture or decode audio")
    transcriber = VoiceTranscriber(model_loader=lambda: loads.append(True) or object(),
                                  audio_decoder=forbidden)
    assert loads == []
    transcriber.warmup()
    transcriber.warmup()
    assert loads == [True]


def test_voice_model_is_reused_and_text_is_joined(tmp_path):
    calls, loads = [], []

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text=" Show the last"),
                         SimpleNamespace(text=" elimination. ")]), None

    transcriber = VoiceTranscriber(model_loader=lambda: loads.append(True) or Model(),
        audio_decoder=lambda path: np.full(16000, .03, dtype=np.float32))
    for name in ("one.wav", "two.wav"):
        assert transcriber.transcribe(tmp_path / name) == "Show the last elimination."
    assert len(loads) == 1
    assert len(calls) == 2
    assert calls[0]["vad_filter"] is True
    assert calls[0]["condition_on_previous_text"] is False


@pytest.mark.parametrize("audio", [np.zeros(16000), np.ones(100),
    np.full(16000, float("nan")), np.ones(16000 * 16)])
def test_invalid_or_silent_audio_never_loads_model(audio, tmp_path):
    def forbidden():
        pytest.fail("Invalid audio must not load a speech model")
    transcriber = VoiceTranscriber(model_loader=forbidden, audio_decoder=lambda _: audio)
    with pytest.raises(ValueError):
        transcriber.transcribe(tmp_path / "voice.wav")


@pytest.mark.parametrize("text", ["", "  ", "x" * 801])
def test_empty_or_overlong_transcripts_do_not_become_questions(text, tmp_path):
    model = SimpleNamespace(transcribe=lambda *args, **kwargs:
        (iter([SimpleNamespace(text=text)]), None))
    transcriber = VoiceTranscriber(model_loader=lambda: model,
        audio_decoder=lambda _: np.full(16000, .03))
    with pytest.raises(ValueError):
        transcriber.transcribe(tmp_path / "voice.wav")


def test_failed_model_load_can_be_retried(tmp_path):
    attempts = []
    def load():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return SimpleNamespace(transcribe=lambda *args, **kwargs:
            (iter([SimpleNamespace(text="Show the last fall")]), None))
    transcriber = VoiceTranscriber(model_loader=load,
        audio_decoder=lambda _: np.full(16000, .03))
    with pytest.raises(RuntimeError, match="model unavailable"):
        transcriber.transcribe(tmp_path / "voice.wav")
    assert transcriber.transcribe(tmp_path / "voice.wav") == "Show the last fall"


def test_missing_gpu_library_during_lazy_decode_falls_back_and_reuses_cpu(tmp_path):
    loads = []
    class BrokenGPU:
        def transcribe(self, audio, **kwargs):
            def segments():
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield
            return segments(), None
    cpu = SimpleNamespace(transcribe=lambda *a, **kw:
        (iter([SimpleNamespace(text="Show yellow falling")]), None))
    transcriber = VoiceTranscriber(model_loader=BrokenGPU,
        audio_decoder=lambda _: np.full(16000, .03),
        fallback_loader=lambda: loads.append(True) or cpu)
    assert transcriber.transcribe(tmp_path / "one.wav") == "Show yellow falling"
    assert transcriber.transcribe(tmp_path / "two.wav") == "Show yellow falling"
    assert loads == [True]


def test_cpu_voice_uses_small_local_model_instead_of_large_gpu_model(monkeypatch):
    import sys
    from smartreplay.demo import voice
    import smartreplay.speech
    calls = []
    monkeypatch.delenv("SMARTREPLAY_VOICE_MODEL", raising=False)
    monkeypatch.delenv("SMARTREPLAY_VOICE_CPU_MODEL", raising=False)
    monkeypatch.setattr(smartreplay.speech, "_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=lambda name, **kw: calls.append((name, kw)) or object()))
    voice._load_model()
    voice._load_cpu_model()
    assert [name for name, _ in calls] == ["base.en", "base.en"]
    assert all(kw["device"] == "cpu" and kw["local_files_only"] for _, kw in calls)


def test_cpu_voice_model_override_is_separate_from_gpu_override(monkeypatch):
    import sys
    from smartreplay.demo import voice
    calls = []
    monkeypatch.setenv("SMARTREPLAY_VOICE_MODEL", "large-v3-turbo")
    monkeypatch.setenv("SMARTREPLAY_VOICE_CPU_MODEL", "small.en")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=lambda name, **kw: calls.append(name) or object()))
    voice._load_cpu_model()
    assert calls == ["small.en"]
