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
