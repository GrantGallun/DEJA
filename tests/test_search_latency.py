"""Avoid repeat decoding while preserving the frame's recorded time."""
import cv2
import numpy as np
import pytest

from smartreplay.demo.search import sample_frames


class VariableCapture:
    times = [0, .08, .21, .55, .97, 1.23, 1.6]

    def __init__(self):
        self.index = -1
        self.seeks = []
        self.released = False

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_MSEC
        self.seeks.append(value)
        nearest = min(range(len(self.times)), key=lambda i: abs(self.times[i] * 1000 - value))
        self.index = nearest - 1
        return True

    def grab(self):
        self.index += 1
        return self.index < len(self.times)

    def retrieve(self):
        return True, np.full((8, 8, 3), self.index, np.uint8)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def get(self, key):
        assert key == cv2.CAP_PROP_POS_MSEC
        return self.times[self.index] * 1000

    def release(self):
        self.released = True


def test_dense_sampling_decodes_forward_once_and_reports_actual_variable_times(monkeypatch):
    capture = VariableCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = sample_frames("recording.mp4", [0, .1, .3, .8, 1.4])
    assert [timestamp for timestamp, _ in frames] == pytest.approx([0, .21, .55, .97, 1.6])
    assert [int(frame[0, 0, 0]) for _, frame in frames] == [0, 2, 3, 4, 6]
    assert len(capture.seeks) <= 1
    assert capture.released


def test_same_recorded_frame_is_not_uploaded_repeatedly_for_sparse_video(monkeypatch):
    capture = VariableCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = sample_frames("recording.mp4", [.1, .12, .15])
    assert [timestamp for timestamp, _ in frames] == pytest.approx([.21])
    assert capture.released
