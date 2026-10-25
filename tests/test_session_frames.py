"""Real MP4 timestamp sampling, including native-style fractional boundaries."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from smartreplay.demo.session_search import _frames


def ramp_video(path, *, frames=20, fps=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    try:
        for number in range(frames):
            writer.write(np.full((48, 64, 3), number * 10, np.uint8))
    finally:
        writer.release()
    return path.resolve()


def shifted_segments(tmp_path):
    # Native manifests accumulate measured segment durations rather than exact
    # integer boundaries. A valid target at 10s lies in the last displayed frame.
    segments = []
    for number, (start, end) in enumerate(((0, 5.01), (5.01, 10.01), (10.01, 15.01))):
        path = ramp_video(tmp_path / f"segment-{number}.mp4")
        segments.append({"path": str(path), "start_s": start, "end_s": end})
    return {"id": "fractional", "duration_s": 15.01, "state": "closed", "segments": segments}


def test_timestamp_seek_rounding_to_eof_uses_the_real_last_frame_timestamp(tmp_path):
    session = shifted_segments(tmp_path)
    second = session["segments"][1]
    capture = cv2.VideoCapture(second["path"])
    try:
        assert capture.set(cv2.CAP_PROP_POS_MSEC, 4990)
        assert not capture.read()[0], "Fixture must exercise OpenCV's near-EOF seek failure"
    finally:
        capture.release()
    frames = _frames(session, 0, 15.01)
    # The last frame was captured at 4.75s and is displayed until 5s. It can
    # represent the request, but must never be relabelled as a frame at 10s.
    timestamp, image = next((timestamp, image) for timestamp, image in frames if 9.5 < timestamp < 10.01)
    assert timestamp == pytest.approx(9.76)
    assert image.mean() == pytest.approx(190, abs=6)
    assert len(frames) <= 14


def test_all_sample_labels_use_decoded_pts_instead_of_requested_seek_time(tmp_path):
    session = shifted_segments(tmp_path)
    frames = _frames(session, 5.11, 9.1)
    # Every source PTS is a multiple of 0.25s, offset by the 5.01s segment origin.
    for timestamp, image in frames:
        pts = timestamp + 5.11 - 5.01
        assert pts * 4 == pytest.approx(round(pts * 4), abs=.004)
        assert 0 <= timestamp < 9.1 - 5.11
        assert image.mean() == pytest.approx(round(pts * 4) * 10, abs=6)


@pytest.mark.parametrize("broken", ["missing", "corrupt", "truncated-timeline"])
def test_bad_segment_never_silently_uses_a_frame_from_an_earlier_time(tmp_path, broken):
    session = shifted_segments(tmp_path)
    path = Path(session["segments"][1]["path"])
    if broken == "missing": path.unlink()
    elif broken == "corrupt": path.write_bytes(b"broken MP4")
    else: ramp_video(path, frames=4)
    with pytest.raises(ValueError, match="cannot be decoded|missing|cover"):
        _frames(session, 0, 15.01)


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_unreadable_short_segment_between_grid_samples_still_blocks_coverage(tmp_path, broken):
    first = ramp_video(tmp_path / "first.mp4", frames=4)
    missing = tmp_path / "missing.mp4"
    if broken == "corrupt": missing.write_bytes(b"broken MP4")
    third = ramp_video(tmp_path / "third.mp4", frames=8)
    session = {"id": "gap", "duration_s": 3, "state": "closed", "segments": [
        {"path": str(first), "start_s": 0, "end_s": 1},
        {"path": str(missing.resolve()), "start_s": 1, "end_s": 1.1},
        {"path": str(third), "start_s": 1.1, "end_s": 3},
    ]}
    with pytest.raises(ValueError, match="cannot be decoded|missing|cover"):
        _frames(session, 0, 3)
