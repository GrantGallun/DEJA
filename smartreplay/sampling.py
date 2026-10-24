"""Decode a video into a low-resolution frame stack for temporal analysis.

Discovery only ever looks at a few frames per second at thumbnail resolution,
so decoding is done with a grab/retrieve stride: cv2.grab() advances the
decoder without paying for colour conversion or a frame copy, and retrieve()
is called only on the frames we actually keep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SampleMeta:
    source: str
    native_fps: float
    native_size: tuple[int, int]      # (w, h) after letterbox crop
    crop: tuple[int, int, int, int]   # (x0, y0, x1, y1) applied to native frames
    sample_fps: float
    size: tuple[int, int]             # (w, h) of the analysis stack
    color_size: tuple[int, int]       # (w, h) of the colour stack kept for reports
    n_frames: int
    duration_s: float
    truncated: bool
    frame_size: tuple[int, int] = (0, 0)   # (w, h) of the source before the crop
    fingerprint: dict | None = None        # identity of the source file
    params: dict | None = None             # sampling settings that produced the stack

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def source_fingerprint(path: str | Path) -> dict:
    """Identity of a source video: resolved path, size and modification time.

    Two inputs with the same stem (match.avi, match.mp4) or a file replaced
    in place both change this, so a cached sample stack can be checked
    against the video it is about to stand in for.
    """
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def sampling_params(sample_fps: float, width: int, color_width: int, max_seconds: float,
                    trim_start: float, trim_end: float) -> dict:
    return {"sample_fps": float(sample_fps), "width": int(width), "color_width": int(color_width),
            "max_seconds": float(max_seconds), "trim_start": float(trim_start),
            "trim_end": float(trim_end)}


def cache_matches(out_dir: str | Path, path: str | Path, params: dict) -> tuple[bool, str]:
    """Whether the stack in out_dir was sampled from this video with these settings."""
    meta_path = Path(out_dir) / "meta.json"
    if not meta_path.exists():
        return False, "no cached samples"
    for name in ("gray.npy", "color.npy", "times.npy"):
        if not (Path(out_dir) / name).exists():
            return False, f"cache incomplete ({name} missing)"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "cached meta.json unreadable"
    cached_fp = meta.get("fingerprint")
    if not cached_fp:
        return False, "cache predates source fingerprints"
    current = source_fingerprint(path)
    if cached_fp != current:
        return False, f"cache was sampled from {cached_fp.get('path')}, not this file"
    if meta.get("params") != params:
        return False, "cache was sampled with different settings"
    return True, "cache matches source and settings"


def _detect_letterbox(probes: list[np.ndarray], thresh: int = 16) -> tuple[int, int, int, int]:
    """Find black bars by taking the per-pixel max across probe frames.

    A row that is dark in *every* probe is a bar; a row that is dark only in
    some frames is just a dark scene, which the max collapses away.
    """
    peak = np.maximum.reduce(probes)
    h, w = peak.shape
    row_max = peak.max(axis=1)
    col_max = peak.max(axis=0)

    y0 = 0
    while y0 < h // 3 and row_max[y0] < thresh:
        y0 += 1
    y1 = h
    while y1 > 2 * h // 3 and row_max[y1 - 1] < thresh:
        y1 -= 1
    x0 = 0
    while x0 < w // 3 and col_max[x0] < thresh:
        x0 += 1
    x1 = w
    while x1 > 2 * w // 3 and col_max[x1 - 1] < thresh:
        x1 -= 1

    # Degenerate crops mean the heuristic misfired; fall back to the full frame.
    if x1 - x0 < w // 2 or y1 - y0 < h // 2:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _probe_frames(path: str, n: int = 24) -> tuple[list[np.ndarray], float, int, tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    probes: list[np.ndarray] = []
    if count > 0:
        for i in np.linspace(0, max(count - 1, 0), num=min(n, max(count, 1)), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                probes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if not probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            probes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not probes:
        raise RuntimeError(f"no decodable frames in {path}")
    return probes, float(fps), count, (w, h)


def sample_video(
    path: str | Path,
    out_dir: str | Path,
    sample_fps: float = 2.0,
    width: int = 384,
    color_width: int = 192,
    max_seconds: float = 600.0,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
) -> SampleMeta:
    """Decode `path` into gray/colour stacks under `out_dir`.

    Writes gray.npy (T,H,W uint8), color.npy (T,H,W,3 uint8 BGR),
    times.npy (T float64 seconds) and meta.json.
    """
    path = str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = source_fingerprint(path)
    params = sampling_params(sample_fps, width, color_width, max_seconds, trim_start, trim_end)
    probes, fps, count, native = _probe_frames(path)
    x0, y0, x1, y1 = _detect_letterbox(probes)
    cw, ch = x1 - x0, y1 - y0

    # Keep the real aspect ratio; forcing 16:9 would distort 4:3 footage.
    height = max(2, int(round(width * ch / cw / 2) * 2))
    # The colour stack only backs report imagery, so it is kept small; exemplar
    # crops are re-read from the source at full resolution instead.
    color_width = min(color_width, width)
    color_height = max(2, int(round(color_width * ch / cw / 2) * 2))

    stride = max(1, int(round(fps / sample_fps)))
    eff_fps = fps / stride

    start_frame = int(trim_start * fps)
    end_frame = count - int(trim_end * fps) if count > 0 and trim_end > 0 else None
    budget = int(max_seconds * eff_fps)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    grays: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    times: list[float] = []
    idx = 0
    truncated = False

    while True:
        if not cap.grab():
            break
        if end_frame is not None and idx >= end_frame:
            break
        if idx >= start_frame and (idx - start_frame) % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                idx += 1
                continue
            crop = frame[y0:y1, x0:x1]
            small = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            colors.append(cv2.resize(crop, (color_width, color_height), interpolation=cv2.INTER_AREA))
            times.append(idx / fps)
            if len(grays) >= budget:
                truncated = True
                break
        idx += 1
    cap.release()

    if len(grays) < 16:
        raise RuntimeError(f"only {len(grays)} samples from {path}; need a longer clip")

    gray = np.stack(grays)
    color = np.stack(colors)
    t = np.asarray(times, dtype=np.float64)

    np.save(out_dir / "gray.npy", gray)
    np.save(out_dir / "color.npy", color)
    np.save(out_dir / "times.npy", t)

    meta = SampleMeta(
        source=path,
        native_fps=float(fps),
        native_size=(cw, ch),
        crop=(x0, y0, x1, y1),
        sample_fps=float(eff_fps),
        size=(width, height),
        color_size=(color_width, color_height),
        n_frames=int(gray.shape[0]),
        duration_s=float(t[-1] - t[0]),
        truncated=truncated,
        frame_size=(int(native[0]), int(native[1])),
        fingerprint=fingerprint,
        params=params,
    )
    (out_dir / "meta.json").write_text(meta.to_json(), encoding="utf-8")
    return meta


def load_samples(out_dir: str | Path):
    out_dir = Path(out_dir)
    gray = np.load(out_dir / "gray.npy")
    color = np.load(out_dir / "color.npy")
    times = np.load(out_dir / "times.npy")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return gray, color, times, meta
