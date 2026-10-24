"""Classify every pixel as world, static HUD, or event HUD.

The discriminating signal is temporal, not spatial. Over a few minutes of
play, a pixel showing 3D world takes a broad spread of values; a pixel inside
an always-on HUD element barely moves; and a pixel inside an *event* HUD
element snaps to one tight value and holds it for seconds, repeatedly.

Finding that tight value is done by histogram peak *prominence* rather than
argmax: the HUD value appears as a narrow spike sitting on top of the broad
base contributed by world content, and a spike is defined by standing above
its own neighbourhood, not by being the tallest bin overall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    label as cc_label,
    uniform_filter,
    uniform_filter1d,
)

N_BINS = 32
BIN_W = 256 // N_BINS


@dataclass
class DiscoveryParams:
    tol: int = 12                  # +/- grey levels counted as "at the mode"
    min_event_s: float = 1.0       # shortest run that counts as an event
    max_event_s: float = 30.0      # longer than this is a menu, not an event
    min_occurrences: int = 3       # a real event recurs
    p_on_lo: float = 0.005         # rare enough to be an event...
    p_on_hi: float = 0.60          # ...but with enough support to measure
    min_std: float = 5.0           # the pixel must actually vary over time
    static_p: float = 0.85         # always-on HUD: at its mode nearly always
    static_std: float = 8.0
    min_prominence: float = 0.010  # spike height as a fraction of all samples
    min_component_px: int = 24     # isolated specks are noise, not interface
    spatial_k: int = 5             # neighbourhood for the at-mode consensus
    spatial_vote: float = 0.60     # fraction of it that must agree
    close_gaps: int = 0            # bridging dropouts here rescues noise too;
    #                              the group activation is the safe place for it


@dataclass
class DiscoveryResult:
    mode_val: np.ndarray    # (H,W) uint8   the tight value each pixel snaps to
    p_on: np.ndarray        # (H,W) float32 fraction of samples at that value
    n_runs: np.ndarray      # (H,W) int32   contiguous visits to the mode
    mean_run: np.ndarray    # (H,W) float32 mean visit length, in samples
    std: np.ndarray         # (H,W) float32
    prominence: np.ndarray  # (H,W) float32
    event_mask: np.ndarray  # (H,W) bool
    static_mask: np.ndarray # (H,W) bool
    on: np.ndarray          # (T,H,W) bool  per-frame "at the mode" mask

    def summary(self) -> dict:
        return {
            "event_pixels": int(self.event_mask.sum()),
            "static_pixels": int(self.static_mask.sum()),
            "total_pixels": int(self.event_mask.size),
            "event_fraction": float(self.event_mask.mean()),
        }


def _pixel_histograms(gray: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Per-pixel intensity histograms, shape (N_BINS, P).

    Chunked over pixels so the intermediate index array stays small.
    """
    t, h, w = gray.shape
    flat = gray.reshape(t, h * w)
    p = h * w
    hist = np.empty((N_BINS, p), dtype=np.int32)
    for s in range(0, p, chunk):
        e = min(s + chunk, p)
        bins = (flat[:, s:e] // BIN_W).astype(np.int32)
        width = e - s
        idx = bins * width + np.arange(width, dtype=np.int32)
        counts = np.bincount(idx.ravel(), minlength=N_BINS * width)
        hist[:, s:e] = counts.reshape(N_BINS, width)
    return hist


def _spatial_consensus(on: np.ndarray, k: int, vote: float, chunk: int = 128) -> np.ndarray:
    """Keep activations where at least `vote` of a k-by-k neighbourhood agrees.

    Chunked over time so the float32 working copy stays bounded regardless of
    how long the footage is.
    """
    if k <= 1:
        return on
    t = on.shape[0]
    out = np.empty_like(on)
    for s in range(0, t, chunk):
        e = min(s + chunk, t)
        block = on[s:e].astype(np.float32)
        uniform_filter(block, size=(1, k, k), output=block, mode="nearest")
        out[s:e] = block >= vote
    return out


def _drop_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1 or not mask.any():
        return mask
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def discover(gray: np.ndarray, sample_fps: float, params: DiscoveryParams | None = None) -> DiscoveryResult:
    p = params or DiscoveryParams()
    t, h, w = gray.shape
    flat = gray.reshape(t, h * w)

    hist = _pixel_histograms(gray)
    base = uniform_filter1d(hist.astype(np.float32), size=9, axis=0, mode="nearest")
    prom = hist - base
    mode_bin = np.argmax(prom, axis=0)
    prominence = (prom[mode_bin, np.arange(h * w)] / float(t)).astype(np.float32)
    mode_val = (mode_bin * BIN_W + BIN_W // 2).astype(np.uint8)

    on = np.abs(flat.astype(np.int16) - mode_val[None, :].astype(np.int16)) <= p.tol

    # Spatial consensus, and the single most important step here. A pixel
    # sitting at its mode value proves nothing on its own: a dark HUD panel and
    # a dark patch of scenery look identical by value. What separates them is
    # that an interface element takes its colour across a whole contiguous
    # region simultaneously, while world content never agrees with itself at
    # that scale. Requiring the neighbourhood to agree removes nearly all of
    # the spurious activations that value alone admits.
    on = _spatial_consensus(on.reshape(t, h, w), p.spatial_k, p.spatial_vote).reshape(t, h * w)

    # Bridging dropouts per pixel is tempting but counter-productive: random
    # world pixels have many one-sample gaps, so closing merges their scattered
    # hits into runs long enough to pass the event test. Left off by default;
    # grouping closes the consensus activation instead, where it is safe.
    if p.close_gaps > 0:
        on = binary_closing(on, structure=np.ones((2 * p.close_gaps + 1, 1), dtype=bool))

    n_on = on.sum(axis=0).astype(np.int32)
    rising = on[1:] & ~on[:-1]
    n_runs = (rising.sum(axis=0) + on[0].astype(np.int32)).astype(np.int32)
    mean_run = (n_on / np.maximum(n_runs, 1)).astype(np.float32)
    p_on = (n_on / float(t)).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)

    min_run = max(2.0, p.min_event_s * sample_fps)
    max_run = p.max_event_s * sample_fps

    event = (
        (n_runs >= p.min_occurrences)
        & (mean_run >= min_run)
        & (mean_run <= max_run)
        & (p_on >= p.p_on_lo)
        & (p_on <= p.p_on_hi)
        & (std >= p.min_std)
        & (prominence >= p.min_prominence)
    )
    static = (p_on >= p.static_p) & (std <= p.static_std)

    shape = (h, w)
    # Interface is contiguous; scattered single pixels that happen to pass the
    # temporal test are noise, and dropping them here keeps grouping tractable.
    event = _drop_small_components(event.reshape(shape), p.min_component_px).reshape(-1)
    return DiscoveryResult(
        mode_val=mode_val.reshape(shape),
        p_on=p_on.reshape(shape),
        n_runs=n_runs.reshape(shape),
        mean_run=mean_run.reshape(shape),
        std=std.reshape(shape),
        prominence=prominence.reshape(shape),
        event_mask=event.reshape(shape),
        static_mask=static.reshape(shape),
        on=on.reshape(t, h, w),
    )


def save(result: DiscoveryResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "discovery.npz",
        mode_val=result.mode_val,
        p_on=result.p_on,
        n_runs=result.n_runs,
        mean_run=result.mean_run,
        std=result.std,
        prominence=result.prominence,
        event_mask=result.event_mask,
        static_mask=result.static_mask,
    )
    (out_dir / "discovery_summary.json").write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
