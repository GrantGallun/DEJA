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
    n_candidates: int = 3          # histogram peaks per pixel tested as the event state
    max_run_cv: float = 1.0        # run lengths of an event state are consistent
    structure: bool = True         # also run the recurring-pattern channel (structure.py)


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


def _on_series(gray: np.ndarray, mode_val: np.ndarray, tol: int, k: int, vote: float,
               chunk: int = 128) -> np.ndarray:
    """Per-frame "at the candidate value" mask with spatial consensus, (T, H*W).

    Computed in time chunks so no full-length int16 or float copy of the
    stack is ever held - on half an hour of footage those alone run to
    gigabytes.
    """
    t, h, w = gray.shape
    mv = mode_val.reshape(h, w).astype(np.int16)
    out = np.empty((t, h * w), dtype=bool)
    for s in range(0, t, chunk):
        e = min(s + chunk, t)
        block = np.abs(gray[s:e].astype(np.int16) - mv[None]) <= tol
        if k > 1:
            f = block.astype(np.float32)
            uniform_filter(f, size=(1, k, k), output=f, mode="nearest")
            block = f >= vote
        out[s:e] = block.reshape(e - s, h * w)
    return out


def _run_stats(on: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n_runs, mean_run, p_on, run_cv) per column of a (T, P) bool array.

    run_cv is the coefficient of variation of the run lengths. An event
    overlay holds for about the same time on every appearance; a patch of
    world that merely passes through the same value does so for wildly
    varying spans.
    """
    t, p = on.shape
    cur = np.zeros(p, dtype=np.int32)
    s1 = np.zeros(p, dtype=np.float64)
    s2 = np.zeros(p, dtype=np.float64)
    n_runs = np.zeros(p, dtype=np.int32)
    for i in range(t):
        o = on[i]
        ended = (cur > 0) & ~o
        if ended.any():
            c = cur[ended].astype(np.float64)
            s1[ended] += c
            s2[ended] += c * c
            n_runs[ended] += 1
        cur = np.where(o, cur + 1, 0)
    ended = cur > 0
    c = cur[ended].astype(np.float64)
    s1[ended] += c
    s2[ended] += c * c
    n_runs[ended] += 1
    n = np.maximum(n_runs, 1)
    mean_run = s1 / n
    var = np.maximum(s2 / n - mean_run ** 2, 0.0)
    cv = np.where(mean_run > 0, np.sqrt(var) / np.maximum(mean_run, 1e-6), 0.0)
    p_on = s1 / float(t)
    return n_runs, mean_run.astype(np.float32), p_on.astype(np.float32), cv.astype(np.float32)


def _std(flat: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """Per-pixel std without a float64 copy of the whole stack."""
    out = np.empty(flat.shape[1], dtype=np.float32)
    for s in range(0, flat.shape[1], chunk):
        out[s:s + chunk] = flat[:, s:s + chunk].astype(np.float32).std(axis=0)
    return out


def _candidate_bins(prom: np.ndarray, n: int, min_count: float, sep: int = 2) -> np.ndarray:
    """Up to n most prominent histogram peaks per pixel, (n, P); -1 = none.

    One peak per pixel is not enough. A pixel shows world most of the time,
    and in a dark game the world's own dominant value - black - is the
    tallest spike, so the event element's value (a grey button on a death
    screen) is only the second or third peak. Each candidate is tested as a
    possible event state and the best one kept.
    """
    work = prom.astype(np.float32).copy()
    nb, p = work.shape
    cols = np.arange(p)
    out = np.full((n, p), -1, dtype=np.int32)
    for i in range(n):
        b = np.argmax(work, axis=0)
        ok = work[b, cols] >= min_count
        out[i, ok] = b[ok]
        lo, hi = np.maximum(b - sep, 0), np.minimum(b + sep, nb - 1)
        for d in range(-sep, sep + 1):
            idx = np.clip(b + d, lo, hi)
            work[idx, cols] = -np.inf
    return out


def discover(gray: np.ndarray, sample_fps: float, params: DiscoveryParams | None = None) -> DiscoveryResult:
    p = params or DiscoveryParams()
    t, h, w = gray.shape
    P = h * w
    flat = gray.reshape(t, P)
    cols = np.arange(P)

    hist = _pixel_histograms(gray)
    base = uniform_filter1d(hist.astype(np.float32), size=9, axis=0, mode="nearest")
    prom = hist - base
    std = _std(flat)

    min_run = max(2.0, p.min_event_s * sample_fps)
    max_run = p.max_event_s * sample_fps
    cands = _candidate_bins(prom, max(1, p.n_candidates), p.min_prominence * t)

    # The most prominent value doubles as the always-on HUD test.
    top = np.where(cands[0] >= 0, cands[0], np.argmax(prom, axis=0))
    mode_val = (top * BIN_W + BIN_W // 2).astype(np.uint8)
    prominence = (prom[top, cols] / float(t)).astype(np.float32)

    best_score = np.full(P, -1.0, dtype=np.float32)
    on = np.zeros((t, P), dtype=bool)
    sel_mode = mode_val.copy()
    sel_prom = prominence.copy()
    n_runs = np.zeros(P, np.int32)
    mean_run = np.zeros(P, np.float32)
    p_on = np.zeros(P, np.float32)
    static = np.zeros(P, dtype=bool)

    for i in range(cands.shape[0]):
        b = cands[i]
        valid = b >= 0
        if not valid.any():
            break
        bi = np.where(valid, b, 0)
        mv = (bi * BIN_W + BIN_W // 2).astype(np.uint8)
        pr = np.where(valid, prom[bi, cols] / float(t), 0.0).astype(np.float32)
        # Spatial consensus, and the single most important step here. A pixel
        # sitting at a value proves nothing on its own: a dark HUD panel and
        # a dark patch of scenery look identical by value. What separates them
        # is that an interface element takes its colour across a whole
        # contiguous region simultaneously, while world content never agrees
        # with itself at that scale.
        on_i = _on_series(gray, mv, p.tol, p.spatial_k, p.spatial_vote)
        if p.close_gaps > 0:
            on_i = binary_closing(on_i, structure=np.ones((2 * p.close_gaps + 1, 1), dtype=bool))
        nr, mr, po, cv = _run_stats(on_i)
        if i == 0:
            static = valid & (po >= p.static_p) & (std <= p.static_std)
            # Pixels with no event candidate still report their top value.
            n_runs, mean_run, p_on = nr, mr, po
        passes = (
            valid
            & (nr >= p.min_occurrences)
            & (mr >= min_run)
            & (mr <= max_run)
            & (po >= p.p_on_lo)
            & (po <= p.p_on_hi)
            & (std >= p.min_std)
            & (pr >= p.min_prominence)
            & (cv <= p.max_run_cv)
        )
        # Among passing candidates prefer the most consistent run lengths:
        # the element's real "on" state holds for the same time on every
        # appearance, while a value the world merely passes through (black,
        # in a dark game) comes and goes irregularly - and is often the more
        # prominent of the two.
        score = (1.0 / (1.0 + cv)).astype(np.float32)
        better = passes & (score > best_score)
        if better.any():
            best_score[better] = score[better]
            on[:, better] = on_i[:, better]
            sel_mode[better] = mv[better]
            sel_prom[better] = pr[better]
            n_runs[better], mean_run[better], p_on[better] = nr[better], mr[better], po[better]
        del on_i

    event = best_score >= 0
    # Pixels without an event candidate keep their top-value activation, so
    # downstream code still sees the static HUD and a full "on" stack.
    if (~event).any() and cands.shape[0]:
        rest = ~event
        on_top = _on_series(gray, mode_val, p.tol, p.spatial_k, p.spatial_vote)
        on[:, rest] = on_top[:, rest]
        del on_top
    sel_mode[~event] = mode_val[~event]
    sel_prom[~event] = prominence[~event]

    # The structure channel catches elements whose value the world shares
    # (see structure.py). Where both channels fire, the pixel channel's
    # finer-grained activation is kept.
    if p.structure:
        from .structure import StructureParams, discover_structure

        s_event, s_on = discover_structure(gray, sample_fps, StructureParams(
            min_occurrences=p.min_occurrences, min_event_s=p.min_event_s,
            max_event_s=p.max_event_s))
        add = s_event.reshape(-1) & ~event
        if add.any():
            on[:, add] = s_on.reshape(t, P)[:, add]
            event = event | add
        del s_on

    shape = (h, w)
    # Interface is contiguous; scattered single pixels that happen to pass the
    # temporal test are noise, and dropping them here keeps grouping tractable.
    event = _drop_small_components(event.reshape(shape), p.min_component_px).reshape(-1)
    return DiscoveryResult(
        mode_val=sel_mode.reshape(shape),
        p_on=p_on.reshape(shape),
        n_runs=n_runs.reshape(shape),
        mean_run=mean_run.reshape(shape),
        std=std.reshape(shape),
        prominence=sel_prom.reshape(shape),
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
