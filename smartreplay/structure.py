"""Find event HUD by recurring *structure* rather than recurring brightness.

The pixel channel (discovery.py) asks which pixels snap to one tight grey
value. That fails where the event's value is also common in the world: on a
dark map, a grey death-screen button sits on the shoulder of the huge black
peak the scenery produces, and by value alone the two cannot be told apart.

What the world never does is draw the same *pattern* at the same place again
and again. Text, button borders and icons have a fixed shape. So the frame is
cut into overlapping blocks, each block in each frame is reduced to a 64-bit
binary pattern (which pixels are brighter than the block's own mean), and a
block is event HUD when one pattern recurs in short, consistent runs.

The pattern is relative to the block's own mean, so it survives the things
that defeat the value test: a translucent tint over a changing background,
exposure changes, a hover highlight that brightens a button without moving
its edges. Flat, textureless blocks carry no pattern and are skipped.

The output has the same shape as the pixel channel's - an event mask and a
per-frame activation stack - so grouping and everything after it consume
either channel, or their union, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_W64 = (np.uint64(1) << np.arange(64, dtype=np.uint64))


@dataclass
class StructureParams:
    block: int = 8               # block side, in 2x-pooled pixels (16 analysis px)
    stride: int = 4              # block step, in pooled pixels (8 analysis px)
    min_texture: float = 6.0     # block std below this has no usable pattern
    max_hamming: int = 6         # bits a recurrence may differ by (codec noise)
    n_candidates: int = 3        # most frequent exact patterns tried per block
    min_exact: int = 4           # a candidate pattern must recur exactly this often
    min_occurrences: int = 3
    min_event_s: float = 1.0
    max_event_s: float = 30.0
    p_on_lo: float = 0.003
    p_on_hi: float = 0.60
    max_run_cv: float = 0.9


def _popcount(a: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    c = np.zeros(a.shape, np.uint8)
    for k in range(64):
        c += ((a >> np.uint64(k)) & np.uint64(1)).astype(np.uint8)
    return c


def _runs(on: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def block_hashes(gray: np.ndarray, p: StructureParams) -> tuple[np.ndarray, np.ndarray, list, list]:
    """(hashes (T,nY,nX) uint64, textured (T,nY,nX) bool, ys, xs) on the pooled grid."""
    t, h, w = gray.shape
    h2, w2 = h // 2, w // 2
    pooled = gray[:, :h2 * 2, :w2 * 2].reshape(t, h2, 2, w2, 2).mean(axis=(2, 4), dtype=np.float32)
    ys = list(range(0, h2 - p.block + 1, p.stride))
    xs = list(range(0, w2 - p.block + 1, p.stride))
    hashes = np.zeros((t, len(ys), len(xs)), np.uint64)
    textured = np.zeros((t, len(ys), len(xs)), bool)
    n = p.block * p.block
    for i, y in enumerate(ys):
        band = pooled[:, y:y + p.block, :]
        for j, x in enumerate(xs):
            patch = band[:, :, x:x + p.block].reshape(t, n)
            mean = patch.mean(axis=1, keepdims=True)
            bits = (patch > mean).astype(np.uint64)
            hashes[:, i, j] = (bits * _W64[:n]).sum(axis=1)
            textured[:, i, j] = patch.std(axis=1) >= p.min_texture
    return hashes, textured, ys, xs


def discover_structure(gray: np.ndarray, sample_fps: float,
                       params: StructureParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (event_mask (H,W) bool, on (T,H,W) bool) from recurring block patterns."""
    p = params or StructureParams()
    t, h, w = gray.shape
    hashes, textured, ys, xs = block_hashes(gray, p)
    min_run = max(2.0, p.min_event_s * sample_fps)
    max_run = p.max_event_s * sample_fps

    event = np.zeros((h, w), bool)
    on_px = np.zeros((t, h, w), bool)
    # Each block reports into its central cell (stride x stride pooled px), so
    # the cells tile the frame and overlapping blocks never overwrite each other.
    half = (p.block - p.stride) // 2
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            hv = hashes[:, i, j]
            tex = textured[:, i, j]
            if tex.sum() < p.min_exact * p.min_occurrences:
                continue
            u, c = np.unique(hv[tex], return_counts=True)
            best, best_cv = None, None
            for k in np.argsort(c)[::-1][:p.n_candidates]:
                if c[k] < p.min_exact:
                    break
                on = tex & (_popcount(hv ^ u[k]) <= p.max_hamming)
                runs = _runs(on)
                if runs.size < p.min_occurrences:
                    continue
                mean = runs.mean()
                cv = runs.std() / mean
                p_on = on.mean()
                if not (min_run <= mean <= max_run and p.p_on_lo <= p_on <= p.p_on_hi
                        and cv <= p.max_run_cv):
                    continue
                if best is None or cv < best_cv:
                    best, best_cv = on, cv
            if best is None:
                continue
            y0, x0 = (y + half) * 2, (x + half) * 2
            y1, x1 = y0 + p.stride * 2, x0 + p.stride * 2
            event[y0:y1, x0:x1] = True
            on_px[:, y0:y1, x0:x1] = best[:, None, None]
    return event, on_px
