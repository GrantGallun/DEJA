"""Group event pixels into HUD elements by temporal co-activation.

Spatial adjacency is the wrong primitive here: a kill feed and a round timer
can sit side by side, and a single element can be split across disjoint
regions. What identifies an element is that all of its pixels light up at the
same moments. So pixels are clustered by the similarity of their activation
time series, and spatial connectivity is only used afterwards to draw boxes.

The activation series of a group is, directly, the event timeline for it:
rising edges are onsets and run lengths are durations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label


@dataclass
class GroupParams:
    jaccard: float = 0.65        # time-series similarity required for membership
    min_pixels: int = 24         # absolute floor on element size
    min_pixel_fraction: float = 0.0012  # and a floor relative to frame area
    min_coherence: float = 0.70  # mean member agreement with the consensus
    min_fill: float = 0.15       # group pixels / bounding box area
    max_duration_cv: float = 0.75  # a real event has a consistent duration
    min_occurrences: int = 3
    min_event_s: float = 1.0
    max_groups: int = 24         # confirmed groups reported, applied after validation
    max_iterations: int = 400    # blobs examined before giving up (the work budget)
    merge_jaccard: float = 0.80  # fold near-duplicate groups together
    containment: float = 0.85    # subset test for a fragment of one element
    adjacency_px: int = 3        # gap still counted as spatially adjacent
    close_gaps: int = 1          # samples of dropout bridged in an activation
    vote: float = 0.5            # fraction of member pixels on => group is on
    explain: list | None = None  # if a list, every rejected candidate is logged into it


@dataclass
class Occurrence:
    start_idx: int
    end_idx: int        # exclusive
    start_s: float
    duration_s: float


@dataclass
class EventGroup:
    gid: int
    n_pixels: int
    roi: tuple[int, int, int, int]            # (x0, y0, x1, y1) in analysis pixels
    roi_norm: tuple[float, float, float, float]
    boxes: list[tuple[int, int, int, int]]    # connected components in the group
    occurrences: list[Occurrence]
    mean_duration_s: float
    median_gap_s: float
    activation: np.ndarray = field(repr=False)  # (T,) bool
    mask: np.ndarray = field(repr=False)        # (H,W) bool

    def to_dict(self) -> dict:
        return {
            "gid": self.gid,
            "n_pixels": self.n_pixels,
            "roi": list(self.roi),
            "roi_norm": [round(v, 5) for v in self.roi_norm],
            "boxes": [list(b) for b in self.boxes],
            "n_occurrences": len(self.occurrences),
            "mean_duration_s": round(self.mean_duration_s, 3),
            "median_gap_s": round(self.median_gap_s, 3),
            "occurrences": [asdict(o) for o in self.occurrences],
        }


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of a 1-D boolean array as (start, end_exclusive)."""
    if not active.any():
        return []
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _close1d(active: np.ndarray, k: int) -> np.ndarray:
    """Bridge gaps of up to k samples in a 1-D boolean series.

    Safe to apply to a consensus activation (a codec dropout on a flat panel
    loses the odd sample), unlike applying it per pixel, where the many short
    gaps in world content get merged into runs long enough to look like events.
    """
    if k <= 0 or not active.any():
        return active
    out = active.copy()
    on_idx = np.flatnonzero(out)
    first, last = int(on_idx[0]), int(on_idx[-1])
    i = first
    while i <= last:
        if out[i]:
            i += 1
            continue
        j = i
        while j <= last and not out[j]:
            j += 1
        if (j - i) <= k:
            out[i:j] = True
        i = j
    return out


def _span_duration(times: np.ndarray, s: int, e: int, dt: float) -> float:
    """Duration of samples s..e-1: each sample stands for one interval of dt.

    Measured from the last sample in the run rather than the first sample
    after it, so an occurrence that reaches the end of the footage is not
    shortened by one interval.
    """
    return float(times[e - 1] - times[s] + dt)


def _bbox(ys: np.ndarray, xs: np.ndarray, members: np.ndarray) -> tuple[int, int, int, int]:
    return (int(xs[members].min()), int(ys[members].min()),
            int(xs[members].max()) + 1, int(ys[members].max()) + 1)


def _near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0]
                or a[3] + gap < b[1] or b[3] + gap < a[1])


def _jaccard_vs(series: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Jaccard of every column of series (T,P) against the (T,) seed."""
    inter = (series & seed[:, None]).sum(axis=0).astype(np.float32)
    union = series.sum(axis=0) + seed.sum() - inter
    return (inter / np.maximum(union, 1)).astype(np.float32)


def _boxes_from_mask(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    lab, n = cc_label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < min_area:
            continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return sorted(out, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)


def group_events(
    on: np.ndarray,
    event_mask: np.ndarray,
    times: np.ndarray,
    sample_fps: float,
    params: GroupParams | None = None,
) -> list[EventGroup]:
    p = params or GroupParams()
    t, h, w = on.shape
    ys, xs = np.nonzero(event_mask)
    if ys.size < p.min_pixels:
        return []

    series = on[:, ys, xs]                      # (T, P) activation per candidate pixel
    min_run = max(2.0, p.min_event_s * sample_fps)
    dt = 1.0 / sample_fps
    # An element worth anchoring a replay to covers a visible part of the
    # screen, so the size floor scales with the frame rather than being fixed.
    min_px = max(p.min_pixels, int(p.min_pixel_fraction * h * w))

    # Column index of every candidate pixel, so a spatial component can be
    # turned into a set of time series.
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(ys.size, dtype=np.int32)
    remaining_mask = event_mask.copy()

    # Note there is deliberately no cap on the number of raw candidates here:
    # most of them are rejected by the validation below, and capping before
    # validation lets clutter use up every slot and hide a real event behind
    # it. max_iterations bounds the work; max_groups bounds the output.
    raw: list[tuple[np.ndarray, np.ndarray]] = []   # (member indices, activation)
    for _ in range(p.max_iterations):
        lab, n = cc_label(remaining_mask)
        if n == 0:
            break
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        comp = int(np.argmax(sizes))
        if sizes[comp] < p.min_pixels:
            break

        # Seeding from the largest remaining blob, with a majority vote across
        # it, is far more robust than seeding from one pixel: HUD elements are
        # big and contiguous, while spurious candidates are scattered.
        blob = lab == comp
        seed_cols = idx_map[blob]
        seed_cols = seed_cols[seed_cols >= 0]
        pool = idx_map[remaining_mask]
        pool = pool[pool >= 0]
        members = np.empty(0, dtype=np.int32)
        activation = _close1d(series[:, seed_cols].mean(axis=1) >= p.vote, p.close_gaps)
        if activation.any():
            members = pool[_jaccard_vs(series[:, pool], activation) >= p.jaccard]

        # On real footage one blob can hold several elements side by side
        # (a death banner touching a chat line touching a hotbar), and the
        # blob-wide vote is then a timeline none of them has. Fall back to
        # the blob's medoid: the pixel whose timeline the most other blob
        # pixels share, which is a seed from the blob's largest coherent part.
        if members.size < min_px:
            rng = np.random.default_rng(len(raw))
            sample = seed_cols if seed_cols.size <= 256 else rng.choice(seed_cols, 256, replace=False)
            a = series[:, sample].astype(np.float32)
            inter = a.T @ a
            n = a.sum(axis=0)
            jac = inter / np.maximum(n[:, None] + n[None, :] - inter, 1.0)
            support = (jac >= p.jaccard).sum(axis=1)
            med = int(np.argmax(support))
            agree = sample[jac[med] >= p.jaccard]
            activation = _close1d(series[:, agree].mean(axis=1) >= p.vote, p.close_gaps)
            if activation.any():
                members = pool[_jaccard_vs(series[:, pool], activation) >= p.jaccard]
            if members.size < min_px:
                # Nothing coherent here at all: drop what the medoid matched,
                # or the whole blob if even that is empty, and move on.
                drop = members if members.size else seed_cols
                remaining_mask[ys[drop], xs[drop]] = False
                if not members.size:
                    remaining_mask[blob] = False
                continue

        # Re-vote over the full membership, which may reach beyond the seed
        # blob (a kill feed has several disjoint rows sharing one timeline).
        activation = _close1d(series[:, members].mean(axis=1) >= p.vote, p.close_gaps)
        remaining_mask[ys[members], xs[members]] = False
        if not activation.any():
            continue
        raw.append((members, activation))

    # Two elements that always co-occur are one event: a panel and the text
    # drawn on it, or the several patches a partly-detected bar breaks into.
    # Co-occurrence alone needs a high bar, but a looser one is safe when the
    # two also sit next to each other on screen, so both rules are applied to
    # a fixed point - fragments chain together transitively.
    merged = list(raw)
    for _ in range(6):
        if len(merged) < 2:
            break
        boxes_now = [_bbox(ys, xs, m) for m, _ in merged]
        acts = np.stack([a for _, a in merged]).astype(np.int32)   # (G, T)
        sums = acts.sum(axis=1)
        inters = acts @ acts.T                                     # (G, G) co-activation
        joined = False
        out: list[tuple[np.ndarray, np.ndarray]] = []
        taken = [False] * len(merged)
        for i in range(len(merged)):
            if taken[i]:
                continue
            m_i, a_i = merged[i]
            grown = False   # once a_i absorbs another timeline the cached sums are stale
            for j in range(i + 1, len(merged)):
                if taken[j]:
                    continue
                m_j, a_j = merged[j]
                sj = int(sums[j])
                if grown:
                    si, inter = int(a_i.sum()), int((a_i & a_j).sum())
                else:
                    si, inter = int(sums[i]), int(inters[i, j])
                union = si + sj - inter
                if not union or not min(si, sj):
                    continue
                jac = inter / union
                contained = inter / min(si, sj)

                if jac >= p.merge_jaccard:
                    # The same timeline seen twice: either activation will do.
                    m_i, a_i = np.concatenate([m_i, m_j]), a_i | a_j
                elif contained >= p.containment and _near(
                    boxes_now[i], boxes_now[j], p.adjacency_px
                ):
                    # A patch of a partly-detected element, whose timeline is a
                    # subset of the whole. Keep the superset rather than OR-ing,
                    # which would invent occurrences the element never had.
                    m_i = np.concatenate([m_i, m_j])
                    a_i = a_i if si >= sj else a_j
                else:
                    continue
                grown = True
                taken[j] = True
                joined = True
            taken[i] = True
            out.append((m_i, a_i))
        merged = out
        if not joined:
            break

    def reject(reason: str, members: np.ndarray, spans: list, **extra) -> None:
        if p.explain is not None:
            p.explain.append({"reason": reason, "bbox": _bbox(ys, xs, members),
                              "n_pixels": int(members.size), "n_spans": len(spans), **extra})

    groups: list[EventGroup] = []
    for members, activation in merged:
        spans = [(s, e) for s, e in _runs(activation) if (e - s) >= min_run]
        if len(spans) < p.min_occurrences:
            reject("too few occurrences", members, spans)
            continue

        if members.size < min_px:
            reject("too small", members, spans)
            continue

        # Members were admitted at the membership threshold; requiring the
        # *average* agreement to be far higher separates an element whose
        # pixels really share one timeline from a loose cloud that merely
        # passed the bar.
        coherence = float(_jaccard_vs(series[:, members], activation).mean())
        if coherence < p.min_coherence:
            reject("incoherent", members, spans, coherence=round(coherence, 3))
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[ys[members], xs[members]] = True
        boxes = _boxes_from_mask(mask, min_area=max(2, p.min_pixels // 4))
        if not boxes:
            continue
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)

        # A HUD element substantially fills its own bounding box; a cloud of
        # coincidentally-correlated world pixels does not.
        bbox_area = max(1, (x1 - x0) * (y1 - y0))
        if members.size / bbox_area < p.min_fill:
            reject("sparse", members, spans, fill=round(members.size / bbox_area, 3))
            continue

        # And it lasts about the same length of time every time it appears.
        durs = np.array([_span_duration(times, s, e, dt) for s, e in spans])
        if durs.mean() > 0 and durs.std() / durs.mean() > p.max_duration_cv:
            reject("inconsistent duration", members, spans,
                   duration_cv=round(float(durs.std() / durs.mean()), 3))
            continue

        occ = [
            Occurrence(
                start_idx=int(s),
                end_idx=int(e),
                start_s=float(times[s]),
                duration_s=_span_duration(times, s, e, dt),
            )
            for s, e in spans
        ]
        starts = np.array([o.start_s for o in occ])
        gaps = np.diff(starts)

        groups.append(
            EventGroup(
                gid=0,
                n_pixels=int(members.size),
                roi=(x0, y0, x1, y1),
                roi_norm=(x0 / w, y0 / h, x1 / w, y1 / h),
                boxes=boxes,
                occurrences=occ,
                mean_duration_s=float(np.mean([o.duration_s for o in occ])),
                median_gap_s=float(np.median(gaps)) if gaps.size else 0.0,
                activation=activation,
                mask=mask,
            )
        )

    # Most-recurring, longest-lived elements first, which puts the useful
    # review anchors (death, round start) at the top of the report.
    groups.sort(key=lambda g: len(g.occurrences) * g.mean_duration_s, reverse=True)
    groups = groups[:p.max_groups]
    for i, g in enumerate(groups):
        g.gid = i
    return groups


def save(groups: list[EventGroup], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(
        json.dumps([g.to_dict() for g in groups], indent=2), encoding="utf-8"
    )


def appearance_consistency(group: EventGroup, gray: np.ndarray, color: np.ndarray) -> tuple[float, float, float]:
    """(colour spread, structural agreement, texture) of a group across its occurrences.

    Colour spread is the median Lab distance of each occurrence's mean colour
    from the median colour; structural agreement is the median correlation of
    the grey ROI between occurrences; texture is the median grey std inside
    the ROI. An interface element looks the same
    every time it appears - same colour, same shape - while a patch of world
    that happened to share a timeline shows whatever was there: sky blue one
    time, lava orange the next. Grey-level discovery cannot tell those apart
    (blue and orange can have the same luminance); this can.
    """
    import cv2

    t, h, w = gray.shape
    ch, cw = color.shape[1:3]
    x0, y0, x1, y1 = group.roi
    cx0, cx1 = int(x0 * cw / w), max(int(x0 * cw / w) + 1, int(round(x1 * cw / w)))
    cy0, cy1 = int(y0 * ch / h), max(int(y0 * ch / h) + 1, int(round(y1 * ch / h)))
    mids = [min(t - 1, (o.start_idx + o.end_idx - 1) // 2) for o in group.occurrences]
    labs = []
    for m in mids:
        patch = np.ascontiguousarray(color[m, cy0:cy1, cx0:cx1])
        labs.append(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32).mean(0))
    labs = np.array(labs)
    spread = float(np.median(np.linalg.norm(labs - np.median(labs, axis=0), axis=1)))

    pts = np.array([gray[m, y0:y1, x0:x1].astype(np.float32).ravel() for m in mids])
    texture = float(np.median(pts.std(axis=1)))
    pts = (pts - pts.mean(1, keepdims=True)) / np.maximum(pts.std(1, keepdims=True), 1e-3)
    corr = pts @ pts.T / max(pts.shape[1], 1)
    agree = float(np.median(np.median(corr, axis=1)))
    return spread, agree, texture


def filter_by_appearance(groups: list[EventGroup], gray: np.ndarray, color: np.ndarray,
                         max_spread: float = 3.5, min_agree: float = 0.5,
                         max_flat_texture: float = 4.0,
                         explain: list | None = None) -> list[EventGroup]:
    """Keep groups whose occurrences share a shape, or are one flat colour.

    Shape agreement is the strong test. An element with no shape to agree on
    (a plain bar, a translucent chat strip) passes only if it is genuinely
    flat and the same colour every time; a dark patch of sky that walls
    sometimes cut into is neither.
    """
    kept = []
    for g in groups:
        spread, agree, texture = appearance_consistency(g, gray, color)
        flat_and_steady = spread <= max_spread and texture <= max_flat_texture
        if agree < min_agree and not flat_and_steady:
            if explain is not None:
                explain.append({"reason": "inconsistent appearance", "bbox": g.roi,
                                "n_pixels": g.n_pixels, "n_spans": len(g.occurrences),
                                "colour_spread": round(spread, 2), "agreement": round(agree, 3),
                                "texture": round(texture, 2)})
            continue
        kept.append(g)
    for i, g in enumerate(kept):
        g.gid = i
    return kept
