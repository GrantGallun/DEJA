"""Emit a draft Game Profile from discovery output.

The profile is the unit of per-game adaptability: it carries the detector
geometry, the exemplar templates a cheap runtime matcher scores against, the
regions the game's own interface occupies, and a suggested spot to park the
mini-player where it occludes nothing. Labels are left null here; naming the
discovered elements is a separate step.

Everything spatial is stored in normalised coordinates so a profile learned at
one resolution is at least a starting point at another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label

from .variants import crop_norm


def exe_id(source: str) -> str:
    """Stand-in for a real executable identity, derived from the file name."""
    stem = Path(source).stem.lower()
    return f"{stem}-{hashlib.sha1(stem.encode()).hexdigest()[:8]}"


def _largest_free_rect(free: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle, via the histogram-stack method.

    Row by row, heights[x] counts how many free rows end at this one, which
    reduces the 2-D problem to the largest rectangle in a histogram.
    """
    h, w = free.shape
    heights = np.zeros(w, dtype=np.int64)
    best_area = 0
    best = (0, 0, 0, 0)

    for y in range(h):
        heights = np.where(free[y], heights + 1, 0)
        hs = heights.tolist()
        stack: list[int] = []
        for x in range(w + 1):
            cur = hs[x] if x < w else 0
            while stack and hs[stack[-1]] >= cur:
                top = stack.pop()
                height = hs[top]
                left = stack[-1] + 1 if stack else 0
                area = height * (x - left)
                if area > best_area:
                    best_area = area
                    best = (left, y - height + 1, x, y + 1)
            stack.append(x)
    return best


def hud_safe_zones(result, groups=None, dilate: int = 2) -> tuple[list[list[float]], list[float]]:
    """Normalised boxes covering the game interface, plus a free rectangle.

    Only *confirmed* interface blocks the player: the always-on HUD, plus the
    pixels of elements that survived grouping. The raw candidate mask is far
    too permissive - on real footage it can cover most of the frame - and
    using it leaves no sensible free space to suggest.
    """
    occupied = result.static_mask.copy()
    if groups:
        for g in groups:
            occupied |= g.mask
    elif groups is None:
        occupied |= result.event_mask
    if dilate:
        occupied = binary_dilation(occupied, iterations=dilate)
    h, w = occupied.shape

    lab, n = cc_label(occupied)
    zones = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < 12:
            continue
        zones.append([
            round(float(xs.min()) / w, 5), round(float(ys.min()) / h, 5),
            round(float(xs.max() + 1) / w, 5), round(float(ys.max() + 1) / h, 5),
        ])
    zones.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    x0, y0, x1, y1 = _largest_free_rect(~occupied)
    free = [round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5)]
    return zones[:32], free


def build(meta: dict, result, groups, variants, gray: np.ndarray) -> tuple[dict, dict]:
    """Return (profile dict, {template key: uint8 array})."""
    by_gid = {gv.gid: gv for gv in variants}
    w, h = meta["size"]
    templates: dict[str, np.ndarray] = {}
    detectors = []

    for g in groups:
        gv = by_gid[g.gid]
        keys = []
        for n, (occ_i, _) in enumerate(gv.exemplars):
            occ = g.occurrences[occ_i]
            mid = min(len(gray) - 1, (occ.start_idx + occ.end_idx) // 2)
            patch = crop_norm(gray[mid], g.roi_norm)
            if patch.size == 0:
                continue
            key = f"d{g.gid}_e{n}"
            templates[key] = np.ascontiguousarray(patch)
            keys.append(key)

        durations = [o.duration_s for o in g.occurrences]
        detectors.append({
            "id": f"d{g.gid}",
            "label": None,
            "roi_norm": [round(v, 5) for v in g.roi_norm],
            "templates": keys,
            "min_duration_ms": int(max(200.0, min(durations) * 1000 * 0.6)),
            "observed": {
                "n_occurrences": len(g.occurrences),
                "mean_duration_s": round(g.mean_duration_s, 3),
                "median_gap_s": round(g.median_gap_s, 3),
                "n_variants": len(gv.variants),
                "n_pixels": g.n_pixels,
            },
            "onsets_s": [round(o.start_s, 3) for o in g.occurrences],
        })

    zones, free = hud_safe_zones(result, groups)

    # Every *_norm coordinate above is relative to the letterbox-cropped
    # frame. Record the crop so a runtime can map back to the full capture.
    crop = [int(v) for v in meta.get("crop", [0, 0, w, h])]
    frame = meta.get("frame_size") or [0, 0]
    if not frame[0] or not frame[1]:
        # Older stacks did not record the source size; with no crop the
        # cropped size is the source size, otherwise the crop is a lower bound.
        frame = [max(crop[2], meta["native_size"][0]), max(crop[3], meta["native_size"][1])]
    fw, fh = int(frame[0]), int(frame[1])
    crop_n = [round(crop[0] / fw, 5), round(crop[1] / fh, 5),
                 round(crop[2] / fw, 5), round(crop[3] / fh, 5)]

    def to_frame(r):
        cx0, cy0, cx1, cy1 = crop_n
        sx, sy = cx1 - cx0, cy1 - cy0
        return [round(cx0 + r[0] * sx, 5), round(cy0 + r[1] * sy, 5),
                round(cx0 + r[2] * sx, 5), round(cy0 + r[3] * sy, 5)]

    for det in detectors:
        det["roi_norm_frame"] = to_frame(det["roi_norm"])

    profile = {
        "schema": 1,
        "exe_id": exe_id(meta["source"]),
        "source": meta["source"],
        "learned_at": {
            "analysis_size": [w, h],
            "native_size": meta["native_size"],
            "aspect": round(meta["native_size"][0] / max(meta["native_size"][1], 1), 4),
            "sample_fps": meta["sample_fps"],
            "footage_s": round(meta["duration_s"], 1),
            "frame_size": [fw, fh],
            "crop": crop,
            "crop_norm": crop_n,
            "coords": "roi_norm and ui boxes are relative to the crop; "
                      "roi_norm_frame is relative to the full source frame",
        },
        "detectors": detectors,
        "ui": {
            "hud_safe_zones": zones,
            "suggested_player_rect": free,
        },
        "overlay_backend": {"mode": "topmost", "anticheat_class": "unknown"},
        "telemetry": None,
    }
    return profile, templates


def save(profile: dict, templates: dict[str, np.ndarray], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile_draft.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    archive = out_dir / "templates.npz"
    if templates:
        np.savez_compressed(archive, **templates)
    elif archive.exists():
        # A profile with no detectors must not leave a previous run's
        # templates next to it, where they would be mistaken for its own.
        archive.unlink()
