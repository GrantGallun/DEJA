"""Visual output: a HUD map per game and a contact sheet per discovered event.

The point of these is to make the pipeline falsifiable by eye. The HUD map
says where the classifier thinks the interface lives, and each contact sheet
says what one discovered element looks like, how often it fired, and when, so
a wrong answer is obvious at a glance instead of hidden in a JSON file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 22, 20)
FG = (238, 238, 238)
DIM = (150, 150, 150)
ACCENT = (90, 180, 255)
EVENT_TINT = (70, 70, 235)
STATIC_TINT = (90, 200, 120)


def _text(img, s, org, scale=0.42, color=FG, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _paste(canvas, img, x, y):
    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]
    if y >= ch or x >= cw:
        return
    h = min(h, ch - y)
    w = min(w, cw - x)
    canvas[y:y + h, x:x + w] = img[:h, :w]


def _fit(img, box_w, box_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((1, 1, 3), np.uint8)
    s = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_NEAREST if s > 2 else cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _context_frame(color: np.ndarray, groups) -> np.ndarray:
    """A frame with as few events active as possible, for use as a backdrop."""
    if not groups:
        return color[len(color) // 2]
    active = np.zeros(len(color), dtype=np.int32)
    for g in groups:
        active[: len(g.activation)] += g.activation.astype(np.int32)
    return color[int(np.argmin(active))]


def hud_map(color: np.ndarray, result, groups, out_path: str | Path, width: int = 1100) -> None:
    base = _context_frame(color, groups)
    h, w = base.shape[:2]
    scale = width / w
    disp = cv2.resize(base, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    def up(mask):
        return cv2.resize(mask.astype(np.uint8), (disp.shape[1], disp.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    confirmed = np.zeros_like(result.event_mask)
    for g in groups:
        confirmed |= g.mask

    # Candidates are tinted faintly and confirmed elements strongly. Painting
    # both at full strength would be misleading: the per-pixel stage is
    # deliberately permissive, and most of what it admits is rejected later.
    cand = up(result.event_mask & ~confirmed)
    st = up(result.static_mask)
    conf = up(confirmed)

    faint = disp.copy()
    faint[cand] = EVENT_TINT
    disp = cv2.addWeighted(faint, 0.22, disp, 0.78, 0)

    strong = disp.copy()
    strong[st] = STATIC_TINT
    strong[conf] = EVENT_TINT
    disp = cv2.addWeighted(strong, 0.60, disp, 0.40, 0)

    gh, gw = result.event_mask.shape
    for g in groups:
        x0, y0, x1, y1 = g.roi
        p0 = (int(x0 / gw * disp.shape[1]), int(y0 / gh * disp.shape[0]))
        p1 = (int(x1 / gw * disp.shape[1]), int(y1 / gh * disp.shape[0]))
        cv2.rectangle(disp, p0, p1, ACCENT, 2)
        label = f"g{g.gid}  x{len(g.occurrences)}  {g.mean_duration_s:.1f}s"
        ty = p0[1] - 6 if p0[1] > 18 else p1[1] + 14
        cv2.rectangle(disp, (p0[0], ty - 12), (p0[0] + 9 * len(label), ty + 4), BG, -1)
        _text(disp, label, (p0[0] + 3, ty), 0.42, ACCENT)

    header = np.full((78, disp.shape[1], 3), BG, np.uint8)
    _text(header, "HUD map", (14, 24), 0.62, FG, 2)
    s = result.summary()
    _text(header, f"{int(confirmed.sum())} px in {len(groups)} confirmed element(s)   "
                  f"{s['event_pixels']} candidate px ({s['event_fraction'] * 100:.1f}%)   "
                  f"{s['static_pixels']} static px   of {s['total_pixels']}",
          (14, 46), 0.42, DIM)
    _text(header, "solid red = confirmed element    faint red = candidate pixel, rejected"
                  "    green = static HUD    blue boxes = element bounds",
          (14, 66), 0.40, DIM)

    out = np.vstack([header, disp])
    cv2.imwrite(str(out_path), out)


def _timeline(activation: np.ndarray, times: np.ndarray, width: int, height: int = 44) -> np.ndarray:
    strip = np.full((height, width, 3), (40, 38, 36), np.uint8)
    t = len(activation)
    xs = (np.arange(t) / max(t - 1, 1) * (width - 1)).astype(int)
    on = np.flatnonzero(activation)
    for x in xs[on]:
        cv2.line(strip, (int(x), 6), (int(x), height - 16), (90, 180, 255), 1)

    total = float(times[-1]) if len(times) else 0.0
    step = 60.0 if total > 240 else 30.0
    mark = 0.0
    while mark <= total:
        x = int(mark / max(total, 1e-6) * (width - 1))
        cv2.line(strip, (x, height - 14), (x, height - 9), DIM, 1)
        _text(strip, f"{int(mark // 60)}:{int(mark % 60):02d}", (max(0, x - 12), height - 2), 0.34, DIM)
        mark += step
    return strip


def contact_sheet(
    group,
    gv,
    color: np.ndarray,
    times: np.ndarray,
    groups,
    out_path: str | Path,
    width: int = 1100,
) -> None:
    pad = 14
    ctx_w, ctx_h = 520, 293
    canvas_h = 64 + max(ctx_h, 293) + 44 + pad * 3
    canvas = np.full((canvas_h, width, 3), BG, np.uint8)

    _text(canvas, f"group {group.gid}", (pad, 26), 0.66, FG, 2)
    _text(canvas,
          f"{len(group.occurrences)} occurrences   mean {group.mean_duration_s:.2f}s   "
          f"median gap {group.median_gap_s:.1f}s   {group.n_pixels} px   "
          f"{len(gv.variants)} variant(s)",
          (pad, 48), 0.44, DIM)
    rn = group.roi_norm
    _text(canvas, f"roi_norm ({rn[0]:.3f}, {rn[1]:.3f}) - ({rn[2]:.3f}, {rn[3]:.3f})",
          (width - 330, 48), 0.44, DIM)

    # Backdrop with this group boxed, so position on screen is unambiguous.
    base = _context_frame(color, groups).copy()
    gh, gw = group.mask.shape
    bh, bw = base.shape[:2]
    x0, y0, x1, y1 = group.roi
    cv2.rectangle(base,
                  (int(x0 / gw * bw), int(y0 / gh * bh)),
                  (int(x1 / gw * bw), int(y1 / gh * bh)), ACCENT, 1)
    ctx = _fit(base, ctx_w, ctx_h)
    top = 64 + pad
    _paste(canvas, ctx, pad, top)

    # Exemplar crops, one per variant, largest variant first.
    ex_x = pad + ctx_w + pad
    ex_w = width - ex_x - pad
    _text(canvas, "exemplars", (ex_x, top - 4), 0.44, DIM)
    cell_w = max(80, ex_w // max(1, min(3, len(gv.exemplars))) - pad)
    cx, cy, row_h = ex_x, top + 8, 0
    for occ_i, crop in gv.exemplars:
        if crop is None or crop.size == 0:
            continue
        thumb = _fit(crop, cell_w, 120)
        if cx + thumb.shape[1] > width - pad:
            cx = ex_x
            cy += row_h + 22
            row_h = 0
        if cy + thumb.shape[0] > top + ctx_h:
            break
        _paste(canvas, thumb, cx, cy)
        cv2.rectangle(canvas, (cx, cy), (cx + thumb.shape[1], cy + thumb.shape[0]), (70, 68, 66), 1)
        _text(canvas, f"t={group.occurrences[occ_i].start_s:.1f}s",
              (cx, cy + thumb.shape[0] + 13), 0.36, DIM)
        row_h = max(row_h, thumb.shape[0])
        cx += thumb.shape[1] + pad

    strip = _timeline(group.activation, times, width - pad * 2)
    _paste(canvas, strip, pad, 64 + pad + ctx_h + pad)
    cv2.imwrite(str(out_path), canvas)


def write_reports(color, times, result, groups, variants, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A re-run with different settings usually finds a different number of
    # groups; leaving the previous run's sheets behind makes the report lie.
    for stale in out_dir.glob("group_*.png"):
        stale.unlink()
    paths = [out_dir / "hud_map.png"]
    hud_map(color, result, groups, paths[0])
    by_gid = {gv.gid: gv for gv in variants}
    for g in groups:
        p = out_dir / f"group_{g.gid:02d}.png"
        contact_sheet(g, by_gid[g.gid], color, times, groups, p)
        paths.append(p)
    return paths
