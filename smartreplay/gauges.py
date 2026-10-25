"""Find bar gauges (health, mana, objective HP) and read their fill over time.

Event HUD snaps to one value and holds; a gauge is different. At any moment
its lit pixels form a prefix along one direction - everything before the fill
edge is lit, everything after is not. That is the whole definition used here:

    a gauge is a direction and the largest set of pixels whose state, frame
    after frame, is explained by a single cut along that direction.

"Explained" is measured per pixel as Cohen's kappa between the pixel's state
and the cut's prediction, over frames where the bar is partly full: agreement
beyond what the pixel's and the prediction's own base rates give by chance.
(Once the direction is fixed, a pixel on the filled side of every such cut,
like a bar's last sliver, is judged over all frames instead - no variance is
not disagreement.)
A pixel that is always lit (a frame, a label) or lit at random (text, world
showing through) has kappa near zero whatever its raw agreement, so it falls
out. The same rule decides where a gauge ends: pieces of one bar separated by
overlaid text rejoin because one cut explains both; a number box drawn over a
bar is dropped because no cut explains it; elements packed into one panel
separate because each is explained by its own cut, not a shared one.

Frames where the bar is covered (a menu, an effect) are found the same way:
a frame whose members disagree with its best cut by more than the gauge's own
per-pixel noise allows (a 3-sigma test) is unreadable, reported as NaN.

The definition has one free parameter, the kappa a member must reach (0.5,
"moderate agreement" by the usual convention). The candidate filter in
`candidates` is a compute prior - it limits which pixels are tried - and is
not part of what makes something a gauge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import label as cc_label

N_BINS = 32
BIN_W = 256 // N_BINS


@dataclass
class GaugeParams:
    min_kappa: float = 0.5         # the definition's one parameter (see module docstring)
    min_evidence: int = 16         # partly-full frames needed before an ordering is believed
    min_px: int = 12               # members needed for a reading to mean anything
    outlier_sigma: float = 3.0     # per-frame disagreement test
    # Compute prior only: pixels tried as members hold one value often but not always.
    min_fill_mass: float = 0.50
    max_fill_mass: float = 0.97


@dataclass
class Gauge:
    gid: int
    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1 in stack pixels
    bbox_norm: tuple[float, float, float, float]
    n_px: int
    angle_deg: float                       # fill direction, filled end first
    kappa: float                           # mean member kappa
    fill: np.ndarray = field(repr=False)   # (T,) float in [0, 1], NaN where unreadable
    valid: float = 0.0                     # fraction of frames readable
    ys: np.ndarray = field(default=None, repr=False)     # member pixels, stack coords
    xs: np.ndarray = field(default=None, repr=False)
    cut: np.ndarray = field(default=None, repr=False)    # (T,) cut position, projection units
    visible: np.ndarray | None = field(default=None, repr=False)  # (T,) bar frame on screen


def _dominant_mode(gray: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: its most-visited value (3-bin window), that window's mass, and
    the mean of everything outside it - the pixel's usual "other" value."""
    T, H, W = gray.shape
    P = H * W
    hist = np.zeros((P, N_BINS), np.int64)
    base = np.arange(P, dtype=np.int64) * N_BINS
    for s in range(0, T, chunk):
        b = (gray[s:s + chunk].reshape(-1, P) // BIN_W).astype(np.int64)
        hist += np.bincount((b + base).ravel(), minlength=P * N_BINS).reshape(P, N_BINS)
    h = hist / T
    # Mass in a 3-bin window, so a mode straddling a bin edge is not split.
    win = h.copy()
    win[:, 1:] += h[:, :-1]
    win[:, :-1] += h[:, 1:]
    a = win.argmax(1)
    mass = win[np.arange(P), a]
    centres = np.arange(N_BINS) * BIN_W + BIN_W // 2
    outside = h.copy()
    outside[np.abs(np.arange(N_BINS)[None, :] - a[:, None]) <= 1] = 0
    other = (outside * centres).sum(1) / np.maximum(outside.sum(1), 1e-9)
    fill = centres[a]
    return (fill.reshape(H, W).astype(np.float32), mass.reshape(H, W),
            other.reshape(H, W).astype(np.float32))


def candidates(gray: np.ndarray, p: GaugeParams) -> tuple[np.ndarray, np.ndarray]:
    """Pixels worth trying, and each pixel's (fill, other) values.

    Tried: pixels that hold one value much, but not all, of the time. Only the
    filled state of a bar is fixed; the empty state is often translucent and
    shows whatever is behind it, so it is summarised by its mean, not a mode.
    """
    fill, mass, other = _dominant_mode(gray)
    return (mass >= p.min_fill_mass) & (mass <= p.max_fill_mass), np.stack([fill, other], -1)


def _states(gray, ys, xs, centre) -> np.ndarray:
    """(T, N) bool: each pixel closer to its fill value than to its other value.

    A two-class decision with no width to tune: a translucent or tinted bar
    whose fill brightness drifts with the scene still sits nearer its fill.
    """
    v = gray[:, ys, xs].astype(np.float32)
    return np.abs(v - centre[..., 0]) < np.abs(v - centre[..., 1])


# ------------------------------------------------------------- the model

def _project(ys, xs, deg: float) -> np.ndarray:
    th = np.radians(deg)
    return xs * np.cos(th) + ys * np.sin(th)


def _best_cut(on_sorted: np.ndarray) -> np.ndarray:
    """(T,) index k splitting sorted pixels into lit [0,k) and unlit [k,N) best."""
    T, N = on_sorted.shape
    lit = np.concatenate([np.zeros((T, 1), np.int32), np.cumsum(on_sorted, 1, dtype=np.int32)], 1)
    return (2 * lit - np.arange(N + 1)[None, :]).argmax(1)


def _cut_position(proj_sorted: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Cut index -> position in projection units, between the pixels it splits."""
    ext = np.concatenate([[proj_sorted[0] - 0.5], proj_sorted, [proj_sorted[-1] + 0.5]])
    return (ext[k] + ext[k + 1]) / 2


def _kappa(on: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-column Cohen's kappa between observed and predicted states."""
    po = (on == pred).mean(0)
    a, b = on.mean(0), pred.mean(0)
    pe = a * b + (1 - a) * (1 - b)
    return np.where(1 - pe > 1e-9, (po - pe) / np.maximum(1 - pe, 1e-9), 0.0)


def _noise_test(wrong: np.ndarray, rows: np.ndarray, sigma: float) -> np.ndarray:
    """Frames whose disagreement count is within `sigma` of what member noise allows.

    Each member has its own rate of disagreeing with the cut (estimated over
    `rows`); the count per frame is then a sum of independent Bernoullis.
    """
    r = 1 - wrong[rows].mean(0)
    mu, sd = (1 - r).sum(), np.sqrt((r * (1 - r)).sum())
    return wrong.sum(1) <= mu + sigma * max(sd, 1.0)


@dataclass
class _Fit:
    deg: float
    members: np.ndarray      # bool over the pool
    kappa: np.ndarray        # per pool pixel
    evidence: int


def _fit_once(ys, xs, on, members, deg, p: GaugeParams, extend: bool = False) -> _Fit | None:
    """Cut from the members along deg; kappa of every pool pixel against it.

    With `extend`, pixels on the filled side of every partly-full cut - a
    bar's last sliver, lit unless the player is nearly dead - are judged over
    all frames instead, since no variance is not disagreement. Only used once
    the direction is fixed: over all frames, anything lit whenever the HUD is
    visible also agrees, and letting that vote would pull the direction
    toward whatever angle puts the most such pixels at the filled end.
    """
    proj = _project(ys, xs, deg)
    m = np.nonzero(members)[0]
    if len(m) < p.min_px:
        return None
    order = m[np.argsort(proj[m], kind="stable")]
    k = _best_cut(on[:, order])
    inform = (k > 0) & (k < len(order))
    if inform.sum() < p.min_evidence:
        return None
    cut = _cut_position(proj[order], k)
    pred = proj[None, :] < cut[:, None]
    kap = _kappa(on[inform], pred[inform])
    if extend:
        filled_side = pred[inform].all(0)
        if filled_side.any():
            kap = np.where(filled_side, _kappa(on, pred), kap)
    return _Fit(deg, kap >= p.min_kappa, kap, int(inform.sum()))


def _converge(ys, xs, on, deg, p: GaugeParams, members=None, rounds: int = 5,
              extend: bool = False) -> _Fit | None:
    """Fixed point of (cut from members) -> (members by kappa) along one direction."""
    members = np.ones(len(ys), bool) if members is None else members
    f = None
    for _ in range(rounds):
        nf = _fit_once(ys, xs, on, members, deg, p, extend)
        if nf is None:
            return f
        if f is not None and np.array_equal(nf.members, f.members):
            return nf
        f, members = nf, nf.members
    return f


def _score(f: _Fit) -> tuple[int, float]:
    """Size of the explained set; mean member kappa breaks ties.

    This is the definition read literally - the largest set one cut explains.
    Summing kappa over the whole pool instead lets many half-explained
    non-members outvote a larger set that is actually explained.
    """
    return int(f.members.sum()), float(f.kappa[f.members].mean()) if f.members.any() else 0.0


def _fit(ys, xs, on, p: GaugeParams, degs=None) -> _Fit | None:
    """Direction and member set that one cut explains, each direction at its fixed point.

    Every candidate direction is first taken to its own fixed point - the
    starting cut, estimated from the whole pool, is polluted by whatever is
    not the bar, and it pollutes every direction alike - and only converged
    fits are compared. The best is then refined in 5-degree steps.
    """
    degs = list(range(0, 360, 15)) if degs is None else list(degs)
    fits = [f for f in (_converge(ys, xs, on, d, p) for d in degs) if f is not None]
    if not fits:
        return None
    best = max(fits, key=_score)
    for d in (-10, -5, 5, 10):
        f = _converge(ys, xs, on, (best.deg + d) % 360, p, best.members.copy())
        if f is not None and _score(f) > _score(best):
            best = f
    return best


def _peel(ys, xs, on, p: GaugeParams, max_gauges: int = 4) -> list[tuple[np.ndarray, _Fit]]:
    """Fit a gauge, set its members aside, fit again on what is left."""
    out = []
    left = np.arange(len(ys))
    for _ in range(max_gauges):
        if len(left) < p.min_px:
            break
        f = _fit(ys[left], xs[left], on[:, left], p)
        if f is None or f.members.sum() < p.min_px:
            break
        out.append((left[f.members], f))
        left = left[~f.members]
    return out


def _collinear(a, b) -> bool:
    """Is b's centre inside a's band along a's fill direction, within reach?"""
    (ya, xa, fa), (yb, xb, _) = a, b
    th = np.radians(fa.deg)
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-u[1], u[0]])
    pa = np.stack([xa, ya], 1).astype(float)
    pb = np.stack([xb, yb], 1).astype(float)
    off = pb.mean(0) - pa.mean(0)
    half_widths = np.ptp(pa @ v) / 2 + np.ptp(pb @ v) / 2 + 1
    lengths = np.ptp(pa @ u) + np.ptp(pb @ u)
    return abs(off @ v) <= half_widths and abs(off @ u) <= lengths


def _merge(parts, sub, centre, p: GaugeParams):
    """Join parts that one cut explains together: pieces of a bar split by text.

    A pair merges when, fitted jointly, each keeps a majority of its members.
    Repeated until nothing merges.
    """
    changed = True
    while changed:
        changed = False
        parts.sort(key=lambda t: -len(t[0]))
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a, b = parts[i], parts[j]
                if not (_collinear(a, b) or _collinear(b, a)):
                    continue
                ys = np.concatenate([a[0], b[0]])
                xs = np.concatenate([a[1], b[1]])
                on = _states(sub, ys, xs, centre[ys, xs])
                near = (-10, -5, 0, 5, 10)
                degs = sorted({(a[2].deg + d) % 360 for d in near} | {(b[2].deg + d) % 360 for d in near})
                f = _fit(ys, xs, on, p, degs)
                if f is None:
                    continue
                na = len(a[0])
                if f.members[:na].mean() > 0.5 and f.members[na:].mean() > 0.5:
                    parts[i] = (ys[f.members], xs[f.members], f)
                    del parts[j]
                    changed = True
                    break
            if changed:
                break
    return parts


def _extend(ys, xs, f: _Fit, mask, sub, centre, p: GaugeParams):
    """With the direction fixed, admit the filled-end pixels no partial cut crosses.

    The pool is the candidate pixels in the gauge's box, grown by one pixel.
    """
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - 1), min(H, ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(W, xs.max() + 2)
    py, px = np.nonzero(mask[y0:y1, x0:x1])
    py, px = py + y0, px + x0
    member = np.zeros((H, W), bool)
    member[ys, xs] = True
    on = _states(sub, py, px, centre[py, px])
    g = _converge(py, px, on, f.deg, p, member[py, px].copy(), extend=True)
    if g is None or g.members.sum() < len(ys):
        return ys, xs, f
    return py[g.members], px[g.members], g


def _readout(on: np.ndarray, proj: np.ndarray, p: GaugeParams, rows=None):
    """Cut, fill fraction and validity per frame for members `on` along `proj`."""
    order = np.argsort(proj, kind="stable")
    ps = proj[order]
    k = _best_cut(on[:, order])
    cut = _cut_position(ps, k)
    wrong = on != (proj[None, :] < cut[:, None])
    ok = _noise_test(wrong, np.ones(len(on), bool) if rows is None else rows, p.outlier_sigma)
    fill = np.clip((cut - ps[0]) / max(ps[-1] - ps[0], 1e-9), 0, 1)
    return fill, cut, ok


def find_gauges(gray: np.ndarray, p: GaugeParams | None = None) -> list[Gauge]:
    p = p or GaugeParams()
    T, H, W = gray.shape
    mask, centre = candidates(gray, p)
    lab, n = cc_label(mask, structure=np.ones((3, 3), int))
    sub = gray[::4]        # fitting needs a few hundred partly-full frames, not all of them
    parts = []
    for r in range(1, n + 1):
        ys, xs = np.nonzero(lab == r)
        if len(ys) < p.min_px:
            continue
        on = _states(sub, ys, xs, centre[ys, xs])
        for idx, f in _peel(ys, xs, on, p):
            parts.append((ys[idx], xs[idx], f))
    parts = _merge(parts, sub, centre, p)

    out: list[Gauge] = []
    for ys, xs, f in parts:
        ys, xs, f = _extend(ys, xs, f, mask, sub, centre, p)
        on = _states(gray, ys, xs, centre[ys, xs])
        fill, cut, ok = _readout(on, _project(ys, xs, f.deg), p)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        # The bar's frame - candidate pixels in its box that are usually lit
        # but are not members - shows whether the bar is on screen at all,
        # which an unlit bar cannot: empty and hidden both read as no fill.
        box = mask[y0:y1, x0:x1].copy()
        box[ys - y0, xs - x0] = False
        fy, fx = np.nonzero(box)
        fy, fx = fy + y0, fx + x0
        visible = None
        if len(fy) >= p.min_px:
            lit = _states(gray, fy, fx, centre[fy, fx])
            usual = lit.mean(0) >= 0.5
            if usual.sum() >= p.min_px:
                visible = lit[:, usual].mean(1) >= 0.5
        out.append(Gauge(
            gid=len(out), bbox=(x0, y0, x1, y1), bbox_norm=(x0 / W, y0 / H, x1 / W, y1 / H),
            n_px=int(len(ys)), angle_deg=float(f.deg), kappa=float(f.kappa[f.members].mean()),
            fill=np.where(ok, fill, np.nan), valid=float(ok.mean()),
            ys=ys, xs=xs, cut=cut, visible=visible,
        ))
    return out


# --------------------------------------------------------- colour re-reading

def read_fill_color(gauge: Gauge, video: str, meta: dict, times: np.ndarray,
                    p: GaugeParams | None = None) -> np.ndarray:
    """Re-read a located gauge from the source video, in colour, at full size.

    The grey stack finds a bar and reads its middle, but a tapering bar's low
    end blurs at analysis size - the readings a near-death moment depends on.
    Here the grey model labels the native pixels (before its cut: fill; after:
    backing) to learn one fill and one backing colour; each native pixel is
    then classed by the nearer colour, and the same model - members by kappa,
    best cut, noise test - is fitted again at full size.
    """
    p = p or GaugeParams()
    sw, sh = meta["size"]
    x0c, y0c, x1c, y1c = meta["crop"]
    sx, sy = (x1c - x0c) / sw, (y1c - y0c) / sh
    ys, xs = gauge.ys, gauge.xs

    # Pool: every native pixel in the gauge's box, grown by two analysis
    # pixels. Membership is decided by the model again at this size, so
    # pixels the grey stack could not resolve (a tapering tip) can come back
    # and pixels that are not bar (a number box) are dropped by their kappa.
    bx0, by0 = int(max(0, xs.min() - 2) * sx), int(max(0, ys.min() - 2) * sy)
    bx1 = int(np.ceil(min(sw, xs.max() + 3) * sx))
    by1 = int(np.ceil(min(sh, ys.max() + 3) * sy))
    py, px = np.mgrid[by0:by1, bx0:bx1]
    py, px = py.ravel(), px.ravel()
    proj_stack = _project(py / sy, px / sx, gauge.angle_deg)     # same units as gauge.cut

    cap = cv2.VideoCapture(video)
    want = np.rint(np.asarray(times) * meta["native_fps"]).astype(int)
    lab = np.zeros((len(want), len(py), 3), np.float32)
    got = np.zeros(len(want), bool)
    j, frame_i = 0, 0
    while j < len(want) and cap.grab():
        if frame_i == want[j]:
            ok, frame = cap.retrieve()
            if ok:
                patch = frame[y0c + by0:y0c + by1, x0c + bx0:x0c + bx1]
                lab[j] = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)[py - by0, px - bx0]
                got[j] = True
            j += 1
            while j < len(want) and want[j] <= frame_i:
                j += 1
        frame_i += 1
    cap.release()

    seen = got & ~np.isnan(gauge.fill)
    if seen.sum() < p.min_evidence:
        return np.full(len(want), np.nan)
    labelled = proj_stack[None, :] < gauge.cut[seen][:, None]
    full = np.median(lab[seen][labelled], axis=0)
    empty = np.median(lab[seen][~labelled], axis=0)
    on = np.linalg.norm(lab - full, axis=2) < np.linalg.norm(lab - empty, axis=2)

    # Same model at full size, started from the grey gauge's members (the
    # pool is mostly not bar, and a fit started from all of it is pulled
    # toward whatever else is there), then let kappa admit the tip pixels the
    # grey stack could not resolve and drop anything that does not belong.
    member = np.zeros((sh, sw), bool)
    member[ys, xs] = True
    seed = member[np.clip((py / sy).astype(int), 0, sh - 1), np.clip((px / sx).astype(int), 0, sw - 1)]
    pys, pxs = py.astype(float), px.astype(float)
    sample = on[got][::2]
    f = _converge(pys, pxs, sample, gauge.angle_deg, p, seed.copy())
    if f is None:
        return np.full(len(want), np.nan)
    f = _converge(pys, pxs, sample, f.deg, p, f.members.copy(), extend=True) or f
    keep = f.members
    fill, _, ok = _readout(on[:, keep], _project(pys[keep], pxs[keep], f.deg), p, rows=got)
    # Not gated on the bar frame's visibility: some games recolour the frame
    # at low health (Deadlock turns it red), which is when this reading matters.
    return np.where(got & ok, fill, np.nan)


# ------------------------------------------------------------ gameplay rules

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _smooth(fill: np.ndarray, k: int = 3) -> np.ndarray:
    """Median over readable neighbours; unreadable samples stay NaN."""
    out = fill.copy()
    half = k // 2
    for i in np.nonzero(~np.isnan(fill))[0]:
        w = fill[max(0, i - half): i + half + 1]
        out[i] = np.nanmedian(w)
    return out


def drain_outs(fill: np.ndarray, times: np.ndarray, visible: np.ndarray | None = None,
               empty: float = 0.04, dark_s: float = 4.0, low_before: float = 0.3,
               drop: float = 0.4, look_s: float = 3.0, merge_s: float = 15.0) -> list[float]:
    """Times health fell and the bar then stayed empty or off screen.

    A death as seen from the health bar: within `look_s` before, health was
    low or had just fallen sharply; then for `dark_s` or more the bar reads
    empty, is unreadable, or its frame is gone (kill-cam, death screen).
    Requiring the fall separates deaths from menus opened at healthy levels,
    at the cost of missing deaths so fast the bar never shows a low reading.
    Readings during the dark stretch are not trusted: a kill-cam can show the
    killer's own HUD, bar included.
    """
    dt = float(np.median(np.diff(times)))
    dark = np.isnan(fill) | (fill <= empty)
    if visible is not None:
        dark |= ~visible
    look = max(1, int(look_s / dt))
    out: list[float] = []
    for s, e in _runs(dark):
        if (e - s) * dt < dark_s or s == 0:
            continue
        before = fill[max(0, s - look):s]
        before = before[~np.isnan(before)]
        if not len(before):
            continue
        fell = before.min() <= low_before or before.max() - before[-1] >= drop
        if not fell:
            continue
        if out and times[s] - out[-1] < merge_s:
            continue
        out.append(float(times[s]))
    return out


def pick_health(gauges: list[Gauge], times: np.ndarray) -> Gauge | None:
    """The gauge that sits near full most of the time and empties into deaths.

    Health is full in most frames of ordinary play; resource bars, cooldown
    sweeps and timers are not. Among gauges with a high median reading, the
    one with the most drain-outs wins.
    """
    scored = [(len(drain_outs(g.fill, times, g.visible)), g.valid, g) for g in gauges
              if np.nanmedian(g.fill) >= 0.7]
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return None
    return max(scored, key=lambda s: (s[0], s[1]))[2]


@dataclass
class Escape:
    t_low: float          # time of the lowest reading
    low: float            # fill fraction at that point
    from_level: float     # highest reading in the lead-up
    t_drop: float         # when the fall started
    recovered: float      # highest reading in the follow-up window
    readable_after: float # fraction of the follow-up window that was readable


def narrow_escapes(fill: np.ndarray, times: np.ndarray, low: float = 0.20,
                   lead_s: float = 15.0, from_level: float = 0.60,
                   after_s: float = 12.0, alive: float = 0.05,
                   min_readable_after: float = 0.6, merge_s: float = 20.0) -> list[Escape]:
    """Health fell from healthy to near-empty and the player lived.

    Rule, all from the bar: some reading <= `low`; within `lead_s` before it, a
    reading >= `from_level` (a real fall, not a respawn); over the next
    `after_s`, the bar stays readable for most of the window and never empties
    into a death (see drain_outs). Lowest point per `merge_s` window wins.
    """
    h = _smooth(fill)
    dt = float(np.median(np.diff(times)))
    deaths = np.asarray(drain_outs(fill, times))
    lead, after = int(lead_s / dt), int(after_s / dt)
    cands = np.nonzero((h <= low) & (h > alive))[0]
    found: list[Escape] = []
    for i in cands:
        pre = h[max(0, i - lead):i]
        if not np.any(pre >= from_level):
            continue
        post = h[i + 1:i + 1 + after]
        if len(post) < after // 2:
            continue
        readable = float((~np.isnan(post)).mean())
        if readable < min_readable_after:
            continue
        if len(deaths) and np.any((deaths >= times[i]) & (deaths <= times[i] + after_s)):
            continue
        if np.nanmin(post) <= alive / 2:
            continue
        j = max(0, i - lead) + int(np.nanargmax(np.where(pre >= from_level, np.arange(len(pre)), -1)))
        found.append(Escape(float(times[i]), float(h[i]), float(np.nanmax(pre)),
                            float(times[j]), float(np.nanmax(post)), readable))
    found.sort(key=lambda e: e.low)
    kept: list[Escape] = []
    for e in found:
        if all(abs(e.t_low - k.t_low) > merge_s for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.t_low)


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    """Find gauges in a sampled clip, read health in colour, list escapes.

        python -m smartreplay.gauges out/bench/deadlock_celeste
        python -m smartreplay.gauges out/bench/deadlock_celeste --gauge 3
    """
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("out_dir", help="a discovery output folder (gray.npy, times.npy, meta.json)")
    ap.add_argument("--gauge", type=int, default=None,
                    help="use this gauge as health instead of the automatic pick")
    ap.add_argument("--low", type=float, default=0.30)
    ap.add_argument("--colour", action="store_true",
                    help="re-read health from the source video in colour (experimental)")
    args = ap.parse_args(argv)

    d = Path(args.out_dir)
    gray = np.load(d / "gray.npy")
    times = np.load(d / "times.npy")
    meta = json.loads((d / "meta.json").read_text())
    gauges = find_gauges(gray)
    for g in gauges:
        print(f"  g{g.gid:<2} bbox {tuple(round(v, 3) for v in g.bbox_norm)}  px {g.n_px:<5} "
              f"angle {g.angle_deg:>5.0f}  kappa {g.kappa:.2f}  valid {g.valid:.2f}  "
              f"median {np.nanmedian(g.fill):.2f}")
    health = (gauges[args.gauge] if args.gauge is not None else pick_health(gauges, times))
    if health is None:
        print("no health gauge found")
        return 1
    print(f"health: g{health.gid} {tuple(round(v, 3) for v in health.bbox_norm)}"
          + ("" if args.gauge is not None else "  (automatic pick - check it)"))
    # The colour re-read (read_fill_color) is meant to recover a tapering bar's
    # low end; it does not yet, so the grey reading is used.
    fill = read_fill_color(health, meta["source"], meta, times) if args.colour else health.fill
    np.save(d / "health_fill.npy", fill)
    deaths = drain_outs(fill, times, health.visible)
    escapes = narrow_escapes(fill, times, low=args.low, lead_s=20)
    print(f"deaths ({len(deaths)}): {[round(t) for t in deaths]}")
    print(f"narrow escapes ({len(escapes)}):")
    for e in escapes:
        print(f"  {e.t_low:7.1f}s  fell {e.from_level:.2f} -> {e.low:.2f} from {e.t_drop:.0f}s, "
              f"back to {e.recovered:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
