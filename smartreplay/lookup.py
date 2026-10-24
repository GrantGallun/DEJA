"""Bootstrap a detector for an event from web images of it.

Discovery (discovery.py) learns a game's event vocabulary from your own
footage, which is precise but needs the event to have already happened several
times, and produces unlabelled clusters. Lookup is the complement: it works
from zero occurrences in your footage and the label is known by construction,
because you searched for it by name.

The one thing lookup must *not* do is use the fetched pixels as the detection
template. A stranger's screenshot is a different resolution, a different UI
scale and a different 3D background than your capture, so matching it directly
against your frames fails. What survives across many players' screenshots of
the same event is the interface chrome itself - the banner, the panel, the
fixed text - while the background, names and scores differ. That is the same
consensus trick discovery uses over time, applied across images instead:

    per-pixel variance across images  ->  low where the UI is, high elsewhere

So lookup produces *geometry and a label*. The actual template is re-derived
from the user's own footage once that geometry has found a single real hit.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmartReplay/0.1"
MAX_BYTES = 12 * 1024 * 1024

# Full-screen game captures land on a small set of display ratios. Share cards,
# article photos and cropped thumbnails mostly do not.
RATIOS = (16 / 9, 16 / 10, 21 / 9, 4 / 3)
RATIO_TOL = 0.04


@dataclass
class Candidate:
    url: str
    path: Path
    width: int
    height: int
    kept: bool = True
    reason: str = ""
    cluster: int = -1
    index: int = -1


@dataclass
class ConsensusResult:
    size: tuple[int, int]                 # (w, h) of the normalised space
    invariant: np.ndarray = field(repr=False)   # (H,W) bool
    template: np.ndarray = field(repr=False)    # (H,W) uint8 median
    rois_norm: list[tuple[float, float, float, float]] = field(default_factory=list)
    n_images: int = 0

    def to_dict(self) -> dict:
        return {
            "size": list(self.size),
            "n_images": self.n_images,
            "invariant_fraction": float(self.invariant.mean()),
            "rois_norm": [[round(v, 5) for v in r] for r in self.rois_norm],
        }


def build_queries(game: str, event: str) -> list[str]:
    """A few phrasings, because image search recall on any single one is poor."""
    event = event.strip()
    game = game.strip()
    base = [
        f"{game} {event}",
        f"{game} {event} screenshot",
        f"{game} {event} screen",
    ]
    if "death" in event.lower():
        base += [f"{game} you died screen", f"{game} defeat screen"]
    # Preserve order, drop repeats.
    seen, out = set(), []
    for q in base:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def search_images(queries: list[str], max_results: int = 40) -> list[dict]:
    """Image search results across several phrasings, deduplicated by URL."""
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("pip install ddgs") from exc

    per = max(4, max_results // max(1, len(queries)))
    seen: set[str] = set()
    out: list[dict] = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                hits = list(ddgs.images(q, max_results=per))
            except Exception as exc:
                print(f"  query failed ({q}): {type(exc).__name__}")
                continue
            for h in hits:
                url = h.get("image") or ""
                if url and url not in seen:
                    seen.add(url)
                    out.append(h)
    return out[:max_results]


def _ratio_ok(w: int, h: int) -> bool:
    if h == 0:
        return False
    r = w / h
    return any(abs(r - target) / target <= RATIO_TOL for target in RATIOS)


def _cache_name(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".png"


def _fetch_one(url: str, out_dir: Path, timeout: float) -> tuple[np.ndarray | None, str]:
    """Return (image, error). Cached on disk by URL hash so re-runs are free."""
    path = out_dir / _cache_name(url)
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return img, ""

    origin = "/".join(url.split("/")[:3])
    headers = {
        "User-Agent": UA,
        # Plenty of CDNs reject image requests that arrive without a referer,
        # which is most of why a naive fetcher silently returns nothing.
        "Referer": origin + "/",
        "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            blob = resp.read(MAX_BYTES + 1)
        if "image" not in ctype and not blob[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0"):
            return None, f"content-type {ctype or 'unknown'}"
        if len(blob) > MAX_BYTES:
            return None, "over size cap"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:
        return None, type(exc).__name__

    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "undecodable"
    cv2.imwrite(str(path), img)
    return img, ""


def download(hits: list[dict], out_dir: Path, min_width: int = 900,
             timeout: float = 12.0, workers: int = 8) -> tuple[list[Candidate], dict[str, int]]:
    """Fetch candidates concurrently, rejecting anything that cannot be a capture.

    Returns the candidates plus a tally of why fetches failed - without that
    tally a run that silently returns nothing is impossible to diagnose.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [h.get("image") or "" for h in hits]
    urls = [u for u in urls if u]

    results: dict[str, tuple[np.ndarray | None, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, u, out_dir, timeout): u for u in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                results[url] = fut.result()
            except Exception as exc:
                results[url] = (None, type(exc).__name__)

    errors: dict[str, int] = {}
    cands: list[Candidate] = []
    for url in urls:
        img, err = results.get(url, (None, "missing"))
        if img is None:
            errors[err] = errors.get(err, 0) + 1
            continue
        h, w = img.shape[:2]
        c = Candidate(url=url, path=out_dir / _cache_name(url), width=w, height=h)
        if w < min_width:
            c.kept, c.reason = False, f"too small ({w}x{h})"
        elif not _ratio_ok(w, h):
            c.kept, c.reason = False, f"aspect {w / max(h, 1):.2f} not a display ratio"
        cands.append(c)
    return cands, errors


def normalise(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Grey, resized to the common analysis space, exposure-normalised.

    Screenshots come from different monitors, codecs and capture tools, so
    absolute brightness carries no signal; only structure does.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-6 else g


def edge_signature(norm: np.ndarray) -> np.ndarray:
    """Unit-norm gradient magnitude - structure that survives recolouring."""
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy).ravel()
    mag -= mag.mean()
    n = np.linalg.norm(mag)
    return mag / n if n > 1e-6 else mag


def dedupe(images: list[np.ndarray], similarity: float = 0.92) -> list[int]:
    """Indices of visually distinct images, keeping the first of each duplicate.

    Image search returns the same screenshot repeatedly, often re-encoded or
    colour-graded differently by each host. Duplicates are poison here:
    consensus measures what *varies* between images, so two copies of one
    screenshot make the whole frame look invariant and the ROI collapses to
    the entire screen.

    Comparison is on the exposure-normalised edge signature rather than a
    difference hash, because a hash of a regraded copy drifts far enough to
    look like a genuinely different image.
    """
    keep: list[int] = []
    sigs: list[np.ndarray] = []
    for i, im in enumerate(images):
        sig = edge_signature(normalise(im, (384, 216)))
        if any(float(np.dot(sig, s)) >= similarity for s in sigs):
            continue
        sigs.append(sig)
        keep.append(i)
    return keep


def robust_select(images: list[np.ndarray], size: tuple[int, int] = (384, 216),
                  trim: float = 0.25, iters: int = 4, min_keep: int = 3) -> list[int]:
    """Iteratively discard the images that agree least with the majority.

    Clustering on global edge correlation does not work for this: two genuine
    screenshots of the same event share only their interface, which may be a
    quarter of the frame, so wildly different backgrounds swamp the signal and
    real matches fail to cluster.

    Trimmed consensus sidesteps that. Take the median image, find where the
    set agrees, score each image by how well it matches the median *there*,
    drop the worst, repeat. The majority look wins without anyone having to
    say what it looks like.
    """
    norms = [normalise(im, size) for im in images]
    idx = list(range(len(norms)))
    if len(idx) <= min_keep:
        return idx

    for _ in range(iters):
        stack = np.stack([norms[i] for i in idx])
        med = np.median(stack, axis=0)
        spread = stack.std(axis=0)
        agree = spread <= np.quantile(spread, 0.40)
        if not agree.any():
            break
        scores = np.array([-np.abs(norms[i][agree] - med[agree]).mean() for i in idx])
        keep_n = max(min_keep, int(round(len(idx) * (1.0 - trim))))
        order = np.argsort(scores)[::-1][:keep_n]
        new_idx = sorted(idx[o] for o in order)
        if new_idx == idx:
            break
        idx = new_idx
    return idx


def cluster_by_structure(sigs: list[np.ndarray], thresh: float = 0.35) -> list[list[int]]:
    """Group images whose edge structure correlates.

    Two screenshots of the same event share their interface chrome and so
    correlate; an unrelated article photo or share card correlates with
    nothing. Keeping the largest cluster is therefore an automatic filter for
    exactly the property we care about.
    """
    if not sigs:
        return []
    m = np.stack(sigs)
    sim = m @ m.T
    np.fill_diagonal(sim, 0.0)

    unassigned = set(range(len(sigs)))
    clusters: list[list[int]] = []
    while unassigned:
        best_seed, best_members = None, []
        for i in unassigned:
            members = [j for j in unassigned if j == i or sim[i, j] >= thresh]
            if len(members) > len(best_members):
                best_seed, best_members = i, members
        if best_seed is None:
            break
        clusters.append(sorted(best_members))
        unassigned -= set(best_members)
    clusters.sort(key=len, reverse=True)
    return clusters


def consensus(
    images: list[np.ndarray],
    size: tuple[int, int] = (384, 216),
    max_std: float = 0.42,
    structure_q: float = 0.70,
    min_area_frac: float = 0.0015,
) -> ConsensusResult:
    """Per-pixel agreement across images: low variance marks the interface."""
    norms = [normalise(im, size) for im in images]
    stack = np.stack(norms)                      # (N, H, W), exposure-normalised
    spread = stack.std(axis=0)

    # Median rather than mean: one bad image should not drag the template.
    median = np.median(stack, axis=0)
    template = cv2.normalize(median, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Agreement alone is not enough. A full-screen tint - Minecraft's red death
    # wash, say - is invariant across every screenshot while carrying no
    # structure to match against, so it would match any dark scene. What makes
    # a region useful is agreeing *and* containing edges: text, borders,
    # buttons. Requiring local gradient energy keeps the chrome and discards
    # the wash.
    gx = cv2.Sobel(median, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(median, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (5, 5))
    structured = energy >= np.quantile(energy, structure_q)

    invariant = (spread <= max_std) & structured

    w, h = size
    from scipy.ndimage import label as cc_label
    lab, n = cc_label(invariant)
    min_area = max(16, int(min_area_frac * w * h))
    rois = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < min_area:
            continue
        rois.append((float(xs.min()) / w, float(ys.min()) / h,
                     float(xs.max() + 1) / w, float(ys.max() + 1) / h))
    rois.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)

    return ConsensusResult(size=size, invariant=invariant, template=template,
                           rois_norm=rois[:8], n_images=len(images))


SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3)


def structure_mask(template: np.ndarray, q: float = 0.70) -> np.ndarray:
    """Where a reference carries edges worth matching against."""
    t = template.astype(np.float32)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (5, 5))
    return energy >= np.quantile(energy, q)


def _rescale_about_centre(img: np.ndarray, s: float) -> np.ndarray:
    """Game UI scales about the screen centre, so the sweep does too."""
    if s == 1.0:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, s)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def build_bank(images: list[np.ndarray], size: tuple[int, int] = (384, 216)) -> list[dict]:
    """Keep the chosen images as individual references, not an average.

    Measured on 12 held-out Minecraft death screens: a bank scored by best
    single match separated 10 of 12 from the negatives, where the averaged
    consensus template managed 4. Averaging screenshots whose UI sits at
    different scales blurs precisely the chrome being matched.
    """
    bank = []
    for im in images:
        norm = normalise(im, size)
        tpl = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bank.append({"template": tpl, "mask": structure_mask(tpl)})
    return bank


def score_against_bank(gray: np.ndarray, bank: list[dict],
                       size: tuple[int, int] = (384, 216)) -> tuple[float, int]:
    """Best edge-structure correlation over every reference and UI scale."""
    best, best_i = -1.0, -1
    cache = {s: normalise(_rescale_about_centre(gray, s), size) for s in SCALES}
    for i, ref in enumerate(bank):
        mask = ref["mask"]
        tv = ref["template"][mask].astype(np.float32)
        tv = (tv - tv.mean()) / max(tv.std(), 1e-6)
        for s in SCALES:
            cv_ = cache[s][mask]
            cv_ = (cv_ - cv_.mean()) / max(cv_.std(), 1e-6)
            v = float(np.dot(cv_, tv) / cv_.size)
            if v > best:
                best, best_i = v, i
    return best, best_i


def scan_footage(gray_stack: np.ndarray, bank: list[dict],
                 min_score: float = 0.32) -> tuple[np.ndarray, int]:
    """Score every sampled frame against the bank.

    This only has to find *one* true hit. Once a frame in the user's own
    capture is confirmed, the detection template is re-derived from it at the
    user's real resolution and UI scale, and cheap correlation takes over for
    the full scan - which is both faster and far more accurate than anything
    a stranger's screenshot can support.
    """
    scores = np.empty(len(gray_stack), dtype=np.float32)
    for i, frame in enumerate(gray_stack):
        scores[i], _ = score_against_bank(frame, bank)
    best = int(np.argmax(scores))
    return scores, best if scores[best] >= min_score else -1


def locate_in_footage(
    gray_stack: np.ndarray,
    result: ConsensusResult,
    roi_norm: tuple[float, float, float, float],
    min_score: float = 0.30,
) -> tuple[np.ndarray, int]:
    """Score every sampled frame against the consensus template, inside the ROI.

    Matching is on edge structure, not raw intensity, because the web template
    and the user's capture differ in colour grading, UI scale and compression.
    This only has to be good enough to find *one* true hit - after that the
    template is re-derived from the user's own frames and plain correlation
    takes over.
    """
    t, h, w = gray_stack.shape
    x0, y0, x1, y1 = roi_norm
    tw, th = result.size

    tpl = result.template[int(y0 * th):max(int(y0 * th) + 1, int(y1 * th)),
                          int(x0 * tw):max(int(x0 * tw) + 1, int(x1 * tw))]
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = max(px0 + 1, int(x1 * w)), max(py0 + 1, int(y1 * h))
    crop_shape = (px1 - px0, py1 - py0)
    tpl = cv2.resize(tpl, crop_shape, interpolation=cv2.INTER_AREA).astype(np.float32)
    tpl_sig = edge_signature((tpl - tpl.mean()) / max(tpl.std(), 1e-6))

    scores = np.empty(t, dtype=np.float32)
    for i in range(t):
        patch = gray_stack[i, py0:py1, px0:px1].astype(np.float32)
        patch = (patch - patch.mean()) / max(patch.std(), 1e-6)
        scores[i] = float(np.dot(edge_signature(patch), tpl_sig))
    best = int(np.argmax(scores))
    return scores, best if scores[best] >= min_score else -1


def contact_sheet(cands: list[Candidate], out_path: Path, cols: int = 5,
                  cell: tuple[int, int] = (256, 144)) -> None:
    """Index card of every fetched image, so a human can pick the keepers."""
    shown = [c for c in cands if c.path.exists()]
    if not shown:
        return
    rows = (len(shown) + cols - 1) // cols
    cw, ch = cell
    pad, label_h = 8, 20
    canvas = np.full((rows * (ch + label_h + pad) + pad, cols * (cw + pad) + pad, 3),
                     (24, 22, 20), np.uint8)
    for n, c in enumerate(shown):
        img = cv2.imread(str(c.path))
        if img is None:
            continue
        r, col = divmod(n, cols)
        x = pad + col * (cw + pad)
        y = pad + r * (ch + label_h + pad)
        canvas[y:y + ch, x:x + cw] = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        tag = f"[{c.index if c.index >= 0 else n}] {c.width}x{c.height}" + (f" c{c.cluster}" if c.cluster >= 0 else "")
        color = (120, 220, 140) if c.kept else (110, 110, 200)
        cv2.rectangle(canvas, (x, y), (x + cw, y + ch), color, 2)
        cv2.putText(canvas, tag, (x + 2, y + ch + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def save(result: ConsensusResult, cands: list[Candidate], game: str, event: str,
         out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "consensus_template.png"), result.template)
    cv2.imwrite(str(out_dir / "invariant_mask.png"),
                (result.invariant.astype(np.uint8) * 255))
    overlay = cv2.cvtColor(result.template, cv2.COLOR_GRAY2BGR)
    tint = overlay.copy()
    tint[result.invariant] = (90, 230, 120)
    overlay = cv2.addWeighted(tint, 0.5, overlay, 0.5, 0)
    for r in result.rois_norm[:6]:
        h, w = result.invariant.shape
        cv2.rectangle(overlay, (int(r[0] * w), int(r[1] * h)),
                      (int(r[2] * w), int(r[3] * h)), (90, 180, 255), 1)
    cv2.imwrite(str(out_dir / "consensus_overlay.png"),
                cv2.resize(overlay, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST))
    payload = {
        "game": game,
        "event": event,
        "consensus": result.to_dict(),
        "sources": [
            {"url": c.url, "size": [c.width, c.height], "kept": c.kept,
             "reason": c.reason, "cluster": c.cluster}
            for c in cands
        ],
        "note": "rois_norm locate and name the element; the detection template "
                "must be re-derived from the user's own footage",
    }
    (out_dir / "lookup.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
