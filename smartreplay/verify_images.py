"""Turn fetched example images into a detector you can trust - or refuse to.

Web images of "a Minecraft death screen" are mostly right and partly junk:
memes, mods, other editions, articles, near-duplicates. Taking a pile of them
and seeing what most of them share is the right instinct, but measured on the
whole frame it fails: two real death screens share a small banner while their
backgrounds share nothing, so the background decides the vote. Four gates fix
that, each cheap and each catching what the one before cannot:

  1 shape     decodes, big enough, a screen aspect ratio, not a duplicate     free
  2 meaning   a fast vision model answers "is this <description>?" and
              boxes the element that shows it                                 1 call per image, cached
  3 agreement crops at those boxes are compared with each other; the element
              must look alike across images, so a "yes" on a modded or
              differently-styled screen is outvoted here                      free
  4 your data the surviving crops become a template bank, scored against
              frames from *your* recording: random frames are negatives
              (nearly every frame is one), held-out crops are positives. The
              threshold is set from the negatives, and recall is measured on
              positives the bank never saw                                    free

A detector that fails gate 4 - too few agreeing exemplars, or no threshold
that keeps false alarms down while catching most positives - is returned with
trusted=False, and the caller falls back to vision checks. That is the
practical meaning of "foolproof": not that every detector works, but that a
detector which doesn't work is never used silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .llm import LLM

SCREEN_RATIOS = (16 / 9, 16 / 10, 21 / 9, 4 / 3)


@dataclass
class Detector:
    concept: str
    box: list[float]                       # expected location, normalised [x0,y0,x1,y1]
    templates: list[np.ndarray] = field(repr=False, default_factory=list)  # edge maps
    threshold: float = 1.0
    trusted: bool = False
    report: dict = field(default_factory=dict)


def _edges(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return mag / (mag.max() + 1e-6)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _crop(img: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    a, b = int(max(0, (x0 - px) * w)), int(max(0, (y0 - py) * h))
    c, d = int(min(w, (x1 + px) * w)), int(min(h, (y1 + py) * h))
    return img[b:max(d, b + 1), a:max(c, a + 1)]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


# ----------------------------------------------------------------- gate 1

def shape_ok(img: np.ndarray, min_width: int = 640) -> tuple[bool, str]:
    h, w = img.shape[:2]
    if w < min_width:
        return False, f"too small ({w}x{h})"
    r = w / max(h, 1)
    if not any(abs(r - t) / t <= 0.05 for t in SCREEN_RATIOS):
        return False, f"aspect {r:.2f} is not a screen"
    return True, ""


def near_duplicates(imgs: list[np.ndarray], thresh: float = 0.92) -> set[int]:
    sigs, dup = [], set()
    for i, im in enumerate(imgs):
        s = cv2.resize(_edges(_gray(im)), (96, 54), interpolation=cv2.INTER_AREA).ravel()
        s = (s - s.mean()) / (s.std() + 1e-6)
        if any(float((s * t).mean()) >= thresh for t in sigs):
            dup.add(i)
        else:
            sigs.append(s)
    return dup


# ----------------------------------------------------------------- gate 3

def _pair_sim(tpl_img: np.ndarray, tpl_box, reg_img: np.ndarray, reg_box,
              scales=(0.8, 0.9, 1.0, 1.1, 1.25), width: int = 192) -> float:
    """How well the element in one image is found near the box in another.

    A template search over scales, because the same banner is drawn at a
    different size at each resolution and UI scale - comparing fixed-size
    crops directly would score a genuine match as a poor one.
    """
    t = _edges(_gray(_crop(tpl_img, tpl_box)))
    r = _edges(_gray(_crop(reg_img, reg_box, pad=0.2)))
    k = width / max(r.shape[1], 1)
    r = cv2.resize(r, (width, max(8, int(r.shape[0] * k))), interpolation=cv2.INTER_AREA)
    base = width / 1.4                       # template width when scales match (pad 0.2 each side)
    best = -1.0
    for sc in scales:
        tw = int(base * sc)
        th = max(6, int(t.shape[0] * tw / max(t.shape[1], 1)))
        if tw >= r.shape[1] or th >= r.shape[0] or tw < 8:
            continue
        tt = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        best = max(best, float(cv2.matchTemplate(r, tt, cv2.TM_CCOEFF_NORMED).max()))
    return best


def agreement(images: list[np.ndarray], boxes: list, min_gap: float = 0.06,
              min_sim: float = 0.3) -> tuple[list[int], np.ndarray]:
    """Indices of the largest group of exemplars whose boxed elements agree.

    "What triggers most of them", measured on the element only, so
    backgrounds cannot vote. Exemplars are clustered by pairwise similarity
    (average linkage) and cut at the biggest jump in merge distance - the
    natural break between "same element" and "something else". The largest
    cluster wins; a vision model's yes on a modded or other-edition screen
    usually lands in a smaller cluster and is dropped.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(images)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i, j] = _pair_sim(images[i], boxes[i], images[j], boxes[j])
    sim = (sim + sim.T) / 2
    if n < 3:
        return list(range(n)), sim
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    heights = z[:, 2]
    jumps = np.diff(heights)
    if jumps.size and jumps.max() >= min_gap:
        cut = heights[int(np.argmax(jumps))] + 1e-6
        labels = fcluster(z, t=cut, criterion="distance")
    else:
        labels = np.ones(n, dtype=int)
    best, best_key = None, None
    for c in set(labels):
        members = [i for i in range(n) if labels[i] == c]
        inner = sim[np.ix_(members, members)]
        mean_sim = (inner.sum() - len(members)) / max(1, len(members) * (len(members) - 1))
        key = (len(members), mean_sim)
        if best_key is None or key > best_key:
            best, best_key = members, key
    if best_key[1] < min_sim and len(best) > 1:
        return [], sim                      # nothing actually agrees
    return best, sim


# ----------------------------------------------------------------- gate 4

def score(frame: np.ndarray, det: Detector, margin: float = 0.08,
          scales=(0.8, 0.9, 1.0, 1.1, 1.25)) -> float:
    """Best edge correlation of any template near the expected location."""
    x0, y0, x1, y1 = det.box
    region = _crop(frame, [x0 - margin, y0 - margin, x1 + margin, y1 + margin])
    if region.size == 0:
        return -1.0
    h, w = frame.shape[:2]
    tw, th = int((x1 - x0) * w), int((y1 - y0) * h)
    er = _edges(_gray(region))
    best = -1.0
    for t in det.templates:
        for s in scales:
            sz = (max(8, int(tw * s)), max(8, int(th * s)))
            if sz[0] >= er.shape[1] or sz[1] >= er.shape[0]:
                continue
            tt = cv2.resize(t, sz, interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(er, tt, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(r.max()))
    return best


def calibrate(det: Detector, crops_full: list[np.ndarray], negatives: list[np.ndarray],
              max_fpr: float = 0.01, min_recall: float = 0.8, min_exemplars: int = 3) -> Detector:
    """Set the threshold from your own negatives; measure recall leave-one-out."""
    n = len(det.templates)
    if n < min_exemplars:
        det.trusted = False
        det.report.update(reason=f"only {n} agreeing exemplar(s); need {min_exemplars}")
        return det
    neg = np.array([score(f, det) for f in negatives]) if negatives else np.array([0.0])
    thr = float(np.quantile(neg, 1.0 - max_fpr)) + 0.02
    loo = []
    for i, img in enumerate(crops_full):
        held = Detector(det.concept, det.box, [t for k, t in enumerate(det.templates) if k != i])
        loo.append(score(img, held))
    loo = np.array(loo)
    recall = float((loo >= thr).mean()) if loo.size else 0.0
    det.threshold = thr
    det.trusted = recall >= min_recall and len(negatives) >= 20
    det.report.update(threshold=round(thr, 3), loo_recall=round(recall, 3),
                      negatives=len(negatives), neg_max=round(float(neg.max()), 3),
                      pos_scores=[round(float(v), 3) for v in loo],
                      reason="" if det.trusted else
                      ("too few negatives to calibrate" if len(negatives) < 20 else
                       f"leave-one-out recall {recall:.2f} < {min_recall}"))
    return det


# --------------------------------------------------------------- pipeline

def build_detector(images: list[np.ndarray], game: str, concept: str, description: str,
                   llm: LLM, negatives: list[np.ndarray]) -> Detector:
    report: dict = {"fetched": len(images), "rejected": {}}

    ok = []
    for i, im in enumerate(images):
        good, why = shape_ok(im)
        if good:
            ok.append(i)
        else:
            report["rejected"][i] = why
    dups = near_duplicates([images[i] for i in ok])
    for k in sorted(dups):
        report["rejected"][ok[k]] = "duplicate"
    ok = [i for k, i in enumerate(ok) if k not in dups]

    q = (f"Is this a screenshot of {game} showing: {description}? Answer yes only for the real "
         "game interface (not edited, not a meme, not a different game). If yes, box the on-screen "
         "element that shows it - the banner, text or icon, not the whole screen.")
    boxes: dict[int, list[float]] = {}
    for i in ok:
        res = llm.check([images[i]], q)
        if res.get("answer") == "yes" and res.get("box"):
            boxes[i] = [float(v) for v in res["box"]]
        else:
            report["rejected"][i] = f"vision: {res.get('answer')} - {res.get('why', '')}"[:120]

    ids = list(boxes)
    keep, _ = agreement([images[i] for i in ids], [boxes[i] for i in ids]) if ids else ([], None)
    for k, i in enumerate(ids):
        if k not in keep:
            report["rejected"][i] = "doesn't look like the other exemplars"
    ids = [ids[k] for k in keep]
    report["kept"] = ids

    if not ids:
        return Detector(concept, [0, 0, 1, 1], [], 1.0, False, dict(report, reason="no exemplars survived"))
    box = list(np.median(np.array([boxes[i] for i in ids]), axis=0))
    templates = [_edges(_gray(_crop(images[i], boxes[i]))) for i in ids]
    det = Detector(concept, [round(float(v), 4) for v in box], templates, report=report)
    return calibrate(det, [images[i] for i in ids], negatives)
