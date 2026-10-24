"""Leave-one-out test of a lookup-derived template.

The question that decides whether web-image lookup is usable: can a template
built from a handful of strangers' screenshots recognise a death screen it has
never seen, at a different resolution, and *not* fire on ordinary gameplay?

Images are hand-labelled once (by index, from candidates.png). For each
positive, a consensus is built from the *other* positives only, and the
held-out image is scored against it - alongside every negative. If the
positives separate from the negatives, lookup works as a bootstrap.

    python tools/lookup_eval.py out/lookup/minecraft-death-screen \
        --positives 0,1,2,3,5,10,11,16 --negatives 15,18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartreplay import lookup as lk

SIZE = (384, 216)
SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3)


def structure_mask(template: np.ndarray, q: float = 0.70) -> np.ndarray:
    t = template.astype(np.float32)
    gx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (5, 5))
    return energy >= np.quantile(energy, q)


def scaled(img: np.ndarray, s: float) -> np.ndarray:
    """Rescale about the frame centre, which is how game UI scale behaves."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, s)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def score(img: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    """Best edge-structure correlation over a sweep of UI scales."""
    tpl = template.astype(np.float32)
    tv = tpl[mask]
    tv = (tv - tv.mean()) / max(tv.std(), 1e-6)

    best = -1.0
    for s in SCALES:
        cand = lk.normalise(scaled(img, s) if s != 1.0 else img, SIZE)
        cv_ = cand[mask]
        cv_ = (cv_ - cv_.mean()) / max(cv_.std(), 1e-6)
        best = max(best, float(np.dot(cv_, tv) / cv_.size))
    return best


def load(out_dir: Path) -> dict[int, np.ndarray]:
    """Index images by sorted filename, which is what the contact sheet shows."""
    imgs: dict[int, np.ndarray] = {}
    for i, p in enumerate(sorted((out_dir / "images").glob("*.png"))):
        im = cv2.imread(str(p))
        if im is not None:
            imgs[i] = im
    return imgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--positives", required=True)
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--mode", choices=("consensus", "bank"), default="bank")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    imgs = load(out_dir)
    pos = [int(i) for i in args.positives.split(",") if i.strip().isdigit()]
    neg = [int(i) for i in args.negatives.split(",") if i.strip().isdigit()]
    pos = [i for i in pos if i in imgs]
    neg = [i for i in neg if i in imgs]
    print(f"{len(imgs)} image(s) available; {len(pos)} positive(s), {len(neg)} negative(s)\n")
    if len(pos) < 3:
        print("need at least 3 positives for leave-one-out")
        return 1

    def against_bank(img, refs: list[int]) -> float:
        """Best match over individual references, rather than their average.

        Averaging screenshots whose UI sits at different scales blurs exactly
        the chrome being matched. Keeping them as a bank and taking the best
        single match avoids that, and is what a curated handful of images is
        actually for.
        """
        best = -1.0
        for r in refs:
            ref = lk.normalise(imgs[r], SIZE)
            ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            best = max(best, score(img, ref_u8, structure_mask(ref_u8)))
        return best

    held_scores, neg_scores = [], []
    for held in pos:
        others = [i for i in pos if i != held]
        if args.mode == "bank":
            s_held = against_bank(imgs[held], others)
            row_neg = [against_bank(imgs[n], others) for n in neg]
        else:
            res = lk.consensus([imgs[i] for i in others], size=SIZE)
            mask = structure_mask(res.template)
            s_held = score(imgs[held], res.template, mask)
            row_neg = [score(imgs[n], res.template, mask) for n in neg]
        held_scores.append(s_held)
        neg_scores.extend(row_neg)
        worst = max(row_neg) if row_neg else float("nan")
        flag = "ok " if row_neg and s_held > worst else "XX "
        print(f"  {flag} held-out [{held:>2}] = {s_held:+.4f}   "
              f"best negative = {worst:+.4f}")

    hp = np.array(held_scores)
    ng = np.array(neg_scores)
    print(f"\npositives  mean {hp.mean():+.4f}  min {hp.min():+.4f}")
    print(f"negatives  mean {ng.mean():+.4f}  max {ng.max():+.4f}")
    margin = hp.min() - ng.max()
    print(f"separation margin (min pos - max neg): {margin:+.4f}")
    if margin > 0:
        print("SEPARABLE - a single threshold distinguishes them")
    else:
        print("NOT SEPARABLE by a global threshold on these images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
