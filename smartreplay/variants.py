"""Cluster each event's occurrences into visual variants, and pick exemplars.

One HUD element usually has several looks: a death banner names a different
killer each time, a kill feed shows a different weapon icon. Those variants
matter later, because a compiled detector should score against a handful of
exemplars and take the best match rather than against one blurred average.

Occurrences are compared with a difference hash, which is cheap and ignores
overall brightness. Exemplar imagery is re-read from the source video at full
resolution, so the contact sheets show what the element actually looks like
rather than a 384-pixel-wide thumbnail of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VariantParams:
    hamming: int = 10        # dhash distance within which two looks are the same
    max_exemplars: int = 6
    pad: float = 0.06        # fraction of ROI size added as context on each side


@dataclass
class Variant:
    members: list[int]            # indices into group.occurrences
    representative: int           # index into group.occurrences
    phash: int


@dataclass
class GroupVariants:
    gid: int
    variants: list[Variant]
    exemplars: list[tuple[int, np.ndarray]] = field(repr=False, default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gid": self.gid,
            "n_variants": len(self.variants),
            "variants": [
                {
                    "n_members": len(v.members),
                    "representative_occurrence": v.representative,
                    "phash": f"{v.phash:016x}",
                }
                for v in self.variants
            ],
        }


def dhash(img: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of a BGR or grayscale crop."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.ravel():
        bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _padded_roi_norm(roi_norm, pad: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = roi_norm
    dx = (x1 - x0) * pad
    dy = (y1 - y0) * pad
    return (max(0.0, x0 - dx), max(0.0, y0 - dy), min(1.0, x1 + dx), min(1.0, y1 + dy))


def crop_norm(frame: np.ndarray, roi_norm) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi_norm
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = max(px0 + 1, int(x1 * w)), max(py0 + 1, int(y1 * h))
    return frame[py0:py1, px0:px1]


class FullResReader:
    """Pulls full-resolution frames at given timestamps, with graceful failure.

    Seeking by millisecond is approximate in some containers, which is fine:
    an event holds for seconds, so landing a frame or two off still shows the
    element. If seeking fails entirely the caller falls back to the thumbnail
    stack.
    """

    def __init__(self, path: str, crop: tuple[int, int, int, int] | None = None):
        self.cap = cv2.VideoCapture(path)
        self.ok = self.cap.isOpened()
        self.crop = crop

    def at(self, seconds: float) -> np.ndarray | None:
        if not self.ok:
            return None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        got, frame = self.cap.read()
        if not got:
            return None
        if self.crop:
            x0, y0, x1, y1 = self.crop
            frame = frame[y0:y1, x0:x1]
        return frame

    def close(self) -> None:
        if self.ok:
            self.cap.release()


def analyse_group(
    group,
    color: np.ndarray,
    reader: FullResReader | None = None,
    params: VariantParams | None = None,
) -> GroupVariants:
    p = params or VariantParams()
    roi = _padded_roi_norm(group.roi_norm, p.pad)

    hashes: list[int] = []
    for occ in group.occurrences:
        mid = min(len(color) - 1, (occ.start_idx + occ.end_idx) // 2)
        hashes.append(dhash(crop_norm(color[mid], roi)))

    # Greedy clustering: the most-populated look becomes a variant, and every
    # occurrence within `hamming` of it joins. Repeat on what is left.
    unassigned = set(range(len(hashes)))
    variants: list[Variant] = []
    while unassigned:
        best_seed, best_members = None, []
        for i in unassigned:
            members = [j for j in unassigned if _hamming(hashes[i], hashes[j]) <= p.hamming]
            if len(members) > len(best_members):
                best_seed, best_members = i, members
        if best_seed is None:
            break
        variants.append(
            Variant(members=sorted(best_members), representative=best_seed, phash=hashes[best_seed])
        )
        unassigned -= set(best_members)

    variants.sort(key=lambda v: len(v.members), reverse=True)

    exemplars: list[tuple[int, np.ndarray]] = []
    for v in variants[: p.max_exemplars]:
        occ = group.occurrences[v.representative]
        mid_s = occ.start_s + occ.duration_s / 2.0
        img = reader.at(mid_s) if reader else None
        if img is not None and img.size:
            exemplars.append((v.representative, crop_norm(img, roi)))
        else:
            mid = min(len(color) - 1, (occ.start_idx + occ.end_idx) // 2)
            exemplars.append((v.representative, crop_norm(color[mid], roi)))

    return GroupVariants(gid=group.gid, variants=variants, exemplars=exemplars)
