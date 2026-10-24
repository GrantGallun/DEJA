"""Command line entry point.

    python -m smartreplay.cli discover footage/cs2.mp4
    python -m smartreplay.cli batch footage/ --out out/

Each run leaves a self-contained directory: the sampled stacks, the pixel
classification, events.json, a draft profile with templates, a HUD map and one
contact sheet per discovered event.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
from pathlib import Path

from . import discovery, grouping, profile as profile_mod, report, sampling, variants

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv"}


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", default="out", help="output root (default: out)")
    ap.add_argument("--fps", type=float, default=2.0, help="analysis sample rate")
    ap.add_argument("--width", type=int, default=384, help="analysis width in pixels")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--trim-start", type=float, default=0.0)
    ap.add_argument("--trim-end", type=float, default=0.0)
    ap.add_argument("--tol", type=int, default=12, help="grey levels counted as at-the-mode")
    ap.add_argument("--min-event-s", type=float, default=1.0)
    ap.add_argument("--max-event-s", type=float, default=30.0)
    ap.add_argument("--min-occurrences", type=int, default=3)
    ap.add_argument("--min-pixels", type=int, default=10)
    ap.add_argument("--jaccard", type=float, default=0.65)
    ap.add_argument("--max-groups", type=int, default=24)
    ap.add_argument("--no-fullres", action="store_true",
                    help="skip re-reading the source for exemplar crops")
    ap.add_argument("--reuse-samples", action="store_true",
                    help="reuse an existing sampled stack if present")


def clip_name(video: Path) -> str:
    """Output directory name for a clip, unique among the videos beside it.

    The stem alone is ambiguous when match.avi and match.mp4 sit in the same
    folder, so any clip that shares its stem with a sibling video keeps its
    extension in the name. The rule depends only on the folder's contents, so
    discover and batch agree on where a clip's output lives.
    """
    try:
        siblings = [p for p in video.parent.iterdir()
                    if p != video and p.stem == video.stem and p.suffix.lower() in VIDEO_EXT]
    except OSError:
        siblings = []
    if siblings:
        return f"{video.stem}-{video.suffix.lstrip('.').lower()}"
    return video.stem


def run_one(video: Path, out_root: Path, args) -> dict:
    name = clip_name(video)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    params = sampling.sampling_params(args.fps, args.width, 192, args.max_seconds,
                                      args.trim_start, args.trim_end)
    reuse, why = (False, "")
    if args.reuse_samples:
        reuse, why = sampling.cache_matches(out_dir, video, params)
        if not reuse:
            print(f"  [sample]   not reusing: {why}")
    if reuse:
        print(f"  [sample]   reusing {out_dir / 'gray.npy'}")
    else:
        meta = sampling.sample_video(
            video, out_dir,
            sample_fps=args.fps, width=args.width,
            max_seconds=args.max_seconds,
            trim_start=args.trim_start, trim_end=args.trim_end,
        )
        print(f"  [sample]   {meta.n_frames} frames @ {meta.sample_fps:.2f}fps, "
              f"{meta.size[0]}x{meta.size[1]}, {meta.duration_s:.0f}s"
              + ("  (truncated)" if meta.truncated else ""))
    timings["sample_s"] = round(time.perf_counter() - t0, 2)

    gray, color, times, meta = sampling.load_samples(out_dir)

    t0 = time.perf_counter()
    result = discovery.discover(
        gray, meta["sample_fps"],
        discovery.DiscoveryParams(
            tol=args.tol,
            min_event_s=args.min_event_s,
            max_event_s=args.max_event_s,
            min_occurrences=args.min_occurrences,
        ),
    )
    discovery.save(result, out_dir)
    timings["discover_s"] = round(time.perf_counter() - t0, 2)
    s = result.summary()
    print(f"  [classify] {s['event_pixels']} event px, {s['static_pixels']} static px "
          f"({s['event_fraction'] * 100:.2f}% of frame)   {timings['discover_s']}s")

    t0 = time.perf_counter()
    groups = grouping.group_events(
        result.on, result.event_mask, times, meta["sample_fps"],
        grouping.GroupParams(
            jaccard=args.jaccard,
            min_pixels=args.min_pixels,
            min_occurrences=args.min_occurrences,
            min_event_s=args.min_event_s,
            max_groups=args.max_groups,
        ),
    )
    grouping.save(groups, out_dir)
    timings["group_s"] = round(time.perf_counter() - t0, 2)
    print(f"  [group]    {len(groups)} event group(s)   {timings['group_s']}s")

    t0 = time.perf_counter()
    reader = None
    if not args.no_fullres:
        reader = variants.FullResReader(str(video), crop=tuple(meta["crop"]))
    gvs = [variants.analyse_group(g, color, reader) for g in groups]
    if reader:
        reader.close()
    timings["variants_s"] = round(time.perf_counter() - t0, 2)

    report.write_reports(color, times, result, groups, gvs, out_dir)
    prof, templates = profile_mod.build(meta, result, groups, gvs, gray)
    profile_mod.save(prof, templates, out_dir)

    if groups:
        print(f"  {'gid':>3}  {'occ':>4}  {'mean':>6}  {'gap':>6}  {'var':>4}  {'px':>5}  roi_norm")
        for g, gv in zip(groups, gvs):
            rn = g.roi_norm
            print(f"  {g.gid:>3}  {len(g.occurrences):>4}  {g.mean_duration_s:>5.1f}s  "
                  f"{g.median_gap_s:>5.0f}s  {len(gv.variants):>4}  {g.n_pixels:>5}  "
                  f"({rn[0]:.2f},{rn[1]:.2f})-({rn[2]:.2f},{rn[3]:.2f})")
    else:
        print("  no recurring event HUD found - see notes in README.md")

    rect = prof["ui"]["suggested_player_rect"]
    print(f"  [profile]  {len(prof['detectors'])} detector(s), "
          f"{len(prof['ui']['hud_safe_zones'])} hud zone(s), "
          f"player rect ({rect[0]:.2f},{rect[1]:.2f})-({rect[2]:.2f},{rect[3]:.2f})")

    return {
        "name": name,
        "video": str(video),
        "n_frames": meta["n_frames"],
        "footage_s": round(meta["duration_s"], 1),
        "event_fraction": round(s["event_fraction"], 5),
        "groups": [
            {
                "gid": g.gid,
                "n_occurrences": len(g.occurrences),
                "mean_duration_s": round(g.mean_duration_s, 2),
                "n_variants": len(gv.variants),
                "roi_norm": [round(v, 4) for v in g.roi_norm],
            }
            for g, gv in zip(groups, gvs)
        ],
        "timings": timings,
        "out_dir": str(out_dir),
    }


def write_index(results: list[dict], out_root: Path) -> Path:
    lines = ["# Discovery results", ""]
    lines.append("| game | footage | event px | groups | detail |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['footage_s']:.0f}s | {r['event_fraction'] * 100:.2f}% | "
            f"{len(r['groups'])} | [{r['name']}/hud_map.png]({r['name']}/hud_map.png) |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['name']}")
        lines.append("")
        if not r["groups"]:
            lines.append("No recurring event HUD found.")
            lines.append("")
            continue
        lines.append("| gid | occurrences | mean duration | variants | roi_norm | sheet |")
        lines.append("|---|---|---|---|---|---|")
        for g in r["groups"]:
            rn = g["roi_norm"]
            sheet = f"{r['name']}/group_{g['gid']:02d}.png"
            lines.append(
                f"| {g['gid']} | {g['n_occurrences']} | {g['mean_duration_s']}s | "
                f"{g['n_variants']} | ({rn[0]:.2f},{rn[1]:.2f})-({rn[2]:.2f},{rn[3]:.2f}) | "
                f"[sheet]({sheet}) |"
            )
        lines.append("")
    path = out_root / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_root / "index.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def run_lookup(args) -> int:
    from . import lookup as lk

    out_dir = Path(args.out) / f"{args.game}-{args.event}".lower().replace(" ", "-")
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = lk.build_queries(args.game, args.event)
    print(f"queries: {queries}")
    hits = lk.search_images(queries, max_results=args.n)
    print(f"  {len(hits)} result(s)")

    cands, errors = lk.download(hits, out_dir / "images", min_width=args.min_width)
    if errors:
        print(f"  fetch failures: {errors}")
    # Every downloaded image gets a stable display index, which is what the
    # contact sheet prints and what --keep refers to. Later filtering only
    # flips `kept`, so those indices never shift under the user.
    shown = [c for c in cands if c.path.exists()]
    for n, c in enumerate(shown):
        c.index = n
    print(f"  {sum(c.kept for c in cands)}/{len(cands)} passed size and aspect filtering")
    for c in cands:
        if not c.kept:
            print(f"    rejected: {c.reason}")

    def usable():
        return [c for c in shown if c.kept]

    if len(usable()) < 2:
        print("  too few usable images; try --n higher or a different --event wording")
        lk.contact_sheet(cands, out_dir / "candidates.png")
        return 1

    loaded = {c.index: cv2.imread(str(c.path)) for c in usable()}
    live = usable()
    distinct = set(c.index for c in [live[i] for i in lk.dedupe([loaded[c.index] for c in live])])
    dropped = 0
    for c in live:
        if c.index not in distinct:
            c.kept, c.reason, dropped = False, "duplicate of an earlier image", dropped + 1
    if dropped:
        print(f"  {dropped} duplicate(s) dropped")

    live = usable()
    if args.keep:
        want = {int(i) for i in args.keep.split(",") if i.strip().isdigit()}
        chosen_idx = [c.index for c in shown if c.index in want and c.path.exists()]
        print(f"  using {len(chosen_idx)} manually selected image(s): {chosen_idx}")
    else:
        survivors = lk.robust_select([loaded[c.index] for c in live], trim=args.trim)
        chosen_idx = [live[s].index for s in survivors]
        for s in survivors:
            live[s].cluster = 0
        print(f"  trimmed consensus kept {len(chosen_idx)}/{len(live)}: {chosen_idx}")
    if len(chosen_idx) < 2:
        print("  need at least 2 agreeing images to form a consensus")
        lk.contact_sheet(cands, out_dir / "candidates.png")
        return 1

    for c in shown:
        c.kept = c.index in chosen_idx
    for i in chosen_idx:
        if i not in loaded:
            loaded[i] = cv2.imread(str(next(c.path for c in shown if c.index == i)))

    result = lk.consensus([loaded[i] for i in chosen_idx])
    lk.contact_sheet(cands, out_dir / "candidates.png")
    lk.save(result, cands, args.game, args.event, out_dir)

    print(f"  invariant {result.invariant.mean() * 100:.1f}% of frame, "
          f"{len(result.rois_norm)} region(s)")
    for i, r in enumerate(result.rois_norm[:4]):
        print(f"    roi {i}: ({r[0]:.2f},{r[1]:.2f})-({r[2]:.2f},{r[3]:.2f})")

    if args.footage:
        sample_dir = Path(args.sample_out) / clip_name(Path(args.footage))
        params = sampling.sampling_params(2.0, 384, 192, 600.0, 0.0, 0.0)
        if not sampling.cache_matches(sample_dir, args.footage, params)[0]:
            sampling.sample_video(args.footage, sample_dir, width=384)
        gray, _, times, meta = sampling.load_samples(sample_dir)
        if result.rois_norm:
            scores, best = lk.locate_in_footage(gray, result, result.rois_norm[0])
            if best >= 0:
                print(f"  located in footage at t={times[best]:.1f}s "
                      f"(score {scores[best]:.2f}) -> re-derive the template here")
            else:
                print(f"  no frame scored above threshold (best {scores.max():.2f}); "
                      f"the element may not occur in this clip")

    print(f"\nwrote {out_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartreplay", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    lu = sub.add_parser("lookup", help="bootstrap an event detector from web images")
    lu.add_argument("game")
    lu.add_argument("event", nargs="?", default="death screen")
    lu.add_argument("--n", type=int, default=40, help="images to fetch")
    lu.add_argument("--out", default="out/lookup")
    lu.add_argument("--min-width", type=int, default=900)
    lu.add_argument("--trim", type=float, default=0.25,
                    help="fraction of worst-agreeing images dropped per round")
    lu.add_argument("--keep", default=None,
                    help="comma-separated indices from candidates.png to use instead")
    lu.add_argument("--footage", default=None, help="try to locate the element in this clip")
    lu.add_argument("--sample-out", default="out")

    d = sub.add_parser("discover", help="run discovery on one video")
    d.add_argument("video")
    _add_common(d)

    b = sub.add_parser("batch", help="run discovery on every video in a folder")
    b.add_argument("folder")
    _add_common(b)

    args = ap.parse_args(argv)
    if args.cmd == "lookup":
        return run_lookup(args)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cmd == "discover":
        videos = [Path(args.video)]
    else:
        folder = Path(args.folder)
        videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
        if not videos:
            print(f"no videos in {folder} (looked for {sorted(VIDEO_EXT)})")
            return 1

    results = []
    for v in videos:
        print(f"\n{v.name}")
        try:
            results.append(run_one(v, out_root, args))
        except Exception as exc:  # one bad clip should not end a batch
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    if results:
        idx = write_index(results, out_root)
        print(f"\nwrote {idx}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
