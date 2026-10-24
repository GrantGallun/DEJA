"""Fulfil a free-form replay request by greedy, cached, cheapest-first resolution.

Every request has the same shape, whatever the game or wording:

    ANCHOR   the kind of moment           death, kill, round_end, item_pickup
    FILTERS  conditions on that moment    cause = fire, weapon = Operator, won = yes
    WINDOW   how much to clip             5 s before, 1 s after

One fast model call (llm.plan) turns the words into that structure. From then
on it is plain code, and each piece is resolved by a ladder that tries the
cheapest evidence first and stops at the first rung that gives an answer:

    anchor   1 a group already labelled with this concept      free
             2 label the recording's groups once (1 text call)   cached per recording
    filter   1 on-screen text: the plan's phrases vs. OCR       free (OCR already indexed)
             2 a verified image detector for the concept         cheap, cached per game
             3 vision check of the frames around the moment      1 call each, cached

The rungs give three-valued answers - yes, no, or unknown - and only
"unknown" moves on to the next, costlier rung. Text gives "no" only when it
positively read something else (a different death message), never because it
read nothing; that is what the plan's anchor_text_terms are for.

Every answer is written back to memory: group labels, per-moment checks, and
the exact on-screen phrases that vision confirmed. So a game gets cheaper with
use - the second "burned to death" request costs zero model calls, and a
"fire" phrase learned from one vision check answers the next one by text.

Nothing is silently trusted. Each result says which rung decided it; moments
nobody could decide are returned as "maybe", not dropped or guessed.
"""

from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from .llm import LLM


# ----------------------------------------------------------------------- spec

@dataclass
class Filter:
    concept: str
    description: str
    include: bool = True
    text_terms: list[str] = field(default_factory=list)
    visual: bool = True


@dataclass
class RequestSpec:
    anchor: str
    filters: list[Filter]
    anchor_text_terms: list[str] = field(default_factory=list)
    before_s: float = 5.0
    after_s: float = 1.0
    limit: int | None = None
    ambiguity: str = ""

    @classmethod
    def from_plan(cls, d: dict) -> "RequestSpec":
        return cls(
            anchor=_slug(d["anchor"]),
            filters=[Filter(concept=_slug(f["concept"]), description=f["description"],
                            include=bool(f.get("include", True)),
                            text_terms=list(f.get("text_terms") or []),
                            visual=bool(f.get("visual", True))) for f in d.get("filters") or []],
            anchor_text_terms=list(d.get("anchor_text_terms") or []),
            before_s=float(d.get("before_s", 5.0)), after_s=float(d.get("after_s", 1.0)),
            limit=d.get("limit"), ambiguity=d.get("ambiguity", ""),
        )


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# --------------------------------------------------------------------- memory

class Memory:
    """What has been learned about a game and a recording, persisted as JSON.

    game file:       learned on-screen phrases per concept, verified detectors
    recording file:  group labels, per-moment check results
    """

    def __init__(self, game_path: Path, rec_path: Path):
        self.game_path, self.rec_path = Path(game_path), Path(rec_path)
        self.game = self._load(self.game_path, {"terms": {}, "detectors": {}})
        self.rec = self._load(self.rec_path, {"labels": {}, "checks": {}})

    @staticmethod
    def _load(path: Path, default: dict) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads(json.dumps(default))

    def save(self) -> None:
        for path, data in ((self.game_path, self.game), (self.rec_path, self.rec)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def terms(self, concept: str) -> list[str]:
        return self.game["terms"].get(concept, [])

    def learn_terms(self, concept: str, phrases: list[str]) -> None:
        have = self.game["terms"].setdefault(concept, [])
        for p in phrases:
            p = p.strip()
            if p and p.lower() not in (h.lower() for h in have):
                have.append(p)


# ------------------------------------------------------------- text matching

def phrase_in(phrase: str, text: str, threshold: float = 0.82) -> bool:
    """Fuzzy containment that tolerates OCR misreads ("suin" for "swim").

    Slides a window of the phrase's word length over the text and compares
    with a character-level ratio, so word order must match but a letter or
    two may be wrong.
    """
    p = re.findall(r"[a-z0-9]+", phrase.lower())
    t = re.findall(r"[a-z0-9]+", text.lower())
    if not p or not t:
        return False
    target = " ".join(p)
    n = len(p)
    for k in (n, n - 1, n + 1):
        if k <= 0:
            continue
        for i in range(0, max(1, len(t) - k + 1)):
            window = " ".join(t[i:i + k])
            if difflib.SequenceMatcher(None, target, window).ratio() >= threshold:
                return True
    return False


def text_verdict(lines: list[str], yes_terms: list[str], other_terms: list[str]) -> tuple[str, str]:
    """yes if a filter phrase is on screen; no if a *different* anchor phrase
    is; unknown if neither was read."""
    blob = " | ".join(lines)
    for t in yes_terms:
        if phrase_in(t, blob):
            return "yes", f'text "{t}"'
    for t in other_terms:
        if phrase_in(t, blob):
            return "no", f'text "{t}" (a different case)'
    return "unknown", "no relevant text read"


# ------------------------------------------------------------------- results

@dataclass
class Moment:
    id: str
    t: float
    verdict: str                 # yes | no | maybe
    evidence: list[str] = field(default_factory=list)


@dataclass
class Fulfilment:
    spec: RequestSpec
    anchor_group: int | None
    anchor_source: str
    moments: list[Moment]
    model_calls: int
    notes: list[str] = field(default_factory=list)

    @property
    def matches(self) -> list[Moment]:
        return [m for m in self.moments if m.verdict == "yes"]

    @property
    def maybes(self) -> list[Moment]:
        return [m for m in self.moments if m.verdict == "maybe"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_matches"], d["n_maybe"] = len(self.matches), len(self.maybes)
        return d


# ----------------------------------------------------------------- the engine

class Fulfiller:
    def __init__(self, index: dict, game: str, llm: LLM, memory: Memory,
                 max_checks: int = 40, workers: int = 6):
        self.index, self.game, self.llm, self.mem = index, game, llm, memory
        self.max_checks, self.workers = max_checks, workers
        self.calls = 0

    # anchor ---------------------------------------------------------------

    def _group_summaries(self) -> list[dict]:
        by_group: dict[int, list[dict]] = {}
        for o in self.index["occurrences"]:
            by_group.setdefault(o["group"], []).append(o)
        out = []
        for g in self.index["groups"]:
            occ = by_group.get(g["group"], [])
            sample = [line for o in occ[:: max(1, len(occ) // 6)][:6] for line in o["text"][:3]]
            out.append({"group": g["group"], "roi": [round(v, 2) for v in g["roi"]],
                        "count": g["n"], "mean_duration_s": round(g["mean_duration"], 1),
                        "text_samples": sample[:12]})
        return out

    def resolve_anchor(self, concept: str) -> tuple[int | None, str]:
        labels = self.mem.rec["labels"]
        if not labels:
            self.calls += 1
            res = self.llm.label_groups(self.game, self._group_summaries())
            for lab in res.get("labels", []):
                labels[str(lab["group"])] = {"concept": _slug(lab["concept"]),
                                             "confidence": lab.get("confidence", 0),
                                             "why": lab.get("why", "")}
            source = "labelled this recording's groups (1 model call)"
        else:
            source = "group labels from memory"
        cands = [(int(g), v) for g, v in labels.items() if v["concept"] == concept]
        if not cands:
            return None, f"no discovered group is a '{concept}' ({source})"
        g, _ = max(cands, key=lambda c: c[1].get("confidence", 0))
        return g, source

    # filters --------------------------------------------------------------

    def _frames(self, t: float, offsets=(-0.5, 0.5, 1.5)) -> list:
        cap = cv2.VideoCapture(self.index["video"])
        frames = []
        for d in offsets:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + d) * 1000.0)
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        return frames

    def _vision(self, occ: dict, flt: Filter) -> tuple[str, str]:
        key = f"{occ['id']}|{flt.concept}"
        cached = self.mem.rec["checks"].get(key)
        if cached:
            return cached["answer"], f"vision (memory): {cached['why']}"
        frames = self._frames(occ["t"])
        if not frames:
            return "unknown", "could not read frames"
        q = (f"These are consecutive screenshots from {self.game} around one '{self.spec.anchor}' "
             f"moment. Question: {flt.description} Answer yes, no, or unsure.")
        self.calls += 1
        res = self.llm.check(frames, q)
        self.mem.rec["checks"][key] = {"answer": res.get("answer", "unsure"), "why": res.get("why", "")}
        return res.get("answer", "unsure"), f"vision: {res.get('why', '')}"

    def fulfil(self, spec: RequestSpec) -> Fulfilment:
        self.spec = spec
        notes = [f"ambiguity: {spec.ambiguity}"] if spec.ambiguity else []
        gid, source = self.resolve_anchor(spec.anchor)
        if gid is None:
            self.mem.save()
            return Fulfilment(spec, None, source, [], self.calls, notes + [
                "next rung (not yet automatic): build a detector for the anchor from verified "
                "exemplars - see verify_images.py"])

        occs = [o for o in self.index["occurrences"] if o["group"] == gid]
        occs = _collapse(occs)
        for f in spec.filters:
            # Memory first: phrases vision confirmed in earlier requests.
            f.text_terms = list(dict.fromkeys(f.text_terms + self.mem.terms(f.concept)))
        other = spec.anchor_text_terms

        state: dict[str, Moment] = {o["id"]: Moment(o["id"], o["t"], "yes") for o in occs}
        pending: list[tuple[dict, Filter]] = []
        for o in occs:
            for f in spec.filters:
                v, why = text_verdict(o["text"], f.text_terms,
                                      [t for t in other if t not in f.text_terms])
                m = state[o["id"]]
                m.evidence.append(f"{f.concept}: {why}")
                if v == "unknown":
                    pending.append((o, f))
                elif (v == "yes") != f.include:
                    m.verdict = "no"

        # Vision only for moments still alive and still undecided.
        pending = [(o, f) for o, f in pending if state[o["id"]].verdict != "no" and f.visual]
        if len(pending) > self.max_checks:
            notes.append(f"{len(pending) - self.max_checks} moment(s) left undecided (check budget)")
        todo, rest = pending[:self.max_checks], pending[self.max_checks:]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            answers = list(pool.map(lambda of: self._vision(*of), todo))
        for (o, f), (ans, why) in zip(todo, answers):
            m = state[o["id"]]
            m.evidence.append(f"{f.concept}: {why}")
            if ans == "unsure":
                m.verdict = "maybe" if m.verdict != "no" else "no"
            elif (ans == "yes") != f.include:
                m.verdict = "no"
            elif ans == "yes" and f.include:
                # Learn the wording this game used, so text answers next time.
                self.mem.learn_terms(f.concept, [ln for ln in o["text"][:2]
                                                 if any(phrase_in(t, ln) for t in other)])
        for o, f in rest + [(o, f) for o, f in pending if not f.visual]:
            if state[o["id"]].verdict == "yes":
                state[o["id"]].verdict = "maybe"

        moments = sorted(state.values(), key=lambda m: m.t)
        if spec.limit:
            keep = {m.id for m in [m for m in moments if m.verdict == "yes"][-spec.limit:]}
            moments = [m for m in moments if m.verdict != "yes" or m.id in keep]
        self.mem.save()
        return Fulfilment(spec, gid, source, moments, self.calls, notes)


def _collapse(occs: list[dict], window: float = 8.0) -> list[dict]:
    """One real moment can register as several occurrences seconds apart."""
    out: list[dict] = []
    for o in sorted(occs, key=lambda o: o["t"]):
        if out and o["t"] - out[-1]["t"] <= window:
            out[-1] = dict(out[-1], text=list(dict.fromkeys(out[-1]["text"] + o["text"])))
            continue
        out.append(dict(o))
    return out


def run(out_dir: str | Path, query: str, game: str, llm: LLM, profiles: str | Path = "profiles",
        reel: bool = True) -> Fulfilment:
    """End to end: plan -> resolve -> (optionally) cut the reel."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    mem = Memory(Path(profiles) / f"{_slug(game)}.json", out_dir / "memory.json")
    catalog = sorted({v["concept"] for v in mem.rec["labels"].values()})
    spec = RequestSpec.from_plan(llm.plan(query, game, catalog))
    f = Fulfiller(index, game, llm, mem)
    f.calls = 1  # the plan
    result = f.fulfil(spec)
    if reel and result.matches:
        from .query import cut_reel
        moments = [{"id": m.id, "t": m.t} for m in result.matches + result.maybes]
        reasons = {m.id: ("MAYBE " if m.verdict == "maybe" else "") + "; ".join(m.evidence)[:80]
                   for m in result.matches + result.maybes}
        slug = _slug(query)[:40]
        cut_reel(index["video"], moments, reasons, out_dir / f"reel_{slug}.mp4", query,
                 spec.before_s, spec.after_s)
    (out_dir / f"request_{_slug(query)[:40]}.json").write_text(
        json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    return result


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fulfil a free-form replay request.")
    ap.add_argument("out_dir", help="discovery output dir with index.json (python -m smartreplay.index)")
    ap.add_argument("query")
    ap.add_argument("--game", required=True)
    ap.add_argument("--profiles", default="profiles")
    ap.add_argument("--no-reel", action="store_true")
    args = ap.parse_args(argv)

    from .llm import default_llm

    res = run(args.out_dir, args.query, args.game, default_llm(), args.profiles, reel=not args.no_reel)
    print(f"anchor: group {res.anchor_group} ({res.anchor_source})")
    for m in res.moments:
        if m.verdict != "no":
            print(f"  {m.verdict:5s} {m.t:8.1f}s  {'; '.join(m.evidence)[:100]}")
    print(f"{len(res.matches)} match(es), {len(res.maybes)} maybe, {res.model_calls} model call(s)")
    for n in res.notes:
        print(f"note: {n}")
    return 0 if res.matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
