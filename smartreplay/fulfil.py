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
    fact_key: str | None = None
    fact_values: list[str] = field(default_factory=list)
    null_is_no: bool = False


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
        spec = cls._parse(d)
        spec.filters = _merge_alternatives(spec.filters)
        return spec

    @classmethod
    def _parse(cls, d: dict) -> "RequestSpec":
        return cls(
            anchor=_slug(d["anchor"]),
            filters=[Filter(concept=_slug(f["concept"]), description=f["description"],
                            include=bool(f.get("include", True)),
                            text_terms=list(f.get("text_terms") or []),
                            visual=bool(f.get("visual", True)),
                            fact_key=f.get("fact_key"),
                            fact_values=[v.lower() for v in f.get("fact_values") or []],
                            null_is_no=bool(f.get("null_is_no", False)))
                     for f in d.get("filters") or []],
            anchor_text_terms=list(d.get("anchor_text_terms") or []),
            before_s=float(d.get("before_s", 5.0)), after_s=float(d.get("after_s", 1.0)),
            limit=d.get("limit"), ambiguity=d.get("ambiguity", ""),
        )


def _merge_alternatives(filters: list[Filter]) -> list[Filter]:
    """Filters are ANDed. Two inclusion filters on the same fact (cause = fire,
    cause = lava) can never both hold for one moment, so they can only have
    meant OR - merge them into one filter that accepts either."""
    out: list[Filter] = []
    for f in filters:
        twin = next((g for g in out if f.include and g.include and f.fact_key
                     and g.fact_key == f.fact_key), None)
        if twin is None:
            out.append(f)
            continue
        twin.concept = f"{twin.concept}_or_{f.concept}"
        twin.description = f"{twin.description} OR {f.description}"
        twin.text_terms = list(dict.fromkeys(twin.text_terms + f.text_terms))
        twin.fact_values = list(dict.fromkeys(twin.fact_values + f.fact_values))
        twin.null_is_no = twin.null_is_no and f.null_is_no
    return out


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


def fact_verdict(facts: dict | None, f: Filter) -> tuple[str, str]:
    """Decide from prewarm's structured facts: exact, no vocabulary needed.

    The moment's recorded value for the filter's key (cause, enemy, ...):
    accepted -> yes; recorded but something else -> no; not recorded -> no
    if the plan said absence fails the filter, otherwise unknown.
    """
    if not facts or not f.fact_key:
        return "unknown", ""
    val = ((facts.get("facts") or {}).get(f.fact_key) or "").lower().strip()
    if not val:
        return ("no", f"no {f.fact_key} recorded") if f.null_is_no else ("unknown", "")
    if "<any>" in f.fact_values:
        return "yes", f"fact {f.fact_key}={val!r}"
    hit = any(v in val or val in v for v in f.fact_values if v)
    return ("yes" if hit else "no"), f"fact {f.fact_key}={val!r}"


def facts_catalog(index: dict, max_values: int = 25) -> str:
    """The fact values actually recorded in this recording, for the plan."""
    from collections import Counter

    counts: dict[str, Counter] = {}
    for o in index["occurrences"]:
        for k, v in ((o.get("facts") or {}).get("facts") or {}).items():
            v = (v or "<none>").lower()
            counts.setdefault(k, Counter())[v if len(v) <= 40 else "<free text>"] += 1
    lines = []
    for k, c in counts.items():
        vals = ", ".join(f"{v} x{n}" for v, n in c.most_common(max_values))
        lines.append(f"{k}: {vals}")
    return "\n".join(lines)


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
                                             "role": lab.get("role", "event"),
                                             "confidence": lab.get("confidence", 0),
                                             "why": lab.get("why", "")}
            source = "labelled this recording's groups (1 model call)"
        else:
            source = "group labels from memory"
        rank = {"event": 0, "feed": 1}
        cands = [(int(g), v) for g, v in labels.items() if v["concept"] == concept]
        if not cands:
            # Two independent calls can name one thing differently ("death" vs
            # "death_screen"): fall back to sharing a word.
            words = set(concept.split("_"))
            cands = [(int(g), v) for g, v in labels.items() if words & set(v["concept"].split("_"))]
        if not cands:
            return None, f"no discovered group is a '{concept}' ({source})"
        # Prefer an event over a feed of the same name, then the surer label.
        g, _ = min(cands, key=lambda c: (rank.get(c[1].get("role", "event"), 2), -c[1].get("confidence", 0)))
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
            return cached["answer"], f"vision (memory): {cached['why']}", cached.get("evidence_text")
        frames = self._frames(occ["t"])
        if not frames:
            return "unknown", "could not read frames", None
        q = (f"These are consecutive screenshots from {self.game} around one '{self.spec.anchor}' "
             f"moment. Read any on-screen text (messages, banners, kill feed) as well as the "
             f"picture. Question: {flt.description} Answer yes, no, or unsure.")
        self.calls += 1
        res = self.llm.check(frames, q)
        ev = res.get("evidence_text")
        self.mem.rec["checks"][key] = {"answer": res.get("answer", "unsure"), "why": res.get("why", ""),
                                       "evidence_text": ev}
        return res.get("answer", "unsure"), f"vision: {res.get('why', '')}", ev

    def fulfil(self, spec: RequestSpec) -> Fulfilment:
        self.spec = spec
        notes = [f"ambiguity: {spec.ambiguity}"] if spec.ambiguity else []
        gid, source = self.resolve_anchor(spec.anchor)
        if gid is None:
            self.mem.save()
            return Fulfilment(spec, None, source, [], self.calls, notes + [
                "next rung (not yet automatic): build a detector for the anchor from verified "
                "exemplars - see verify_images.py"])

        occs = _collapse([_with_facts(o) for o in self.index["occurrences"] if o["group"] == gid])
        # Keyed by the group's stable label, not the plan's wording for the
        # anchor, so every request about these moments shares what was learned.
        cases_key = f"{self.mem.rec['labels'].get(str(gid), {}).get('concept', spec.anchor)}__cases"
        # Inclusion filters first: they are the selective ones, and a moment
        # one of them rules out never needs the others.
        filters = sorted(spec.filters, key=lambda f: not f.include)
        for f in filters:
            f.text_terms = list(dict.fromkeys(f.text_terms + self.mem.terms(f.concept)))

        def vocab_for(f: Filter) -> list[str]:
            """Phrases that, if read, prove a case *other* than f's."""
            pool = spec.anchor_text_terms + self.mem.terms(cases_key)
            for g in filters:
                if g is not f:
                    pool += g.text_terms
            pool = [t for t in dict.fromkeys(pool) if t not in f.text_terms]
            # A phrase present at most of the anchor's moments describes the
            # anchor itself ("You died!"), not one case of it - measured, not assumed.
            return [t for t in pool
                    if sum(phrase_in(t, " | ".join(o["text"])) for o in occs) <= 0.5 * max(1, len(occs))]

        state = {o["id"]: Moment(o["id"], o["t"], "yes") for o in occs}
        open_: dict[str, list[Filter]] = {o["id"]: list(filters) for o in occs}
        by_id = {o["id"]: o for o in occs}

        def settle_by_text() -> None:
            others = {f.concept: vocab_for(f) for f in filters}
            for oid, todo in open_.items():
                m = state[oid]
                for f in list(todo):
                    if m.verdict == "no":
                        break
                    v, why = fact_verdict(by_id[oid].get("facts"), f)
                    if v == "unknown":
                        v, why = text_verdict(by_id[oid]["text"], f.text_terms, others[f.concept])
                    if v == "unknown":
                        continue
                    todo.remove(f)
                    m.evidence.append(f"{f.concept}: {why}")
                    if (v == "yes") != f.include:
                        m.verdict = "no"

        def apply(oid: str, f: Filter, ans: str, why: str, ev: str | None) -> None:
            m = state[oid]
            open_[oid].remove(f)
            m.evidence.append(f"{f.concept}: {why}")
            if ans == "unsure":
                m.verdict = "maybe" if m.verdict == "yes" else m.verdict
            elif (ans == "yes") != f.include:
                m.verdict = "no"
            # Learn the wording vision relied on, so text settles it next time:
            # a "yes" phrase belongs to this filter, a "no" phrase is another
            # case of the anchor. Only phrases OCR can actually see are kept.
            if ev and ans in ("yes", "no") and phrase_in(ev, " | ".join(by_id[oid]["text"]), 0.75):
                if ans == "yes":
                    self.mem.learn_terms(f.concept, [ev])
                    f.text_terms = list(dict.fromkeys(f.text_terms + [ev]))
                else:
                    self.mem.learn_terms(cases_key, [ev])

        settle_by_text()
        budget = self.max_checks
        # Vision in waves. After each wave, newly learned phrases are tried on
        # every moment still open, so one look at "was shot by arrow" settles
        # the other arrow deaths by text instead of by more vision calls.
        while budget > 0:
            wave = [(oid, todo[0]) for oid, todo in open_.items()
                    if todo and state[oid].verdict != "no"][:min(self.workers, budget)]
            if not wave:
                break
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                answers = list(pool.map(lambda w: self._vision(by_id[w[0]], w[1]), wave))
            budget -= len(wave)
            for (oid, f), (ans, why, ev) in zip(wave, answers):
                apply(oid, f, ans, why, ev)
            settle_by_text()

        undecided = [oid for oid, todo in open_.items() if todo and state[oid].verdict != "no"]
        for oid in undecided:
            state[oid].verdict = "maybe"
            state[oid].evidence.append("undecided: check budget used up")
        if undecided:
            notes.append(f"{len(undecided)} moment(s) left undecided (check budget)")

        moments = sorted(state.values(), key=lambda m: m.t)
        if spec.limit:
            keep = {m.id for m in [m for m in moments if m.verdict == "yes"][-spec.limit:]}
            moments = [m for m in moments if m.verdict != "yes" or m.id in keep]
        self.mem.save()
        return Fulfilment(spec, gid, source, moments, self.calls, notes)


def _with_facts(o: dict) -> dict:
    """Occurrence text plus whatever prewarm's vision pass read and noted."""
    f = o.get("facts")
    if not f:
        return o
    extra = list(f.get("on_screen_text") or [])
    if f.get("what_happened"):
        extra.append(f["what_happened"])
    return dict(o, text=list(dict.fromkeys(extra + o["text"])))


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
    f = Fulfiller(index, game, llm, mem)
    # Label first (once per recording), so the plan can pick its anchor from
    # the names this recording actually has.
    f.resolve_anchor("")
    catalog = sorted({f"{v['concept']} ({v.get('role', 'event')})" for v in mem.rec["labels"].values()
                      if v["concept"] != "unknown"})
    spec = RequestSpec.from_plan(llm.plan(query, game, catalog, facts_catalog(index)))
    f.calls += 1  # the plan
    result = f.fulfil(spec)
    if hasattr(llm, "live_calls"):
        result.model_calls = llm.live_calls  # cache hits cost nothing; count only real calls
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
