"""
pipeline.py -- THE JOIN. One verified path from transcript to a structured sheet.

WHY THIS FILE EXISTS
--------------------
Until now there were two half-systems:

  * `understudy_app.py` had a face (web UI, runtime switcher, pre-warm) but the OLD brain -- it
    asked the model to WRITE the clinical note, so nothing in the output had provenance.
  * `gate/` + `intake/` had the verified brain (arm E: the model returns IDs, code assembles) but
    no face, and its entry points were experiment scripts.

This module is the join. It exposes ONE function, `build_sheet`, that takes a transcript and a
chat function and returns STRUCTURED data -- not a formatted string -- so the UI can do the two
things a string cannot support: show each line's receipt, and let the patient edit or delete it.

THE GUARANTEE, RESTATED
-----------------------
The model never emits the patient's words. It receives numbered candidate pieces and replies with
`{"ids": [...]}`. The text and the character offsets both come from the SEGMENTATION of the
patient's own transcript, so a line that was not in the transcript cannot appear in the sheet.
Fabrication is structurally impossible rather than detected afterwards. Every line still runs
through `span_check.verify_span` -- redundant by construction, kept as defence in depth and because
its SpanResult carries the offsets the UI displays.

WHY THIS DOES NOT IMPORT THE EXPERIMENT SCRIPTS
-----------------------------------------------
`segment()` and the arm-E prompt originate in `intake/anvil/build_structure_arms.py` and
`build_structure_v2.py`. Those import medspaCy and a cloud client at module scope, so importing
them would make the product fail to start whenever an experiment dependency is missing, and would
couple shipping code to files whose whole purpose is to be rewritten. They are reproduced here
VERBATIM instead.

  !! If `segment()` or the arm-E prompt is changed in either place, the 180-cell gemma4:e2b
  !! validation (0 gate failures, 0.00 restatement, 55.9% coverage on real transcripts) no longer
  !! describes this code. Change both, or re-run the arms.

ONE SIMPLIFICATION OVER a3.py
-----------------------------
`a3.assemble` needed a second LLM call to tag tier and rank, because arm A returns free text with
no ordering guarantee. Arm E returns ids "most important first" -- the ordering IS the rank. So the
tagger call is dropped: one fewer model round-trip, one fewer failure mode, and materially less
wall-clock on a CPU laptop. Tier 3 (family history) is still decided by rule via medspaCy, and
still degrades to "no tier 3" when medspaCy is unavailable, because an undefined tier is worse
than no tier.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from span_check import verify_span          # noqa: E402
import safety_net                            # noqa: E402


def _quiet_medspacy() -> None:
    """medspaCy's sentencizer (PyRuSH) logs every token decision at DEBUG through loguru.

    On the first real run it emitted several screens of token/tag mappings before the output. Noise
    in a terminal; in the web app it would be written straight to the server's stdout during a live
    demo. Disabled by NAME rather than by calling logger.remove(), so this does not silently take
    out anyone else's logging.
    """
    try:
        from loguru import logger as _loguru
        for name in ("PyRuSH", "medspacy", "PyRuSH.PyRuSHSentencizer"):
            _loguru.disable(name)
    except Exception:
        pass


_quiet_medspacy()


# ---------------------------------------------------------------------------
# Reproduced VERBATIM from intake/anvil/build_structure_arms.py -- see header.
# ---------------------------------------------------------------------------
TASK = ("You are helping a patient prepare for a doctor visit. From what the patient said, choose "
        "the parts that should go in a short letter to their doctor. Choose what matters most "
        "first. Do not invent anything. Do not add clinical terms the patient did not use.")

# Reproduced VERBATIM from intake/anvil/build_structure_v2.py (arm E).
BUDGET = ("Include EVERYTHING that belongs in the letter — typically 8 to 15 pieces. "
          "Do not stop at three or four; a doctor would rather see the whole picture. "
          "Still put the most important first.")

# How many leading selections read as "what matters most". The model was told to order by
# importance; this is only where the visual break falls, and it is deliberately a display concern
# rather than a claim about clinical priority.
PRIMARY_CUTOFF = 4

CONTEXT_LABEL = "Background — family history"

# Caps on the accommodations block (see build_sheet/assemble_from_ids below). Conservative and
# arbitrary, matching the precedent set by /polish's own [:12] lines / [:600] chars limits --
# this is a request body, not a trusted internal value, and the block has no model in the loop to
# push back on abuse.
ACCOM_MAX_ITEMS = 20
ACCOM_MAX_CHARS = 300


# Words that make a clause DEPENDENT on the one before it. The segmenter splits on
# ", and|but|so|then|because", which is exactly the joint where a clause is attached to the thing
# it is about -- so a candidate can begin with one of these and carry no subject of its own.
# Measured 2026-07-31 over 851 sheet lines from 160 transcripts: 30.6% of lines began with one of
# these, and 18.1% of ALL lines were STRANDED -- their head clause was not on the sheet anywhere.
# On the page that reads as "and it's getting worse." / "but now it's almost black in the middle."
# Every one of those is perfectly verbatim and passes span verification: this is a failure of
# REFERENCE, not of provenance, which is why no existing check could see it.
DEPENDENT_STARTS = ("and", "but", "so", "then", "because", "which", "though", "although", "or")


def _self_contained(transcript: str, cands: list[dict], cid: int, selected: set) -> dict:
    """Give a selected candidate back its head clause when it would otherwise be stranded.

    If the candidate starts with a dependent word AND its immediately preceding candidate is
    contiguous in the transcript AND that head was not itself selected, extend this line's span
    LEFT to the head's start. The result is still a literal substring of the transcript (we only
    widen within text we already own), so the receipt, the offsets and verify_span all still hold
    -- the line simply carries the words that make it mean something.

    If the head WAS selected, nothing is done: both halves are already on the sheet.
    """
    c = cands[cid]
    first = c["text"].lstrip().split(" ", 1)[0].strip(",.;:!?").lower()
    if first not in DEPENDENT_STARTS or cid == 0 or (cid - 1) in selected:
        return c
    head = cands[cid - 1]
    gap = transcript[head["end"]:c["start"]]
    # contiguous means "same sentence, separated only by the joint the splitter cut at"
    if len(gap) > 3 or "\n" in gap or any(ch in gap for ch in ".!?"):
        return c
    start = head["start"]
    return {"text": transcript[start:c["end"]], "start": start, "end": c["end"],
            "rejoined": True}


def segment(text: str) -> list[dict]:
    """Clause-level candidates over the WHOLE transcript. Everything is a candidate.

    Reproduced verbatim from build_structure_arms.segment. Each candidate carries its character
    offsets, and those offsets are what later become the line's receipt.
    """
    cands, pos = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            pos += 1
            continue
        start = text.index(line, pos); pos = start + len(line)
        parts = [q for q in (x.strip() for x in re.split(
            r"(?<=[.!?])\s+|,\s+(?=(?:and|but|so|then|because)\b)|\s+-\s+", line)) if q]
        # The two-word floor exists to drop junk FRAGMENTS left by clause splitting. Applied to a
        # whole line it silently deletes real content: a terse bullet list ("Headaches / Dizzy /
        # Tired") produced ZERO candidates and an empty sheet with no error -- total, silent loss,
        # and most likely for exactly the terse / executive-function users this is built for.
        # So the floor now applies only to SPLIT fragments; a line that was never split is the
        # patient's whole utterance and is kept even at one word.
        whole_line = len(parts) == 1
        off = start
        for p in parts:
            s = text.index(p, off); off = s + len(p)
            if len(p.split()) >= 2 or whole_line:
                # sid = the span's start offset. Stable under APPEND-ONLY transcript
                # growth (earlier offsets never move), which positional indices are not:
                # a list index silently renumbers every time the patient says one more
                # thing. That renumbering is what blocks incremental rebuild, save/load,
                # and stable card identity in the three-pane plan -- all three at once.
                cands.append({"text": p, "start": s, "end": s + len(p), "sid": s})
    return cands


def prompt_E(cands: list[dict]) -> str:
    """Arm E: segmented candidates + ID reply + explicit budget. The validated arm."""
    body = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(cands))
    return (f"{TASK}\n\nThe patient's words are split into numbered pieces below. Choose the pieces "
            f"that belong in the letter, most important first. Do NOT rewrite them.\n{BUDGET}\n\n"
            f"Reply ONLY with JSON: {{\"ids\": [3, 7, 1]}} using the numbers.\n\nPIECES:\n{body}")


def _parse_ids(out: str, n_cands: int) -> list[int]:
    """Pull `{"ids": [...]}` out of a model reply, tolerantly, then drop anything out of range.

    Out-of-range ids are the one way this arm can still go wrong: a hallucinated index cannot
    fabricate TEXT (the text is ours), but it could silently select a piece the model never meant.
    Dropping them is safe; inventing a fallback would not be.
    """
    t = out.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        obj = json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j < 0:
            return []
        try:
            obj = json.loads(t[i:j + 1])
        except Exception:
            return []
    ids = obj.get("ids", obj if isinstance(obj, list) else [])
    seen, out_ids = set(), []
    for v in ids:
        if isinstance(v, bool) or not isinstance(v, int):
            continue
        if 0 <= v < n_cands and v not in seen:
            seen.add(v)
            out_ids.append(v)
    return out_ids


_NLP = None            # cached pipeline; loading medspaCy costs seconds and was being paid per sheet
_NLP_ERR = None        # why it is unavailable, if it is -- surfaced rather than swallowed


def _load_nlp():
    """Load medspaCy once and remember the outcome, including the failure.

    The previous version rebuilt the pipeline on EVERY call and caught every exception with a bare
    `return set()`. Both were wrong in the same direction: the reload made each sheet slower than it
    needed to be on the machine that can least afford it, and the silent catch meant a broken
    medspaCy looked exactly like a transcript with no family history. That is the B3 failure mode --
    a capability check that fails OPEN produces confident, plausible, wrong output. It cost a whole
    benchmark once; here it quietly emptied a tier, and the first browser run showed a patient's
    aunt's cancer filed under "what matters most" with nothing anywhere saying why.
    """
    global _NLP, _NLP_ERR
    if _NLP is not None or _NLP_ERR is not None:
        return _NLP
    try:
        import medspacy
        from medspacy.ner import TargetRule
        nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_target_matcher",
                                    "medspacy_context"])
        nlp.get_pipe("medspacy_target_matcher").add([
            TargetRule(t, "FINDING", pattern=[{"LOWER": t}]) for t in
            ("cancer", "diabetes", "heart", "stroke", "pain", "lump", "surgery", "attack",
             "problem", "condition", "died", "illness")])
        _NLP = nlp
    except Exception as e:
        _NLP_ERR = f"{type(e).__name__}: {e}"
        print(f"[understudy] family-history tagging OFF -- medspaCy unavailable ({_NLP_ERR}). "
              f"Sheets will have no 'Background' tier.", file=sys.stderr, flush=True)
    return _NLP


def context_status() -> str:
    """'on' or 'off: <reason>'. Goes into the sheet's stats so the absence of a tier is visible."""
    _load_nlp()
    return "on" if _NLP is not None else f"off: {_NLP_ERR}"


def _family_history_idx(texts: list[str]) -> set[int]:
    """Which selected lines are about someone else. Rule-based, via medspaCy's experiencer axis.

    Returns an empty set when medspaCy is unavailable -- no tier 3 at all rather than a tier whose
    meaning is undefined. The label promises family history ONLY: ConText's temporality axis does
    not fire on lay phrasing ("I had surgery on my knee years back" is not marked historical), so
    claiming past-episode detection would promise what the mechanism does not deliver.
    """
    nlp = _load_nlp()
    if nlp is None:
        return set()
    out = set()
    for i, s in enumerate(texts):
        try:
            ents = list(nlp(s).ents)
        except Exception:
            continue
        if ents and all(e._.is_family or e._.is_historical for e in ents):
            out.add(i)
    return out


def _norm(s: str) -> str:
    """Loose comparison key: lowercase, collapsed whitespace, no punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", " ".join(s.lower().split()))


def _cue_list(transcript: str, lines: list[dict], dropped_texts: list[str],
              keep_cues: list[int] | None) -> list[dict]:
    """The cue block, as {id, text}, after every filter the patient's choices imply.

    Cue ids index into safety_net.flagged_words(transcript), which is deterministic, so an id means
    the same sentence on every call and the client can delete one by id without sending text back.

    Three filters, in order:
      1. novelty   -- drop cues already represented among the KEPT lines. `safety_net.
                      flagged_words` returns every sentence that caused a safety topic to be
                      included, so that anything the patient later waved off still reaches the
                      doctor in their own words -- the right behaviour for the thing it was built
                      for. But on the first end-to-end run the transcript was about chest
                      symptoms, so nearly every sentence WAS a cue, and the sheet printed the
                      whole letter twice. The purpose is preserving what would otherwise be LOST,
                      not restating what is already there, so a cue is dropped once it is already
                      represented among the selected lines. The comparison just below (`k in x or
                      x in k`) is loose and two-directional because segmentation splits on clause
                      boundaries: a cue sentence often CONTAINS a selected clause rather than
                      equalling it. (This subsumed a standalone `_novel_cues` helper that did the
                      same check against the pre-deletion line set; removed once `_cue_list`
                      covered its case with the deletion/explicit filters below added on top --
                      kept as one function so there is one place this rationale has to stay true.)
      2. deletion  -- drop cues matching a line the patient REMOVED (see assemble_from_ids)
      3. explicit  -- if keep_cues is given, keep only those ids
    """
    all_cues = safety_net.flagged_words(transcript)
    kept_keys = [_norm(l["text"]) for l in lines]
    dropped_keys = [_norm(t) for t in dropped_texts]
    out = []
    for i, c in enumerate(all_cues):
        k = _norm(c)
        if not k:
            continue
        if any(k in x or x in k for x in kept_keys if x):        # already on the sheet
            continue
        if any(k in x or x in k for x in dropped_keys if x):     # patient removed it
            continue
        if keep_cues is not None and i not in keep_cues:         # patient removed the cue itself
            continue
        out.append({"id": i, "text": c})
    return out


def _sentence(span: str) -> str:
    """Capitalize the first letter, ensure end punctuation. These are the only two normalizations
    applied to a patient's words anywhere in the product, and both are whitelisted by the gate.

    Note what is deliberately NOT done: grammar is not smoothed. "is problem with leg" stays. For
    this population non-standard grammar is the VOICE, not a defect, and smoothing it is rewording.
    """
    s = span.strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".?!":
        s += "."
    return s


def build_sheet(transcript: str, chat_fn, model: str, *, max_tokens: int = 700,
                app_name: str = "My Say",
                chip_spans: list[list[int]] | None = None,
                accommodations: list[str] | None = None) -> dict:
    """Transcript -> structured sheet. The only entry point the UI should call.

    Returns:
      {
        "lines":   [ {id, text, display, tier, start, end, source_text, verified, chip_origin} ],
        "safety":  "<generic safety information, always present>",
        "cues":    ["the patient's own sentences that triggered a safety topic"],
        "accommodations": ["how I need this visit to go, in my own words"],
        "stats":   {n_candidates, n_selected, n_verified, n_dropped_out_of_range},
        "transcript": the raw string, so the UI can highlight source offsets
        "chip_spans": the ranges the caller passed in, echoed back so a later /export call (which
                       only has `transcript` + id lists, not the original chat messages) can pass
                       the SAME ranges again and keep the stamp -- see assemble_from_ids.
      }

    `tier` is one of "primary" | "secondary" | "context". `verified` is the gate's verdict; it
    should always be True here, and a False would mean segmentation and verification disagree,
    which is worth surfacing rather than hiding.

    `chip_spans` -- [[start,end], ...] char ranges into `transcript` that came from a starter-chip
    tap rather than something the patient typed (see the CHIP PROVENANCE note on assemble_from_ids
    for why this has to be computed by the CALLER, not here).

    `accommodations` -- see assemble_from_ids. Independent of the conversation and the model: it
    is never sent to chat_fn, never touched by the selection prompt, never rewritten. Threaded
    through here only so the sheet build_sheet returns is complete on its own, matching what
    /export re-derives.
    """
    cands = segment(transcript)
    if not cands:
        return {"lines": [], "safety": safety_net.build(transcript, app_name=app_name), "cues": [],
                "accommodations": _clean_accommodations(accommodations),
                "stats": {"n_candidates": 0, "n_selected": 0, "n_verified": 0,
                          "n_dropped_out_of_range": 0},
                "transcript": transcript, "app_name": app_name,
                "chip_spans": chip_spans or []}

    raw_out = chat_fn(model,
                      [{"role": "user", "content": prompt_E(cands)}],
                      temperature=0, max_tokens=max_tokens, force_json=True)
    pos_ids = _parse_ids(raw_out, len(cands))
    # the model speaks positions; everything past this line speaks stable ids
    ids = [cands[i]["sid"] for i in pos_ids]
    sheet = assemble_from_ids(transcript, ids, _cands=cands, app_name=app_name,
                              chip_spans=chip_spans, accommodations=accommodations)
    sheet["stats"]["n_dropped_out_of_range"] = max(
        0, len(_parse_ids(raw_out, 10 ** 9)) - len(ids))
    return sheet


def _clean_accommodations(accommodations: list[str] | None) -> list[str]:
    """Normalize the accommodations block the same way any other patient text is normalized.

    Two sources ever reach this list, both wired through understudy_app.py's /sheet and /export
    handlers: a FIXED curated statement the patient chose (see ACCOM_OPTIONS in the page's own
    JS -- choosing counts as authoring, same principle as `chosen_by: "patient"` elsewhere in this
    file) or the patient's own free typing. No model is in this path at all -- not a design
    omission, a requirement: this block is the patient's statement of how THEY need the visit to
    go, and it must never be paraphrased on their behalf.

    `_sentence()` (capitalize + end punctuation) is the same, and only, normalization already
    applied to every other patient-authored line -- extended here to a third input source rather
    than inventing a new one. Length/count caps mirror /polish's own request-body caps: this is
    attacker-reachable input with no model or gate downstream to catch pathological cases.
    """
    if not accommodations:
        return []
    out = []
    for a in accommodations[:ACCOM_MAX_ITEMS]:
        s = str(a or "").strip()[:ACCOM_MAX_CHARS]
        if s:
            out.append(_sentence(s))
    return out


def assemble_from_ids(transcript: str, ids: list[int], _cands=None,
                      dropped_ids: list[int] | None = None,
                      keep_cues: list[int] | None = None,
                      added_ids: list[int] | None = None,
                      app_name: str = "My Say",
                      chip_spans: list[list[int]] | None = None,
                      accommodations: list[str] | None = None,
                      diary: list[str] | None = None) -> dict:
    """The deterministic half of build_sheet, callable on its own. NO MODEL IS INVOLVED.

    CHIP PROVENANCE (design review, 2026-08-01). `chosen_by` above answers "who put this line on
    the sheet" (model selection vs. the patient adding it back). It does NOT answer a different
    question: "did the WORDS in this line come from the patient composing them, or from tapping an
    app-authored starter chip?" Before this, they could not be told apart anywhere past the click --
    startWith() set no marker, so a chip's sentence sat in msgs[], the transcript, and the receipt
    looking exactly like typed text, meaning app-suggested wording could reach the reviewed sheet
    wearing the same "these are your own words" receipt as anything the patient actually said.
    `chip_spans` closes that gap: it is a list of [start,end) char ranges into THIS transcript that
    the caller (understudy_app.py's /sheet and /export handlers) marks as chip-authored, computed
    from msgs[].origin==='chip' at the one point that still has that information -- pipeline.py
    itself never sees msgs[], only the flattened transcript string, so it cannot discover this on
    its own. Any assembled line whose verified span overlaps a chip range is stamped
    `"chip_origin": True`. This is deliberately a SEPARATE field from `chosen_by`, not a third value
    of it: the two questions are independent (a chip-originated line can still be model-selected OR
    patient-re-added), and overloading one field would silently lose one of the two facts.
    Round trip: build_sheet() echoes chip_spans back on the returned dict; the client stores it on
    SHEET.chip_spans and resends it verbatim to /export, which passes it straight through here again
    -- so the stamp survives the same re-derive-from-transcript path /export already uses for
    everything else, no client-trusted rendering involved.

    Exists so that EXPORT can re-derive the sheet from the transcript plus the ids the patient kept,
    rather than trusting a rendering the client is holding. That matters: the sheet the patient
    approves is the one after their deletions, and a stale server-side render would email the
    doctor something they did not approve. Re-assembling means the guarantee -- every line traceable
    to the transcript -- holds for the exported artifact too, not just the on-screen one.

    THE PATIENT IS GROUND TRUTH (Alex's ruling, 2026-07-27, recorded because it is a principle and
    not a preference): "The patient is the one with agency. The patient is the one making the
    decisions. The patient has full control." There is no line the product keeps against their
    wishes because it judges the line important.

    That ruling fixed a real defect. Deleting a line USED to move it rather than remove it: the
    safety-cue block suppresses a cue when it is already represented among the selected lines, so
    deleting the line made its cue "novel" again and the sentence reappeared under a different
    heading -- and the export got LONGER after a deletion. Silent resurrection is the worst
    available behaviour, because it neither honours the deletion nor admits that it didn't.

      * dropped_ids -- lines the patient removed. Their text is suppressed from the cue block too,
        because it is the same sentence wearing a different hat.
      * keep_cues   -- cue indices to retain. Cues are the patient's words as well, so they are
        deletable on the same terms; passing None keeps them all.
      * app_name    -- threaded through to safety_net.build and carried on the returned sheet
        (as "app_name") so render_text can label the added-content notice without this module
        hardcoding a product name it does not own. The caller (understudy_app.py) is the one
        place APP_NAME is actually defined; a rename there is still a one-line edit as long as
        every caller passes it through instead of a module here re-hardcoding the old name.
      * accommodations -- the "how I need this visit to go" block (see _clean_accommodations).
        Not derived from `transcript` or `ids` at all -- it has no candidate, no span, no receipt,
        because it is not extracted from anything: it is either the patient's own typing or a
        fixed statement they chose (chosen_by: patient by construction, same as any other patient-
        added line). Carried through export so the re-derived sheet the patient actually approves
        still has it, exactly like every other patient control on this sheet.
    """
    def _chip_hit(s, e):
        if not chip_spans:
            return False
        return any(s < ce and e > cs for cs, ce in chip_spans)

    cands = _cands if _cands is not None else segment(transcript)
    # ids arriving here are STABLE ids (span starts). Legacy callers may still pass list
    # positions; a position is only accepted when it is not also a valid stable id, so the
    # unambiguous case never changes meaning.
    by_sid = {c["sid"]: k for k, c in enumerate(cands)}

    def _pos(i):
        if i in by_sid:
            return by_sid[i]
        return i if 0 <= i < len(cands) else None

    positions = [p for p in (_pos(i) for i in ids) if p is not None]
    ids = [cands[p]["sid"] for p in positions]
    dropped_texts = [cands[p]["text"] for p in
                     (_pos(i) for i in (dropped_ids or [])) if p is not None]
    sel_positions = set(positions)
    chosen = [_self_contained(transcript, cands, p, sel_positions) for p in positions]
    fam = _family_history_idx([c["text"] for c in chosen])

    lines, n_verified = [], 0
    for pos, (cid, c) in enumerate(zip(ids, chosen)):
        res = verify_span(transcript, c["text"])
        if res.ok:
            n_verified += 1
        tier = ("context" if pos in fam
                else "primary" if pos < PRIMARY_CUTOFF
                else "secondary")
        line_start = res.source_start if res.ok else c["start"]
        line_end = res.source_end if res.ok else c["end"]
        lines.append({
            "id": cid,
            "text": c["text"],
            "display": _sentence(c["text"]),
            "tier": tier,
            "start": line_start,
            "end": line_end,
            "source_text": res.matched_source_text if res.ok else c["text"],
            "verified": bool(res.ok),
            # Who put this line on the sheet. A patient-added line has STRONGER provenance than a
            # model-selected one -- same verified words, but chosen by the person they belong to.
            "chosen_by": "patient" if (added_ids and cid in added_ids) else "model",
            # Where the WORDS came from -- see the CHIP PROVENANCE docstring note above. Independent
            # of chosen_by: a chip-origin line can still be model-selected or patient-re-added.
            "chip_origin": _chip_hit(line_start, line_end),
        })

    # family-history lines sink to the bottom regardless of the model's ordering: they are
    # background, and letting them sit above a current symptom misrepresents the patient.
    lines.sort(key=lambda l: (l["tier"] == "context", ))

    return {
        "lines": lines,
        "safety": safety_net.build(transcript, app_name=app_name),
        "cues": _cue_list(transcript, lines, dropped_texts, keep_cues),
        "app_name": app_name,
        "accommodations": _clean_accommodations(accommodations),
        # MY WEEK -- the symptom diary (2026-08-01). Same authorship rule and the same
        # normalization as accommodations: every entry is a number the patient chose plus words
        # they typed -- patient-authored by construction, no candidate, no receipt.
        "diary": _clean_accommodations(diary),
        "stats": {
            "n_candidates": len(cands),
            "n_selected": len(ids),
            "n_verified": n_verified,
            "n_dropped_out_of_range": 0,
            "context_tagging": context_status(),
        },
        "transcript": transcript,
        "ids": ids,
        # Echoed back verbatim so the client can resend it on /export -- see the CHIP PROVENANCE
        # note above for why that round trip is what makes the stamp survive re-derivation.
        "chip_spans": chip_spans or [],
        # EVERYTHING THE PATIENT SAID THAT IS NOT ON THE SHEET.
        #
        # Without this the patient has veto power, not control: the model picked 3 of 6 pieces, and
        # they could delete those 3 but never learned the other 3 existed. Subtracting from someone
        # else's selection is a much thinner agency than choosing your own -- and thin agency is
        # exactly what this product exists to correct, for people whose experience is being spoken
        # over. Adding one back is putting an id in a list; the words were already segmented and
        # numbered before the model ever saw them.
        #
        # A line the patient DELETED reappears here, which is correct: it is still something they
        # said, and it is still theirs to put back.
        #
        # Each entry gets the same _self_contained rejoin as a chosen line: a dependent fragment
        # ("and it's getting worse") added back by the patient must carry its head clause, or the
        # add-back path re-creates the exact stranded lines the rejoin pass exists to prevent.
        "unselected": _unselected_list(transcript, cands, sel_positions),
    }


def _unselected_list(transcript: str, cands: list[dict], sel_positions: set) -> list[dict]:
    out = []
    for k, c in enumerate(cands):
        if k in sel_positions:
            continue
        rc = _self_contained(transcript, cands, k, sel_positions)
        out.append({"id": c["sid"], "text": rc["text"], "display": _sentence(rc["text"]),
                    "start": rc["start"], "end": rc["end"]})
    return out


def append_answer(old_transcript: str, answer: str, *, parent_id: int | None = None,
                  probe_element: str | None = None) -> dict:
    """"Ask me more about this" lands here. Appends a patient's answer to a probe question at the
    END of the transcript and returns the new candidate line(s) it produces -- WITHOUT touching
    any existing id, WITHOUT a model call, and without re-running selection.

    This is the mechanism the whole feature depends on. Every line is already keyed by a STABLE
    id -- the candidate's span START OFFSET (see segment(), "sid = the span's start offset...
    stable under APPEND-ONLY transcript growth"). Appending strictly at the end means every
    character before the appended text is byte-identical to before, so segment() re-run over the
    new transcript reproduces the exact same start offsets, and therefore the exact same ids, for
    every candidate that existed already. Nothing the patient dropped, pinned, added, or had
    polished needs to be replayed or reconciled against a re-run -- it was never invalidated.
    (Verified empirically, not just argued: see BUILD_REPORT.md for the before/after id diff.)

    This is why the answer does NOT go through the chat/rebuild path (which would call the model
    again, re-select from scratch, and reset every client-side edit) and does not need to overwrite
    the transcript's middle (which is the scenario that WOULD break existing ids/offsets).

    Returns {"transcript": <new full transcript>, "lines": [ {...} ]} where each line dict is in
    the exact shape assemble_from_ids's own per-line dicts use, so the client can push it onto
    SHEET.lines with the same code path addLine() already uses for a patient-restored line.
    """
    answer = (answer or "").strip()
    if not answer:
        return {"transcript": old_transcript, "lines": []}
    new_transcript = (old_transcript + "\n" + answer) if old_transcript else answer
    cands = segment(new_transcript)
    cutoff = len(old_transcript)
    new_cands = [c for c in cands if c["start"] >= cutoff]
    lines = []
    for c in new_cands:
        res = verify_span(new_transcript, c["text"])
        lines.append({
            "id": c["sid"],
            "text": c["text"],
            "display": _sentence(c["text"]),
            "tier": "secondary",
            "start": res.source_start if res.ok else c["start"],
            "end": res.source_end if res.ok else c["end"],
            "source_text": res.matched_source_text if res.ok else c["text"],
            "verified": bool(res.ok),
            "chosen_by": "patient",
            "probe_of": parent_id,
            "probe_element": probe_element,
        })
    return {"transcript": new_transcript, "lines": lines}


TIER_HEADINGS = {
    "primary": "What matters most",
    "secondary": "Also on my mind",
    "context": CONTEXT_LABEL,
}


def render_text(sheet: dict) -> str:
    """Plain-text rendering, for printing, email, and the terminal.

    Kept OUT of build_sheet on purpose: the sheet is data, and rendering is a separate concern that
    will have several skins. Every word the product adds is [bracketed]; everything unbracketed is
    the patient's, which makes the guarantee checkable by eye and by `a3.bracket_discipline`.

    The product name in the closing notice comes from `sheet["app_name"]`, set by build_sheet /
    assemble_from_ids from whatever the caller passed in -- not hardcoded here. This module has no
    business owning the app's name; understudy_app.APP_NAME does, and a rename there should not
    require finding every other file that also happened to spell the name out.
    """
    app_name = sheet.get("app_name", "My Say")
    out = []
    # ACCOMMODATIONS FIRST -- above "What matters most" and everything else, matching the sheet's
    # own on-screen order (see PAGE's #accom, placed before #body in understudy_app.py). This is
    # the AASPIRE-style accommodations block (Nicolaidis et al. 2016: a similar report in the
    # AASPIRE Healthcare Toolkit cut clinician-reported communication barriers 4.07->2.82,
    # p<0.0001, n=259 patients + 51 PCPs, and was read by ~97% of clinicians) -- the thing a
    # clinician reads first is put first. The label is bracketed like every other section heading
    # here; the statements themselves are not, because they are the patient's own words (typed, or
    # chosen from a fixed list -- choosing counts as authoring, same as chosen_by: "patient"
    # elsewhere in this file) and never a model's.
    if sheet.get("accommodations"):
        out.append("[How I need this visit to go, in my own words:] "
                   + " ".join(sheet["accommodations"]))
    # The symptom diary follows the same authorship rule: the entries are unbracketed because
    # every one is a number the patient chose and words they typed; only the heading is the
    # product's.
    if sheet.get("diary"):
        out.append("[My week, day by day — numbers I chose myself, 0–10:] "
                   + " ".join(sheet["diary"]))
    for tier in ("primary", "secondary", "context"):
        rows = [l for l in sheet["lines"] if l["tier"] == tier]
        if rows:
            out.append(f"[{TIER_HEADINGS[tier]}:] " + " ".join(r["display"] for r in rows))
    if sheet.get("cues"):
        out.append("[Things I mentioned, in my own words:] "
                   + " ".join(_sentence(c["text"]) for c in sheet["cues"]))
    if sheet.get("safety"):
        out.append(sheet["safety"])
    # The claim is SCOPED. It used to read "the unbracketed text is verbatim", which was false as
    # soon as safety information was added: the criteria bullets are unbracketed and are the
    # source's words, not the patient's. gate.a3.bracket_discipline flagged 63 of them. The
    # guarantee itself was never broken -- only the sentence asserting it, which had quietly grown
    # to cover text it was never about. An over-broad honesty claim is still a false claim.
    out.append("[My own words above are exactly as I said them. Anything in square brackets was "
               f"added by {app_name}. The safety information is general, is quoted from the sources "
               "named beside it, and was not chosen by looking at my case.]")
    return "\n\n".join(out)


def speech_segments(sheet: dict) -> list[tuple[str, str]]:
    """Another skin on the same data as render_text (see the note above it) -- for "read my sheet
    aloud" instead of print/email. render_text marks "this is My Say's, not yours" with
    [brackets]; read aloud there is no typography, so the same distinction has to survive as an
    audible marker instead (see gate/speech.py, which turns "app" into one spoken voice and
    "patient" into another). The rule for which is which is IDENTICAL to render_text's bracket
    rule -- including the fact that the safety block is "app" throughout even where its own text
    is unbracketed, because (per the note on render_text) that text is the SOURCE's words, quoted,
    never the patient's.

    Returns an ordered list of (speaker, text) where speaker is "app" or "patient".
    """
    segs: list[tuple[str, str]] = [("app", "Notes for my appointment.")]
    for tier in ("primary", "secondary", "context"):
        rows = [l for l in sheet["lines"] if l["tier"] == tier]
        if not rows:
            continue
        segs.append(("app", TIER_HEADINGS[tier] + "."))
        for r in rows:
            segs.append(("patient", r["display"]))
    if sheet.get("cues"):
        segs.append(("app", "Things I mentioned, in my own words."))
        for c in sheet["cues"]:
            segs.append(("patient", _sentence(c["text"])))
    if sheet.get("safety"):
        segs.append(("app", "Safety information."))
        for line in sheet["safety"].splitlines():
            # The print rendering's "[" / "]" / "-" are typographic conventions (bracket = ours,
            # dash = a bullet), not words -- reading them aloud as literal punctuation would be
            # noise, so they're stripped here. Nothing else about the text is touched.
            t = line.strip().strip("[]").lstrip("-").strip()
            if t:
                segs.append(("app", t))
    return segs


def _local_ollama_chat(model: str, messages: list, temperature: float = 0,
                       max_tokens: int = 700, force_json: bool = True) -> str:
    """Minimal, self-contained local-Ollama chat call, used only by `_demo()` below.

    This used to be `from intake.ollama_adapter import chat`. That adapter is a real, fuller
    client (retries, a keep_alive tune, a workaround for the empty-content-on-cold-decode trap)
    but it lives in the experiment tree (`intake/`), which this repo does not ship -- exactly the
    coupling the module docstring above says pipeline.py avoids for `segment()`/`prompt_E`. The
    demo entry point quietly broke that rule: anyone who cloned the public repo and ran
    `python gate/pipeline.py` out of curiosity got `ModuleNotFoundError: intake` before seeing
    anything. This is the stripped-down replacement -- no retries, no keep_alive tuning, just
    enough stdlib-only urllib to prove the pipeline end-to-end against a local Ollama.
    """
    import urllib.request
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": max(max_tokens, 512)}}
    if force_json:
        body["format"] = "json"
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d.get("message", {}).get("content") or ""


def _demo():
    """Runs the whole path on one built-in transcript against local Ollama (requires Ollama
    running locally with the model named below or passed as argv[1] -- no other setup)."""
    raw = ("ok so um the main thing is like my chest has been doing this weird thing, "
           "not like a heart attack or anything, more like a flutter? when I stand up too fast "
           "mostly. and honestly I've been so tired, like bone tired, for maybe three weeks, "
           "could be the new job I don't know. my aunt had breast cancer so I worry. "
           "oh and I keep meaning to ask about the thing on my shoulder but that's probably "
           "nothing. I just don't want to waste his time.")

    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
    sheet = build_sheet(raw, _local_ollama_chat, model)
    s = sheet["stats"]
    print(f"model={model}  candidates={s['n_candidates']}  selected={s['n_selected']}  "
          f"gate-verified={s['n_verified']}/{s['n_selected']}\n")
    for l in sheet["lines"]:
        mark = "ok " if l["verified"] else "FLAG"
        print(f"  [{mark}] {l['tier']:9s} [{l['start']:>4}:{l['end']:<4}] {l['display']}")
    print("\n" + "-" * 76 + "\n")
    print(render_text(sheet))


if __name__ == "__main__":
    _demo()
