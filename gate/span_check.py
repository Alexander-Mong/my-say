"""
span_check.py -- deterministic span-verification gate for My Say.

THE GUARANTEE
-------------
A candidate span tagged "extracted / verbatim" is PROVEN to be the patient's own
words, drawn IN-ORDER from the raw transcript, with only WHITELISTED drops
(fillers, disfluency repeats, capture artifacts, and whitespace/punctuation/case
normalization). Any added, changed, or reordered token -- OR any non-whitelisted
deletion *inside* the span (e.g. dropping "not") -- is a FLAG.

This converts "these are their words" from TRUST into a machine-proven fact.
There is NO LLM anywhere in this file (deterministic == reliable). A PASS returns
char-offset provenance into the raw source, so this file IS the provenance chain
(PIPELINE_ARCHITECTURE.md commitments 3 + 8).

THE TUNING SURFACE (deliberately left easy to iterate)
------------------------------------------------------
`FILLERS` and the tokenizer are the "needs sitting with" surface
(Alex, 2026-07-25). They are plain, editable data / a swappable function on
purpose. The test suite (test_span_check.py) documents every boundary case so
tuning has concrete examples to react to, not abstractions.

Raw source = the saved transcript from whatever STT software we choose
(Alex, 2026-07-25). The gate treats it as one plain string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# TUNING SURFACE 1 -- the drop-whitelist.
# Only these may be dropped *inside* a claimed-verbatim span. Everything else
# dropped inside a span is a meaningful deletion => FLAG. Start conservative;
# loosen only against real false-flags (Alex: strict for now).
# --------------------------------------------------------------------------
FILLERS: set[str] = {
    "um", "umm", "uh", "uhh", "er", "erm", "ah", "eh", "hmm", "mm",
    "like", "y'know",  # note: "like"/"so"/"well" can be meaningful -- the sitting-with case
}

# Multi-word discourse fillers, matched as token sequences (kept small on purpose).
FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
    ("you", "know", "what", "i", "mean"),
)

# --------------------------------------------------------------------------
# TUNING SURFACE 2 -- the tokenizer.
# One regex, three token kinds: a bracketed capture artifact ([inaudible],
# [00:12], speaker labels get bracketed by most STT), a bare timestamp, or a
# word (apostrophes kept so contractions stay one token). Swap freely later.
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\[[^\]]*\]|\d{1,2}:\d{2}(?::\d{2})?|[A-Za-z0-9']+")
_ARTIFACT_RE = re.compile(r"\[[^\]]*\]|\d{1,2}:\d{2}(?::\d{2})?")


@dataclass
class Token:
    text: str    # normalized form used for matching (lowercased)
    raw: str     # original surface form
    start: int   # char offset into the source string
    end: int


@dataclass
class SpanResult:
    ok: bool
    candidate: str
    source_start: int | None = None      # char-offset provenance into raw
    source_end: int | None = None
    matched_source_text: str | None = None
    dropped: list[str] = field(default_factory=list)   # whitelisted drops, for transparency
    flags: list[str] = field(default_factory=list)     # human-readable reasons on FAIL

    def __bool__(self) -> bool:  # so `if verify_span(...)` reads naturally
        return self.ok


def tokenize(s: str) -> list[Token]:
    """Split a string into offset-tracked tokens. Normalization = lowercase only;
    the regex already discards surrounding whitespace/punctuation."""
    out: list[Token] = []
    for m in _TOKEN_RE.finditer(s):
        surface = m.group(0)
        out.append(Token(text=surface.lower(), raw=surface, start=m.start(), end=m.end()))
    return out


def _is_artifact(tok: Token) -> bool:
    return bool(_ARTIFACT_RE.fullmatch(tok.raw))


def _phrase_drop_len(raw_toks: list[Token], i: int) -> int:
    """If a whitelisted filler PHRASE starts at raw index i, return its length
    (so the aligner can skip the whole phrase); else 0."""
    for phrase in FILLER_PHRASES:
        n = len(phrase)
        if i + n <= len(raw_toks) and all(raw_toks[i + k].text == phrase[k] for k in range(n)):
            return n
    return 0


def _droppable(raw_toks: list[Token], i: int) -> int:
    """Return how many raw tokens starting at i may be dropped as whitelisted
    (0 if the token at i is NOT droppable). Covers: single-word fillers,
    capture artifacts, immediate-repeat disfluencies ('I was-- I was'), and
    multi-word filler phrases."""
    tok = raw_toks[i]
    if tok.text in FILLERS:
        return 1
    if _is_artifact(tok):
        return 1
    # immediate repetition (stutter / false start): drop this copy
    if i + 1 < len(raw_toks) and raw_toks[i].text == raw_toks[i + 1].text:
        return 1
    n = _phrase_drop_len(raw_toks, i)
    return n  # 0 if no phrase


def _align(raw_toks: list[Token], cand_toks: list[Token], start: int):
    """Try to match every candidate token, in order, against raw beginning at
    raw index `start`. Between matches, only whitelisted raw tokens may be
    skipped. Returns (matched_end_index, dropped_surfaces) or None."""
    ri = start
    dropped: list[str] = []
    for ct in cand_toks:
        while ri < len(raw_toks) and raw_toks[ri].text != ct.text:
            step = _droppable(raw_toks, ri)
            if step == 0:
                return None  # non-whitelisted token in the way => meaningful deletion / mismatch
            for k in range(step):
                dropped.append(raw_toks[ri + k].raw)
            ri += step
        if ri >= len(raw_toks):
            return None  # ran out of source => candidate has an added/changed token
        matched_end = ri
        ri += 1
    return matched_end, dropped


def verify_span(raw: str, candidate: str) -> SpanResult:
    """Verify that `candidate` is a whitelisted-only extraction of some in-order
    region of `raw`. PASS => char-offset provenance; FAIL => human-readable flag(s)."""
    raw_toks = tokenize(raw)
    cand_toks = tokenize(candidate)

    if not cand_toks:
        return SpanResult(ok=False, candidate=candidate, flags=["empty candidate span"])

    raw_text_set = {t.text for t in raw_toks}

    # Try every anchor where the candidate's first token appears in raw.
    first = cand_toks[0].text
    anchors = [i for i, t in enumerate(raw_toks) if t.text == first]
    for s in anchors:
        res = _align(raw_toks, cand_toks, s)
        if res is not None:
            matched_end, dropped = res
            src_start = raw_toks[s].start
            src_end = raw_toks[matched_end].end
            return SpanResult(
                ok=True,
                candidate=candidate,
                source_start=src_start,
                source_end=src_end,
                matched_source_text=raw[src_start:src_end],
                dropped=dropped,
            )

    # No alignment -> diagnose the most useful reason.
    missing = [t.raw for t in cand_toks if t.text not in raw_text_set]
    if missing:
        reason = (f"token(s) {missing} not in source -- ADDED or CHANGED "
                  f"(synonym / tense / number)")
    elif not anchors:
        reason = f"first token '{cand_toks[0].raw}' never appears in source"
    else:
        reason = ("could not align in order -- REORDERED, or a meaningful word "
                  "was dropped inside the span (e.g. 'not')")
    return SpanResult(ok=False, candidate=candidate, flags=[reason])


if __name__ == "__main__":  # tiny smoke demo
    raw = "so um I have this like chest pain that isn't going away, you know?"
    for cand in ["I have this chest pain", "I have chest tightness", "chest pain going away"]:
        r = verify_span(raw, cand)
        mark = "PASS" if r.ok else "FLAG"
        detail = f"[{r.source_start}:{r.source_end}] '{r.matched_source_text}'" if r.ok else r.flags[0]
        print(f"{mark}  {cand!r}\n      {detail}\n")
