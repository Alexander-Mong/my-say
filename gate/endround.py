"""
endround.py -- the end-of-conversation HIGHER-REASONING round ("look back over everything").

WHAT THIS IS
------------
Alex's own idea, captured in three_pane/BRAINSTORM_ALEX_2026-07-31.md (section 2) and deferred
by the capturing agent, not by him: after the sheet exists, one deliberate pass over the WHOLE
conversation with deep reasoning ON (think=true) that produces:

  (a) QUESTIONS THE PATIENT MIGHT WANT TO ASK THEIR DOCTOR, and
  (b) THREADS that came up and were dropped without being followed.

THE INVARIANT THIS FILE MUST NOT BREAK
---------------------------------------
The doctor-facing sheet (pipeline.py / build_sheet / assemble_from_ids) contains ONLY the
patient's own words -- the model selects ids, code assembles, and /sheet builds its transcript
from role=="user" messages only, because anything the assistant said would verify as the
patient's own words and could land on the sheet wearing a valid receipt.

Questions are the ONE safe exception, and only if handled honestly. This file picks option (i)
from the brief: suggested questions and dropped-thread prompts are rendered in a clearly
separate, clearly labelled, APP-AUTHORED block -- never woven into the patient's-own-words
tiers, never given a "source" receipt chip that implies transcript provenance. That is the same
convention safety_net.build() already uses for the safety section: a whole block that is
honestly not the patient's words, bracketed as such in the exported text, kept structurally
distinct on the sheet (its own heading, no click-to-open receipt on the suggestion text itself).

WHY (i) OVER (ii) -- reasoned in the module docstring so the choice travels with the code:
Option (ii) (treat a suggested question as a prompt the patient answers, with the ANSWER
reaching the sheet) is architecturally cleaner -- it would route through the existing verified
ids-only gate with zero new surface. But it does not fit what a "question for your doctor"
actually IS: it is not a fact about the patient that needs to be voiced and then extracted: it
is literally the sentence the patient will say OUT LOUD in the exam room. Forcing it through a
round-trip ("here's a question -- please answer it in chat first") would turn a one-shot,
deliberate finale ("take a breath while it reads everything back over") into a follow-up
interview, undermining the exact framing Alex wrote down. So: (i), with the same honesty
discipline the safety block already established.

ONE MODEL CALL, NOT TWO
------------------------
Ollama serializes across agents in this dev environment and gemma4 thinking is slow -- the
budget note in the build task says "keep model calls few". Both lists come out of a single
think=true call.

GROUNDING, NOT JUST LABELLING
------------------------------
A suggested "question" carries no claim to be the patient's words, so it needs no verification --
only honest labelling. A "dropped thread" is different: its whole point is to quote something
the patient actually said ("you mentioned the tablets"), so a hallucinated quote there would be
exactly the B3 failure mode this codebase keeps calling out (confident, plausible, wrong). Every
dropped-thread quote is run through span_check.verify_span against the PATIENT-ONLY transcript
(the same one /sheet builds) before it is ever shown to the patient; a quote that does not
verify is dropped from the candidate list rather than shown with a caveat.

A light vocabulary gate (reusing polish.CLINICAL -- the same list, not a new one) also drops any
candidate that introduces a clinical term the patient never used, for the same reason polish.py
keeps that gate hard: the sheet's voice is the patient's own vocabulary, and a suggestion that
puts a clinical word in the patient's mouth would misrepresent them the moment it's approved.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from span_check import verify_span   # noqa: E402
import polish                         # noqa: E402  -- reusing polish.CLINICAL, not duplicating it

MAX_QUESTIONS = 5
MAX_DROPPED = 5

PROMPT = """You have been shown the FULL conversation between a patient and their private health \
confidant, from just before they take notes to their own doctor. Read it back over once, \
carefully, and produce two short lists.

1. QUESTIONS: up to {max_q} clear, specific questions this patient might want to ask their OWN \
doctor, grounded only in what they actually said. Write each one the way the patient would say it \
out loud in the room -- first person, one question each ("Could my ...", "Should I ...", "Is it \
possible that ..."). Every item MUST end in a question mark. Do NOT diagnose, do NOT name a \
condition the patient did not name, and do NOT state or imply what is wrong with them -- draft \
only the QUESTION, never an answer or an opinion about their health.

2. OPEN THREADS: up to {max_d} things the patient mentioned once but the conversation never came \
back to -- a detail, a worry, or a plan that was raised and then dropped. For EACH one, quote the \
patient's OWN sentence that raised it EXACTLY as they said it, character for character, copied \
from the PATIENT lines below (never the confidant's lines, never a paraphrase) -- then write one \
short, warm follow-up question the patient could answer for themselves or bring to the doctor.

Only use what is actually in the conversation below. If there is nothing real to add to a list, \
return an empty list for it -- never invent filler to fill a quota.

Reply ONLY with JSON of exactly this shape:
{{"questions": ["...", "..."], "dropped": [{{"quote": "the patient's exact sentence", "prompt": "the follow-up question"}}]}}

CONVERSATION (Patient / Confidant, in order):
{conversation}"""


def format_conversation(messages: list[dict]) -> str:
    """Both roles, labelled, in order -- the model needs the FLOW to see what got dropped.

    This is a separate, throwaway string built only for THIS prompt's context. It is never the
    transcript the sheet gate verifies against (that stays patient-only, see patient_transcript()
    below) -- mixing the two would let an assistant line's phrasing quietly influence what counts
    as "the patient's own sentence."
    """
    out = []
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        label = "Patient" if role == "user" else "Confidant" if role == "assistant" else None
        if label:
            out.append(f"{label}: {text}")
    return "\n".join(out)


def patient_transcript(messages: list[dict]) -> str:
    """Identical join to understudy_app.py's /sheet handler -- PATIENT TURNS ONLY -- so a quote
    verified here verifies against the exact same source the rest of the app treats as ground
    truth, and its offsets are safe to reuse for a receipt."""
    return "\n".join(m["content"].strip() for m in messages
                      if m.get("role") == "user" and (m.get("content") or "").strip())


def _parse(raw: str) -> dict:
    """Tolerant JSON extraction, same shape as pipeline._parse_ids / polish.parse_rewrite."""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        obj = json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j < 0:
            return {}
        try:
            obj = json.loads(t[i:j + 1])
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}


def _introduces_clinical_vocab(text: str, source: str) -> str | None:
    """Same doctrine as polish.run_gates's hard vocabulary gate, reused rather than reinvented:
    a clinical term that does not appear anywhere in the patient's own transcript is grounds to
    drop the candidate outright. Returns the offending word, or None if clean."""
    low = text.lower()
    src_low = source.lower()
    for w in polish.CLINICAL:
        if w in low and w not in src_low:
            return w
    return None


def build_review(messages: list[dict], chat_fn, model: str, *, max_tokens: int = 900) -> dict:
    """The one entry point. `chat_fn(model, messages, temperature, max_tokens, force_json)` mirrors
    pipeline.build_sheet's adapter signature so the same call_model wrapper in understudy_app.py
    covers both. Caller is responsible for passing think=True through chat_fn -- this module has
    no opinion on transport, only on the prompt and the gates.

    Returns:
      {"questions": [{"id","text"}],
       "dropped":   [{"id","quote","prompt","start","end"}],
       "stats": {"n_questions_raw","n_questions_kept","n_dropped_raw","n_dropped_kept"}}
    """
    convo = format_conversation(messages)
    patient_only = patient_transcript(messages)
    if not convo or not patient_only:
        return {"questions": [], "dropped": [],
                "stats": {"n_questions_raw": 0, "n_questions_kept": 0,
                          "n_dropped_raw": 0, "n_dropped_kept": 0}}

    prompt = PROMPT.format(max_q=MAX_QUESTIONS, max_d=MAX_DROPPED, conversation=convo)
    raw = chat_fn(model, [{"role": "user", "content": prompt}],
                  temperature=0, max_tokens=max_tokens, force_json=True)
    obj = _parse(raw)

    raw_q = [str(q).strip() for q in obj.get("questions", []) if str(q).strip()][:MAX_QUESTIONS]
    questions, seen_q = [], set()
    for q in raw_q:
        if not q.endswith("?"):
            continue                                        # must actually be a question
        bad = _introduces_clinical_vocab(q, patient_only)
        if bad:
            continue                                        # hard gate, same doctrine as polish.py
        key = q.lower()
        if key in seen_q:
            continue
        seen_q.add(key)
        questions.append({"id": f"q{len(questions)}", "text": q})

    raw_d = [d for d in obj.get("dropped", []) if isinstance(d, dict)][:MAX_DROPPED]
    dropped, seen_d = [], set()
    for d in raw_d:
        quote = str(d.get("quote", "")).strip()
        prompt_txt = str(d.get("prompt", "")).strip()
        if not quote or not prompt_txt:
            continue
        res = verify_span(patient_only, quote)
        if not res.ok:
            continue                                        # unverifiable quote -- never shown
        bad = _introduces_clinical_vocab(prompt_txt, patient_only)
        if bad:
            continue
        key = (res.source_start, res.source_end)
        if key in seen_d:
            continue
        seen_d.add(key)
        dropped.append({
            "id": f"d{len(dropped)}",
            "quote": res.matched_source_text,
            "prompt": prompt_txt,
            "start": res.source_start,
            "end": res.source_end,
        })

    return {
        "questions": questions,
        "dropped": dropped,
        "stats": {
            "n_questions_raw": len(raw_q), "n_questions_kept": len(questions),
            "n_dropped_raw": len(raw_d), "n_dropped_kept": len(dropped),
        },
    }


def verify_export_item(kind: str, item: dict, patient_only: str) -> dict | None:
    """Re-derives an approved item at EXPORT time rather than trusting the client's echo verbatim
    -- the same distrust the rest of the export path applies (tier-line TEXT is always re-derived
    from transcript+id server-side, never taken from the client as-is). A question is re-checked
    against the vocabulary gate only (it carries no quote to verify); a dropped-thread item has its
    quote RE-VERIFIED against the live transcript via span_check, so a tampered or stale quote
    cannot ride into the printed/emailed letter. Returns a clean dict to render, or None to drop.
    """
    if kind == "question":
        text = str(item.get("text", "")).strip()
        if not text or not text.endswith("?"):
            return None
        if _introduces_clinical_vocab(text, patient_only):
            return None
        return {"text": text}
    if kind == "dropped":
        quote = str(item.get("quote", "")).strip()
        prompt_txt = str(item.get("prompt", "")).strip()
        if not quote or not prompt_txt:
            return None
        res = verify_span(patient_only, quote)
        if not res.ok:
            return None
        if _introduces_clinical_vocab(prompt_txt, patient_only):
            return None
        return {"quote": res.matched_source_text, "prompt": prompt_txt}
    return None


def render_additions(questions: list[dict], dropped: list[dict]) -> str:
    """Bracketed text blocks for the exported letter, following pipeline.render_text's own
    convention exactly: every word the product adds is [bracketed]; nothing here claims to be the
    patient's verbatim words even though a dropped-thread quote genuinely is (the quote is shown
    inline, attributed, inside an app-authored sentence -- same shape as safety_net's cue quoting)."""
    out = []
    if questions:
        out.append("[Questions I might want to ask my doctor -- suggested by Understudy, "
                   "approved by me:]\n" + "\n".join(f"  - {q['text']}" for q in questions))
    if dropped:
        lines = [f"  - You mentioned: \"{d['quote']}\" -- {d['prompt']}" for d in dropped]
        out.append("[Something I mentioned but didn't finish -- approved by me to follow up:]\n"
                   + "\n".join(lines))
    return "\n\n".join(out)
