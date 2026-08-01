"""
paste_parser.py -- deterministic (NO model call) turn-splitter for pasted conversations.

THE FEATURE this serves: paste in a conversation someone else had ABOUT the patient or WITH an
AI, and turn it into material the patient confirms in their own words. Two source shapes:

  (a) a conversation the patient had with a general AI assistant (ChatGPT/Claude/Gemini/...),
      copied out of that product's UI or a shared-conversation page.
  (b) a note from a trusted advocate (family member, friend) who helps the patient.

WHY THIS IS PATTERN-MATCHING, NOT CLASSIFICATION
-------------------------------------------------
This module never guesses who wrote a sentence from its CONTENT or STYLE -- that is exactly the
move the build brief calls out as the dangerous one: "if you auto-attribute and get it wrong, AI-
authored text becomes a patient-authored line with a valid receipt." It only recognizes LITERAL
speaker labels that chat products already stamp into a copied transcript -- "You:", "You said:",
"ChatGPT said:", "Claude:", "Assistant:" and the like. Finding a label is a much weaker claim than
inferring authorship, and it is never trusted alone: parse_conversation() always reports its own
confidence, and the caller (understudy_app.py's /parse_paste, and the client JS) is required to
ask the patient once before anything is used ("these look like your messages, these look like the
assistant's -- right?"), or to fall back to unattributed, patient-recovers-by-hand chunks when no
label was recognized. Nothing in this module ever decides that a piece of text belongs to the
patient; it only proposes a grouping for the patient to confirm or reject at a glance.

ROLE, NOT CLAIM
---------------
An AI's own turns in (a) can never become sheet content even after confirmation -- it has no
observations of its own, only what the patient already told it, so its words can at most become a
QUESTION offered back to the patient (see understudy_app.py: askTopic()). An advocate's note in
(b) is different in kind (a human WHO ACTUALLY OBSERVED something) but is still never the
patient's own words, so it is handled the same conservative way here: parse_advocate_topics()
returns candidate TOPICS, never candidate sheet lines. Only the patient's own answer, typed live
after being asked, ever reaches the sheet -- through the exact same /sheet pipeline as everything
else the patient says.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Labels a conversational product uses for the HUMAN side. "you"/"me"/"i" cover the overwhelming
# majority of ChatGPT/Claude/Gemini copy-paste exports, which are written from the perspective of
# whoever did the copying -- which, dropped into THIS app's paste box, is the patient.
PATIENT_LABELS = {"you", "me", "i", "user", "myself", "human", "patient"}

# Labels a conversational product uses for ITS OWN side. Deliberately a fixed, closed vocabulary
# of product names and generic AI role-words -- not a guess from phrasing. An unrecognized label
# (a friend's actual name, say) never lands in this set, which is what forces the safe default
# (manual recovery) for anything this module cannot literally read off the page.
ASSISTANT_LABELS = {
    "chatgpt", "gpt", "gpt4", "gpt-4", "gpt4o", "gpt-4o", "chat gpt",
    "claude", "gemini", "bard", "copilot", "assistant", "ai", "bot", "the ai", "meta ai",
}


def _role_for(label: str) -> str | None:
    key = re.sub(r"\s+", " ", label.strip().strip(":").strip("*").strip().lower())
    if key in PATIENT_LABELS:
        return "patient"
    if key in ASSISTANT_LABELS:
        return "assistant"
    return None


# A line that is JUST a label -- optionally markdown-bold, optionally "said", optionally a
# trailing colon -- and nothing else. This is the common "copied out of the ChatGPT/Claude app"
# shape:
#   You said:
#   <message text, maybe several lines>
#   ChatGPT said:
#   <message text>
_LABEL_LINE = re.compile(
    r"^\s*\**\s*([A-Za-z][A-Za-z \-]{0,20}?)\s*\**\s*(?:said)?\s*:?\s*$", re.IGNORECASE)

# "You: hello there" -- label and content on the same line; the plain-text messaging-app shape.
_INLINE_LABEL = re.compile(r"^\s*\**\s*([A-Za-z][A-Za-z \-]{0,20}?)\s*\**\s*:\s+(\S.*)$")


def parse_conversation(text: str) -> dict:
    """Split a pasted two-party transcript into turns, without ever inferring a role from content.

    Returns:
      {"turns": [{"role_guess": "patient"|"assistant"|None, "label": str|None, "text": str}],
       "confidence": "high"|"low",
       "counts": {"patient": n, "assistant": n}}

    confidence == "high" only when at least one patient-labeled turn AND at least one assistant-
    labeled turn were both found -- i.e. this really does read as a two-party AI conversation, not
    a single stray line that happens to start with a recognized word. confidence == "low" means
    the labels were absent, one-sided, or unrecognized; the caller must not guess further and
    should fall back to unattributed chunks the patient recovers by hand.
    """
    lines = text.splitlines()

    # Pass 1: "label on its own line" shape.
    turns: list[dict] = []
    cur_role, cur_label, buf = None, None, []
    saw_any_label = False
    for line in lines:
        m = _LABEL_LINE.match(line)
        role = _role_for(m.group(1)) if m else None
        if m and role is not None:
            if buf and (cur_role is not None or cur_label is not None):
                turns.append({"role_guess": cur_role, "label": cur_label,
                              "text": "\n".join(buf).strip()})
            cur_role, cur_label, buf = role, m.group(1).strip(), []
            saw_any_label = True
            continue
        if line.strip():
            buf.append(line)
        elif buf:
            buf.append("")
    if buf and (cur_role is not None or cur_label is not None):
        turns.append({"role_guess": cur_role, "label": cur_label, "text": "\n".join(buf).strip()})
    turns = [t for t in turns if t["text"]]

    if not saw_any_label or len(turns) < 2:
        # Pass 2: "Label: content" on the same line -- one turn per matching line.
        turns = []
        for line in lines:
            m = _INLINE_LABEL.match(line)
            if not m:
                continue
            role = _role_for(m.group(1))
            if role is None:
                continue
            turns.append({"role_guess": role, "label": m.group(1).strip(),
                          "text": m.group(2).strip()})

    counts = {"patient": sum(1 for t in turns if t["role_guess"] == "patient"),
              "assistant": sum(1 for t in turns if t["role_guess"] == "assistant")}
    confidence = "high" if (counts["patient"] > 0 and counts["assistant"] > 0) else "low"

    if confidence == "low":
        # Nothing recognizable as a two-party AI conversation -- hand back unattributed chunks
        # (paragraph-sized, falling back to line-sized) so the UI can offer per-chunk manual
        # recovery instead of a guess this module cannot support.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        if len(chunks) <= 1:
            line_chunks = [c.strip() for c in text.splitlines() if c.strip()]
            if len(line_chunks) > 1:
                chunks = line_chunks
        if not chunks and text.strip():
            chunks = [text.strip()]
        turns = [{"role_guess": None, "label": None, "text": c} for c in chunks]

    return {"turns": turns, "confidence": confidence, "counts": counts}


def parse_advocate_topics(text: str) -> list[dict]:
    """Clause-level topic candidates from an advocate's note.

    Reuses pipeline.segment() -- the SAME clause-splitter the live transcript is chunked with --
    so an advocate's note gets identical granularity to anything the patient says out loud. These
    are topics, never claims: an advocate's word is never verbatim-verified against anything the
    PATIENT said, so it cannot honestly carry the sheet's "your own words, with a receipt"
    guarantee. It can only seed a question (see the module docstring).
    """
    import pipeline  # local import: gate/ is already on sys.path by the time this runs
    cands = pipeline.segment(text)
    return [{"id": c["sid"], "text": c["text"]} for c in cands]
