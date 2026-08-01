"""probe.py -- "Ask me more about this": deepening questions for ONE patient-chosen line.

WHY THIS SHAPE
--------------
The app has no model of what is clinically important -- "What matters most" (pipeline.PRIMARY_
CUTOFF) is literally just the first four model-selected ids, a display convenience, not a
priority claim. Rather than build a severity model to decide what to probe deeper on, the PATIENT
points at the one thing that matters to them by clicking it. This module only decides WHAT KIND
OF QUESTION to ask about the thing they clicked, using the same fixed elements every doctor is
trained to ask about, one at a time -- see ELEMENTS below.

WHY TEMPLATED, NOT MODEL-AUTHORED
----------------------------------
Every other model call in this app produces either an id selection (pipeline.py, which the model
never emits patient TEXT for) or a gated draft the patient explicitly approves before it counts
(polish.py). A "what should I ask next" call would be the first place a model's own PROSE reaches
the patient with no receipt and no accept/reject gate -- exactly the "model writes the note"
architecture the rest of this file's siblings exist to avoid. Templated questions carry the same
guarantee: the phrasing is fixed, reviewed text; the only new content that ever reaches the sheet
is what the patient types back, verified through span_check.verify_span like every other line
(see pipeline.append_answer). It also means this feature costs ZERO extra model calls -- load-
bearing on a machine where Ollama serializes across every concurrent user of it.

THE ELEMENTS
------------
The CMS "HPI elements" (location, quality, severity, duration, timing, context, modifying
factors, associated signs/symptoms) are the eight elements a history-of-present-illness note is
graded on. OPQRST and OLDCARTS are mnemonics for the same underlying set, plus onset -- included
here too because "when did this start" is almost always the doctor's own first question and
grounds everything asked after it. This list is that merged, de-duplicated set; nothing invented.

ONE AT A TIME, NEVER A BUNDLE
------------------------------
next_question() returns exactly one element's question, matching the app's existing chat
discipline (CHAT_SYS: "ONE detail per question, never several bundled together"). The caller
(the client) tracks which elements have already been asked *for this line* and passes that list
back in; next_question() also skips an element outright when the line's own words already answer
it (a light keyword heuristic -- see _AUTO_SKIP) so the patient is never asked to repeat
something they already said.
"""
from __future__ import annotations

import re

# Order matters: this IS the sequence questions are offered in, absent any auto-skip. Onset first
# (matches how a doctor usually opens), then the physical description elements, then the
# situational ones.
ELEMENTS = [
    {"key": "onset", "label": "Onset",
     "q": "When did this first start — even roughly?"},
    {"key": "location", "label": "Location",
     "q": "Where exactly do you feel it?"},
    {"key": "quality", "label": "Quality",
     "q": "What does it actually feel like — can you describe the sensation?"},
    {"key": "severity", "label": "Severity",
     "q": "On a rough scale, how bad is it when it happens?"},
    {"key": "duration", "label": "Duration",
     "q": "Once it starts, how long does it usually last?"},
    {"key": "timing", "label": "Timing / frequency",
     "q": "How often does it happen, or is there a pattern to when?"},
    {"key": "modifying", "label": "What changes it",
     "q": "Is there anything that makes it better, or makes it worse?"},
    {"key": "associated", "label": "Associated symptoms",
     "q": "Does anything else happen along with it?"},
    {"key": "context", "label": "Context",
     "q": "What's usually going on right before or when it happens?"},
]
_BY_KEY = {e["key"]: e for e in ELEMENTS}

# Light, best-effort signal that the line's OWN words already cover an element -- so the patient
# is never asked to repeat something they already told the app. False negatives (missing a case
# and asking anyway) are harmless -- worst case, one extra question; false positives (skipping an
# element that was NOT really covered) are the risk, so these stay narrow and literal rather than
# clever. Mirrors the spirit of polish.py's auto_truth: cheap regex over the source, no model call.
_TIME_RE = re.compile(
    r"\b(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|"
    r"since|ago|started|been (going|happening|doing) (on|this) for)\b", re.I)
_SEVERITY_RE = re.compile(
    r"\b(mild|moderate|severe|terrible|awful|unbearable|worst|out of (ten|10)|scale of)\b", re.I)
_MODIFY_RE = re.compile(
    r"\b(makes? it (better|worse)|helps?|relieves?|when i (stand|sit|lie|walk|eat|move|bend)|"
    r"after i (eat|stand|walk|exercise))\b", re.I)

_AUTO_SKIP = {
    "onset": _TIME_RE,
    "duration": _TIME_RE,
    "timing": _TIME_RE,
    "severity": _SEVERITY_RE,
    "modifying": _MODIFY_RE,
}


def next_question(text: str, asked: list[str] | None = None) -> dict | None:
    """The next un-asked, not-already-evident element for this line, in fixed clinical order.

    Returns {"element", "label", "question"} or None once every element is asked or skipped --
    the caller shows "you've covered the standard details for this one" in that case.
    """
    asked_set = set(asked or [])
    for e in ELEMENTS:
        if e["key"] in asked_set:
            continue
        pat = _AUTO_SKIP.get(e["key"])
        if pat and text and pat.search(text):
            continue
        return {"element": e["key"], "label": e["label"], "question": e["q"]}
    return None


def label_for(key: str) -> str:
    e = _BY_KEY.get(key)
    return e["label"] if e else (key or "")
