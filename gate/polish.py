"""polish.py -- runtime "Say it better for me": drafter prompt + deterministic gates.

Ports the TESTED machinery from intake/anvil/rewrite_build.py + rewrite_grade.py (80-cell
test, 2026-07-27, graded in intake/anvil/REWRITE_RESULTS.md: 0/80 clinical-vocab escalations
across four model families; e2b met the sealed buildable bar at 10/20 all-gates). The PROMPT
and lexicons below are copied from that test verbatim -- the evidence covers THIS prompt.

Runtime difference vs the test: the test hand-labeled each utterance's number/negation/hedge
features; at runtime they are AUTO-extracted from the source line, so those gates are
approximations and their failures surface as CAUTION strings for the patient's per-line
review (the test's own conclusion: per-unit patient review is load-bearing regardless of
gate scores). The vocabulary gate needs no labels -- it compares draft against source
directly -- and stays HARD: a draft that introduces any clinical term is suppressed
entirely, never shown. The model cannot edit anything silently; every accepted draft is
labeled on the sheet and carries a receipt to the verbatim original.
"""
import json
import re

PROMPT = """You help a patient say something to their doctor more clearly. Rewrite what they said \
into one or two plain, first-person sentences.

STRICT RULES:
1. Keep every number, duration and count EXACTLY as said (you may write digits as words or words \
as digits, nothing else).
2. If they said something is NOT the case, the rewrite must still clearly say it is not.
3. Use NO medical or clinical word the person did not use themselves. Their everyday words are \
correct: keep "flutter", "leak", "tips" — do not translate them.
4. If they hedged ("probably", "maybe", "I think", "not sure"), the rewrite must keep that \
uncertainty. Never make them sound more certain than they were.
5. Stay "I" and "my" — never "the patient".
6. Do not add advice, reassurance, or anything they did not say.

Example:
They said: "it's like my knee gives, going DOWN stairs mostly, not up, been on and off since \
maybe March, I think it's fine really"
Rewrite: "My knee gives way now and then, mostly going down stairs, not up. It has been on and \
off since around March. I think it is probably fine, but I wanted to mention it."

Example:
They said: "not chest pain exactly more a squeeze? twice at work, few seconds, gone. didn't tell \
anyone"
Rewrite: "Twice at work I have felt a squeezing feeling in my chest — not pain exactly — for a \
few seconds each time, then it went away. I have not told anyone until now."

They said: "{utterance}"
Reply ONLY with JSON: {{"rewrite": "your one-or-two sentence rewrite"}}"""

NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twice": "2",
             "second": "2", "thirty": "30", "couple": "2"}
NEG_TOKENS = {"not", "no", "never", "n't", "dont", "don't", "didnt", "didn't", "isnt", "isn't",
              "wasnt", "wasn't", "doesnt", "doesn't", "without", "barely", "nothing"}
HEDGES = ("probably", "maybe", "i think", "not sure", "kind of", "sort of", "i guess",
          "might", "or so", "-ish", "around ", "about ", "wanted to mention", "could be")
CLINICAL = ("palpitation", "arrhythmia", "syncope", "vertigo", "presyncope", "dyspnea",
            "dyspnoea", "angina", "oedema", "edema", "urinary incontinence", "stress incontinence",
            "dysphagia", "odynophagia", "anhedonia", "depression", "melanoma", "lesion",
            "claudication", "aura", "seizure", "epilep", "dysuria", "uti", "urinary tract",
            "tremor", "tinnitus", "pulsatile", "peripheral", "fluid retention", "nocturnal",
            "hematuria", "haematuria", "urticaria", "hives", "cardiac", "neurological",
            "paresthesia", "paraesthesia", "myoclon", "hypnic", "gustatory", "olfactory")
DIAGNOSIS_GRADE = ("angina", "melanoma", "seizure", "epilep", "arrhythmia", "claudication",
                   "uti", "urinary tract", "urticaria")
_STOP = {"a", "an", "the", "my", "me", "i", "it", "its", "is", "was", "be", "been", "really",
         "exactly", "just", "like", "that", "this", "to", "of", "in", "on", "at", "so", "and",
         "but", "or", "very", "quite", "bit", "more", "them", "they"}


def _toks(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def build_prompt(utterance):
    return PROMPT.format(utterance=utterance)


def parse_rewrite(raw):
    """Tolerant JSON extraction, same shape as the test's parse_cell."""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        return json.loads(t).get("rewrite")
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1]).get("rewrite")
            except Exception:
                return None
    return None


def auto_truth(source):
    """Best-effort feature extraction the test did by hand."""
    toks = _toks(source)
    numbers = [t for t in toks if t.isdigit() or t in NUM_WORDS]
    negations = []
    for i, t in enumerate(toks):
        if t in NEG_TOKENS:
            for j in range(i + 1, min(i + 4, len(toks))):
                if toks[j] not in _STOP and toks[j] not in NEG_TOKENS:
                    negations.append(toks[j])
                    break
    hedged = any(h in source.lower() for h in HEDGES)
    return {"numbers": numbers, "negations": negations, "hedges": hedged}


def run_gates(draft, source):
    """Returns {suppressed, cautions}. Vocab introduction = hard suppression; the rest =
    caution strings for the patient's review."""
    low = draft.lower()
    src_low = source.lower()
    vocab = [w for w in CLINICAL if w in low and w not in src_low]
    if vocab:
        return {"suppressed": True,
                "cautions": ["the draft added a clinical word (" + ", ".join(vocab)
                             + ") — it was withheld; your own words stay"]}
    t = auto_truth(source)
    dtoks = _toks(draft)
    cautions = []
    for n in t["numbers"]:
        forms = {n}
        if n in NUM_WORDS:
            forms.add(NUM_WORDS[n])
        forms |= {w for w, d in NUM_WORDS.items() if d == n}
        if not any(f in dtoks for f in forms):
            cautions.append("check: the number '" + n + "' may be missing")
    for subj in t["negations"]:
        hit = False
        for i, x in enumerate(dtoks):
            if x == subj and set(dtoks[max(0, i - 6):i + 7]) & NEG_TOKENS:
                hit = True
                break
        if not hit:
            cautions.append("check: 'not " + subj + "' may have been lost")
    if t["hedges"] and not any(h in low for h in HEDGES):
        cautions.append("check: your uncertainty (probably/maybe) is gone — "
                        "this sounds more certain than you did")
    # first-person check on tokens so contractions count ("I'm", "I've" — the test's original
    # substring check missed them and threw false cautions on perfectly first-person drafts)
    first_person = any(x == "i" or x.startswith("i'") or x == "my" for x in dtoks)
    if ("the patient" in low) or not first_person:
        cautions.append("check: it stopped sounding like you (no 'I'/'my')")
    return {"suppressed": False, "cautions": cautions}
