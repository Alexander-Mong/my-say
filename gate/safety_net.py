"""
safety_net.py -- SAFETY-NETTING, replacing the detect->confirm->stand-down binary.

WHY THE OLD DESIGN WAS REPLACED (measured 2026-07-26):
  - Emergency detection was 5 regexes an AI invented. Tested against realistic speech it missed
    4 of 4 gradual/euphemistic/indirect presentations (chest heaviness building across turns;
    "pressure in my chest" failing on word order; indirect suicidal ideation).
  - On explicit emergencies it DETECTED, asked "is this happening right now?", and then STOOD DOWN
    when the patient said no -- a regex matching the word "no". 2 of 5 real emergencies were talked
    out of escalation. The population this product serves minimises by default, so it asks the one
    question our users reliably answer wrong.
  - And the stand-down left NO TRACE: `escalations` never reached the letter, so the one artifact
    that goes to the doctor was the one place the flag didn't go.

WHAT REPLACES IT: NICE-style safety-netting. Instead of deciding escalate-vs-stand-down -- a
calibration the literature says even professionally-run triage misses (a validated protocol
under-triaged 27% of real ACS) -- we ALWAYS tell the person what would make something urgent and
what to do about it. No binary to get wrong.

STRUCTURE TAKEN FROM NICE NG12 (via NCBI Bookshelf NBK555330, accessed 2026-07-26):
  1.14.9  "Explain to people which symptoms to look out for and when they should return for
          re-evaluation. It may be appropriate to provide written information."
  1.15.2  review "may be planned within a time frame agreed with the person or patient-initiated
          if new symptoms develop, the person continues to be concerned, or their symptoms recur,
          persist or worsen."
So each entry below carries: what to look out for, when to act, and what action to take.

EVERY CLINICAL CRITERION BELOW IS QUOTED FROM A PUBLIC PATIENT-FACING SOURCE, WITH ITS URL.
Nothing here is written by an AI. That is the whole point of this file -- the thing it replaces was.

LICENSING, HONESTLY: nhs.uk content is Crown copyright under OGL-adjacent terms; quoting with
attribution for a non-commercial prototype is the assumption here, and nhs.uk's own reuse-terms
page has NOT been checked. Do not ship commercially without checking it. The NHS brand must never
be used in a way implying endorsement.

DELIBERATELY NOT INCLUDED: the C-SSRS suicide screener. Its own document states "If applied, it is
intended to be followed exactly according to the instructions and cannot be altered", and whether a
consumer product falls inside its free-use terms ("community and healthcare settings") is unclear.
Embedding it without confirmation from Columbia / The Research Foundation for Mental Hygiene would
be exactly the sloppiness this file exists to correct. Until then we show the 988 line, which is a
plain fact, and make no claim to screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetEntry:
    topic: str
    action: str = ""           # the source's OWN verbatim instruction. Added 2026-07-27 with the
                               # move to US federal sources, because they do not all say the same
                               # thing: NHLBI and MedlinePlus say "call 9-1-1"/"call 911" for
                               # cardiac and stroke, but NIDDK and MedlinePlus say "seek medical
                               # help right away" for GI bleeding and sepsis -- one notch below an
                               # emergency-services instruction. The old renderer hardcoded a
                               # "call 911" header for every topic, which would have UPGRADED those
                               # two. Same class of error as putting a caveat under an action
                               # header: the product must not escalate what the source said.
    emergency: tuple = ()      # verbatim criteria that go with `action`
    urgent: tuple = ()         # verbatim "get help today / tell your provider" criteria
    note: tuple = ()           # verbatim CAVEATS -- true statements that are not action criteria.
                               # Rendering a caveat under an action header (as the first draft did)
                               # turns a nuance into an instruction. Kept separate.
    source: str = ""
    publisher: str = ""        # who to attribute in the letter; no longer hardcoded to "NHS"
    accessed: str = "2026-07-27"


# ---------------------------------------------------------------------------------------------
# SOURCE SWAP, 2026-07-27: nhs.uk -> US federal (NHLBI, MedlinePlus health-topic pages, NIDDK).
#
# WHY. Two reasons, and the second is the better one.
#   1. Licensing. nhs.uk text is lawful to reuse commercially, but only under OGL v3.0 plus the
#      NHS website terms -- per-instance attribution, a date stamp or a <=7-day refresh, a
#      prominent OGL notice, and no charging users for access. Those conditions cannot be absorbed
#      by an MIT grant, so the repo could not be uniformly permissive. US federal works are public
#      domain under 17 U.S.C. 105 with no conditions at all.
#   2. Fit. We were already localising 999 -> 911 and refusing to map NHS 111 onto a service that
#      does not exist here, i.e. carrying England-scoped clinical text into a US-facing product and
#      papering over the seam. Sourcing it from US federal pages removes the seam instead.
#
# EXCLUSION THAT MATTERS: medlineplus.gov/ency/* is the A.D.A.M. Medical Encyclopedia -- COPYRIGHTED
# third-party content, explicitly carved out of MedlinePlus's public-domain statement. Nothing here
# comes from /ency/. The rule is "federally authored", not ".gov".
#
# TWO CLINICAL DIFFERENCES FROM THE NHS TEXT, FLAGGED RATHER THAN SMOOTHED (Alex's call to keep or
# change -- these were NOT silently harmonised):
#   * HEADACHE. NHS-style copy routes a sudden "worst ever" headache straight to emergency services.
#     MedlinePlus does not: it reserves "get medical help right away" for head injury, or headache
#     WITH stiff neck / fever / confusion / loss of consciousness / eye or ear pain, and routes a
#     sudden severe headache on its own to "let your health care provider know". That is a real
#     routing change, not a wording change. The federal threshold is what is encoded below.
#   * BREATHING. No federal patient-facing page reachable gives a general "severe difficulty
#     breathing -> emergency" line; medlineplus.gov/breathingproblems.html carries no emergency
#     threshold at all. The text below is verbatim and public domain but was written in a COPD
#     context, and is marked as such so it is never mistaken for a general rule.
#
# `us_localise` is retained and now a near no-op: federal text already says 911. Note the sources
# are internally inconsistent about spelling it ("9-1-1" at NHLBI, "911" at MedlinePlus) and BOTH
# ARE PRESERVED AS WRITTEN, because normalising them would be editing a quotation for tidiness.
# ---------------------------------------------------------------------------------------------
NET = [
    NetEntry(
        topic="chest",
        action="Call 9-1-1 for emergency medical care, even if you are not sure that you're "
               "having a heart attack.",
        emergency=("Chest pain, heaviness, or discomfort in the center or left side of the chest",
                   "Pain or discomfort in one or both arms, your back, shoulders, neck, jaw, or "
                   "above your belly button",
                   "Shortness of breath when resting or doing a little bit of physical activity",
                   "Sweating a lot for no reason",
                   "Light-headedness or sudden dizziness",
                   "Rapid or irregular heartbeat"),
        note=("Never delay calling 9-1-1, taking aspirin or doing anything else you think might "
              "help.",),
        source="https://www.nhlbi.nih.gov/health/heart-attack/symptoms",
        publisher="NHLBI",
    ),
    NetEntry(
        topic="chest_atypical",
        # Kept as a SEPARATE topic because a criteria list keyed to crushing chest pain misses
        # people. NO PERCENTAGE IS CLAIMED anywhere: the commonly-cited "20-30% present atypically"
        # figure is CONTESTED and appears on no federal patient-facing page, so it is not asserted.
        # The two caveats below are attached by their source to specific groups -- unusual tiredness
        # in women, silent heart attacks in older adults and people with high blood sugar/diabetes
        # -- and are NOT generalised beyond what the source says.
        action="Call 9-1-1 for emergency medical care, even if you are not sure that you're "
               "having a heart attack.",
        emergency=("Feeling unusually tired for no reason, sometimes for days",
                   "Nausea (feeling sick to the stomach) and vomiting",
                   "Shortness of breath. Sometimes this is your only symptom."),
        note=("Feeling unusually tired for no reason, sometimes for days (this is more common in "
              "women)",
              "Silent heart attacks are more common in older adults and in people who have high "
              "blood sugar or diabetes."),
        source="https://www.nhlbi.nih.gov/health/heart-attack/symptoms",
        publisher="NHLBI / MedlinePlus",
    ),
    NetEntry(
        topic="stroke",
        action="If you think that you or someone else is having a stroke, call 911 right away.",
        emergency=("Face drooping on one side when smiling.",
                   "Arm weakness occurs when the arms are raised, and one arm drifts downward.",
                   "Speech is slurred or strange.",
                   "Sudden numbness or weakness of the face, arm, or leg (especially on one side "
                   "of the body)",
                   "Sudden confusion, trouble speaking, or understanding speech",
                   "Sudden trouble seeing in one or both eyes",
                   "Sudden difficulty walking, dizziness, loss of balance or coordination",
                   "Sudden severe headache with no known cause"),
        note=("Every minute counts during a stroke.",),
        source="https://medlineplus.gov/stroke.html",
        publisher="MedlinePlus",
    ),
    NetEntry(
        topic="breathing",
        # ⚠ CONTEXT CAVEAT, deliberate and load-bearing: these strings are verbatim and public
        # domain but were written on the COPD topic page. No reachable federal patient-facing page
        # states a GENERAL "severe difficulty breathing -> emergency" threshold -- the generic
        # Breathing Problems page has none. The letter therefore attributes them to their real
        # context rather than presenting them as a universal rule.
        action="You should get emergency care if you have severe symptoms, such as trouble "
               "catching your breath or talking.",
        urgent=("Call your health care provider if your symptoms are getting worse or if you have "
                "signs of an infection, such as a fever.",),
        note=("When you're short of breath, it's hard or uncomfortable for you to take in the "
              "oxygen your body needs. You may feel as if you're not getting enough air.",
              "(The emergency wording above is quoted from MedlinePlus's COPD page; no federal "
              "patient-facing page states a general breathing-emergency threshold.)"),
        source="https://medlineplus.gov/copd.html",
        publisher="MedlinePlus (COPD page)",
    ),
    NetEntry(
        topic="anaphylaxis",
        action="If someone is having a serious allergic reaction, call 911. If an auto-injector "
               "is available, give the person the injection right away.",
        # The organ labels ("Throat: ...") are part of the source's own wording and are kept, so
        # nothing here is a paraphrase.
        emergency=("Mouth: itching, swelling of the lips or tongue",
                   "Throat: itching, tightness, trouble swallowing, swelling of the back of the "
                   "throat",
                   "Chest: shortness of breath, coughing, wheezing, chest pain or tightness",
                   "Heart: weak pulse, passing out, shock",
                   "Nervous system: dizziness or fainting"),
        source="https://medlineplus.gov/anaphylaxis.html",
        publisher="MedlinePlus",
    ),
    NetEntry(
        topic="gi_bleed",
        # NOT "call 911" -- NIDDK says "seek medical help right away", one notch below. Preserved.
        action="If you have symptoms of acute or severe GI bleeding, seek medical help right away.",
        emergency=("black or tarry stool",
                   "dark or bright red blood mixed with stool",
                   "bright red blood in vomit",
                   "vomit that looks like coffee grounds",
                   "fainting or feeling lightheaded or dizzy"),
        note=("Shock is life-threatening. If you have symptoms of shock, seek emergency medical "
              "help right away.",),
        source="https://www.niddk.nih.gov/health-information/digestive-diseases/"
               "gastrointestinal-bleeding/symptoms-causes",
        publisher="NIDDK",
    ),
    NetEntry(
        topic="sepsis",
        # Also NOT "call 911" in the source. Preserved as written.
        action="It's important to get medical care right away if you think you might have sepsis "
               "or if your infection is not getting better or is getting worse.",
        emergency=("Rapid breathing and heart rate",
                   "Shortness of breath",
                   "Confusion or disorientation",
                   "Extreme pain or discomfort",
                   "Fever, shivering, or feeling very cold",
                   "Clammy or sweaty skin"),
        note=("Sepsis is a life-threatening medical emergency. Without quick treatment, it can "
              "lead to tissue damage, organ failure, and even death.",),
        source="https://medlineplus.gov/sepsis.html",
        publisher="MedlinePlus",
    ),
    NetEntry(
        topic="headache",
        # ⚠ THRESHOLD DIFFERS FROM THE NHS TEXT THIS REPLACED -- see the block comment above.
        # MedlinePlus does NOT route a sudden severe headache on its own to emergency services.
        action="Get medical help right away if you have a headache after a blow to your head, or "
               "if you have a headache along with a stiff neck, fever, confusion, loss of "
               "consciousness, or pain in the eye or ear.",
        urgent=("Let your health care provider know if you have sudden, severe headaches.",),
        note=("If a brain aneurysm bursts, symptoms can include a sudden, severe headache, nausea "
              "and vomiting, stiff neck, loss of consciousness, and signs of a stroke.",),
        source="https://medlineplus.gov/headache.html",
        publisher="MedlinePlus",
    ),
]

CRISIS = ("If you are thinking about harming yourself, you can call or text 988 (the 988 Suicide "
          "and Crisis Lifeline) any time, day or night. It is free and confidential.")
CRISIS_SRC = "https://988lifeline.org/"


def us_localise(text: str) -> str:
    """Only the number dialled changes. Criteria wording is never altered."""
    return text.replace("999", "911").replace("A&E", "the emergency room")


# Which topics to surface. Deliberately GENEROUS and keyword-light: this is not a detector and
# makes no claim to detect. It decides what safety information to INCLUDE, and including too much
# costs the reader a few lines, while including too little costs what the old design cost.
TOPIC_CUES = {
    "chest": ("chest", "heart", "arm", "jaw", "sweat", "clammy", "pressure", "tight", "squeez",
              "indigestion", "heavy"),
    "chest_atypical": ("chest", "heart", "tired", "indigestion", "sick", "nausea", "panic"),
    # "arm" alone is NOT a stroke cue -- it fired on "goes down my arm" (a cardiac description)
    # and printed stroke criteria into a chest-pain letter. Stroke needs a stroke-specific sign.
    "stroke": ("droop", "speech", "slur", "one side", "face", "numb", "vision", "confus"),
    "breathing": ("breath", "breathe", "breathing", "wheez", "gasp", "cough", "winded", "puff"),
    "anaphylaxis": ("allerg", "swell", "throat", "tongue", "sting", "rash", "hives", "reaction"),
    "gi_bleed": ("blood", "bleed", "stool", "poo", "vomit", "sick", "black", "stomach", "bowel"),
    "sepsis": ("fever", "temperature", "shiver", "confus", "rash", "infection", "chills"),
    # "forehead" listed explicitly: boundary matching (relevant_topics) no longer reaches the
    # "head" inside it.
    "headache": ("headache", "head", "forehead", "migraine", "worst"),
}
# SELF-HARM CUES — ONE SOURCE OF TRUTH, deliberately.
# Bug found 2026-07-26 before shipping: the agent had its own regex list and this module had its
# own substring list, and their coverage was DISJOINT. "I've been thinking about killing myself"
# got 988 spoken aloud but left NO trace in the letter (the substring "kill myself" does not occur
# in "killing myself"); "they'd be better off without me" did the exact opposite. A person saying
# the most explicit thing possible got the weakest record. Regexes now, in one place, and the
# agent imports THIS rather than keeping a second copy.
#
# NOT a screening instrument and not claimed to be one. The validated instrument for this is the
# C-SSRS, which we do not embed (its own document forbids alteration and its consumer-product
# licensing is unresolved). This list decides only whether to SHOW A PHONE NUMBER and whether to
# carry the person's own words forward — both of which are safe to over-trigger.
SELF_HARM_PATTERNS = (
    r"\bkill(ing)?\s+(myself|my\s?self)\b",
    r"\bend(ing)?\s+(it|my\s+life)\b",
    r"\bsuicid\w*",
    r"\b(don'?t|do\s+not|didn'?t)\s+want\s+to\s+(be\s+alive|live|wake\s+up|be\s+here)\b",
    r"\bnot\s+want\s+to\s+(be\s+alive|live|wake\s+up)\b",
    # "off" optional: the phrasing that actually occurs in the s3 test script is "better without
    # me around" -- the stricter form missed it (found 2026-08-01 review).
    r"\bbetter\s+(off\s+)?without\s+me\b",
    r"\b(hurt|harm)(ing)?\s+(myself|my\s?self)\b",
    r"\btook\s+(too\s+many|a\s+bunch\s+of|all\s+(the|my)?)\s*(pills|meds|tablets)\b",
    r"\b(be|being)\s+a\s+burden\b",
    r"\bno\s+(point|reason)\s+(in\s+)?(going\s+on|living|any\s?more)\b",
    r"\bthinking\s+about\s+how\s+I'?d\s+do\s+it\b",
    r"\bwish(ed)?\s+I\s+(was|were)\s+dead\b",
)


def relevant_topics(text: str) -> list[str]:
    import re as _re
    t = " ".join(text.lower().split())
    # Left word-boundary, not raw substring: "ahead" must not fire "head" (it printed a
    # headache/aneurysm block into a stomach-pain letter, found 2026-08-01), "pharmacy" must
    # not fire "arm", "shampoo" must not fire "poo", "surface" must not fire "face". Cues stay
    # prefix-stems on purpose: r"\bsqueez" still matches "squeezing", "confus" -> "confused".
    return [e.topic for e in NET
            if any(_re.search(r"\b" + _re.escape(c), t) for c in TOPIC_CUES.get(e.topic, ()))]


def mentions_self_harm(text: str) -> bool:
    import re as _re
    t = " ".join(text.lower().split())
    # re.I explicitly: the text is lower-cased here but the patterns contain capitals ("I"), and
    # relying on the two agreeing silently lost "I wish I was dead" in testing. Matching must not
    # depend on a casing convention holding across two places.
    return any(_re.search(p, t, _re.I) for p in SELF_HARM_PATTERNS)


def build(transcript: str, always_include: bool = True, app_name: str = "My Say") -> str:
    """The safety-netting section that goes IN THE LETTER, every time.

    always_include: even when nothing matched, emit the general line. A letter that sometimes
    carries safety information and sometimes doesn't teaches the reader that its absence means
    'nothing to worry about' -- which is a claim we must never make.

    app_name: the product name to print in the header line. Passed in by the caller (ultimately
    understudy_app.APP_NAME) rather than hardcoded here, so a rename is the one-line edit the
    APP_NAME constant exists to guarantee, not a grep across every module that mentions the name.
    """
    topics = relevant_topics(transcript)
    lines = [f"[The following is general safety information, included with every {app_name} letter.",
             " It is not a diagnosis and nothing here was decided by looking at this person's case.]"]
    # Entries that give the SAME instruction are merged under one header. chest and
    # chest_atypical both quote NHLBI's "Call 9-1-1 for emergency medical care..." and both fire on
    # a chest transcript; printing that sentence twice reads as two different instructions rather
    # than one with more criteria under it.
    seen, groups = set(), []
    for e in NET:
        if e.topic not in topics or e.topic in seen:
            continue
        seen.add(e.topic)
        for g in groups:
            if g["action"] == e.action:
                g["entries"].append(e)
                break
        else:
            groups.append({"action": e.action, "entries": [e]})

    for g in groups:
        ents = g["entries"]
        lines.append("")
        # The HEADER is the source's own instruction, verbatim, rather than a house phrase.
        # Previously this hardcoded "Get emergency help now (call 911)" for every topic, which
        # would silently upgrade NIDDK's "seek medical help right away" into a 911 instruction.
        # Attribution is deduped BY SOURCE URL, not by the publisher+URL string: chest and
        # chest_atypical quote the same NHLBI page under slightly different publisher labels, and
        # keying on the label printed the identical URL twice.
        by_src, crit, notes, urgent = {}, [], [], []
        for e in ents:
            pubs = by_src.setdefault(e.source, [])
            for p in (e.publisher or "").split(" / "):
                if p and p not in pubs:
                    pubs.append(p)
            crit += [c for c in e.emergency if c not in crit]
            notes += [n for n in e.note if n not in notes]
            urgent += [u for u in e.urgent if u not in urgent]
        attrib = " · ".join(
            (f"{' / '.join(pubs)}, {src}" if pubs else src) for src, pubs in by_src.items())
        if g["action"]:
            lines.append(f"[{us_localise(g['action'])} — {attrib}:]")
            for c in crit:
                lines.append(f"  - {us_localise(c)}")
        elif notes:
            lines.append(f"[Worth knowing — {attrib}:]")
        for n in notes:
            lines.append(f"  [worth knowing: {n}]")
        if urgent:
            # No house header here either: each urgent string already carries its own instruction
            # ("Call your health care provider if..."), so adding one would double the imperative.
            lines.append(f"[Also — {attrib}:]")
            for c in urgent:
                lines.append(f"  - {us_localise(c)}")
    if mentions_self_harm(transcript):
        lines += ["", f"[{CRISIS} — {CRISIS_SRC}]"]
    if len(lines) == 2 and always_include:
        lines += ["", "[If anything gets suddenly worse, or you become frightened by it, do not wait "
                  "for your appointment — call 911 or go to the emergency room.]"]
    lines += ["", "[If new symptoms appear, or these get worse, or you stay worried, seek "
              "re-evaluation rather than waiting for your scheduled visit. — structure per NICE "
              "NG12 1.14.9 / 1.15.2]"]
    return "\n".join(lines)


def flagged_words(transcript: str) -> list[str]:
    """The patient's OWN sentences that caused a topic to be included, verbatim.

    These go in the letter so the doctor sees what the person said, in their words -- including
    anything they later waved off. The old design's fatal gap was that a stand-down erased this.
    """
    out = []
    for sent in [s.strip() for s in transcript.replace("\n", ". ").split(".") if s.strip()]:
        if relevant_topics(sent) or mentions_self_harm(sent):
            out.append(sent)
    return out


if __name__ == "__main__":
    demo = ("my chest has been feeling heavy today\nit kind of goes into my shoulder and down my arm\n"
            "and I've been sweating a lot, more than normal\nhonestly it's happening right now")
    print(build(demo))
    print("\n--- patient's own words that triggered inclusion ---")
    for w in flagged_words(demo):
        print("  •", w)
