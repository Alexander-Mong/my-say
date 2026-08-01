"""
test_span_check.py -- the span-check gate's boundary cases, as executable docs.

Run:  python -m pytest test_span_check.py -q      (if pytest is around)
  or: python test_span_check.py                   (self-contained runner)

Each case below is a concrete point on the "curated verbatim <-> paraphrase"
line. This file IS the artifact to SIT WITH when tuning the whitelist/tokenizer
(Alex, 2026-07-25): if a verdict here feels wrong, that's the whitelist telling
you where its edge is.
"""
from span_check import verify_span


# A realistic messy STT transcript, with the artifacts real STT emits.
RAW = (
    "so um yeah I've been having like this chest pain, you know, "
    "for about [inaudible] three weeks now and it's it's not going away. "
    "[00:41] I don't have any shortness of breath though."
)


# (name, candidate, expect_ok)  -- expect_ok True => must PASS, False => must FLAG
CASES = [
    # ---- MUST PASS: legitimate curated verbatim -------------------------
    ("clean verbatim",            "I've been having this chest pain",           True),
    ("drop single filler 'um'",   "yeah I've been having",                      True),
    ("drop filler 'like'",        "having this chest pain",                     True),
    ("drop multiword 'you know'", "chest pain for about",                       True),
    ("drop capture artifact",     "for about three weeks now",                  True),  # [inaudible] dropped
    ("drop timestamp",            "going away. I don't have any",               True),  # [00:41] dropped
    ("drop stutter repeat",       "not going away",                             True),  # "it's it's" repeat
    ("case-insensitive",          "SO um YEAH i've been having",                True),

    # ---- MUST FLAG: paraphrase sneaking in ------------------------------
    ("added token",               "having sharp chest pain",                    False),  # 'sharp' not in raw
    ("synonym swap",              "having this chest tightness",                False),  # tightness != pain
    ("tense change",              "I have been have chest pain",                False),  # 'have' for 'having'
    ("number change",            "for about four weeks now",                    False),  # four != three
    ("reordered tokens",          "chest pain having been",                     False),
    ("MEANINGFUL DELETION 'not'", "it's going away",                            False),  # dropped 'not' inside span
    ("negation flip via drop",    "I do have shortness of breath",              False),  # dropped "don't ... any"
    ("fabricated symptom",        "I also have dizziness",                      False),  # nothing in raw
]


def run():
    passed, failed = 0, 0
    for name, cand, expect_ok in CASES:
        r = verify_span(RAW, cand)
        good = (r.ok == expect_ok)
        passed += good
        failed += (not good)
        mark = "ok  " if good else "FAIL"
        verdict = "PASS" if r.ok else "FLAG"
        extra = (f"[{r.source_start}:{r.source_end}]" if r.ok else r.flags[0])
        print(f"  {mark} {name:28} -> {verdict:4} {extra}")
        if not good:
            print(f"       !! expected {'PASS' if expect_ok else 'FLAG'}, got {verdict}")
    print(f"\n  {passed}/{passed + failed} boundary cases behave as documented.")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
