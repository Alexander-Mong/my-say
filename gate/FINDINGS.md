# Gate + extraction — findings (2026-07-25, ~01:10 EDT)

> **SNAPSHOT — PARTLY SUPERSEDED. Read this as a decision trail, not as current status.**
>
> This file is a dated snapshot from **2026-07-25**, written *before* the Gemma 4 e2b runs were
> possible (Ollama was unavailable at the time). Everywhere below that says the demo model is
> **"untested"** or **"Ollama-blocked"** — including the caveats and the "Still open" list — that
> is **no longer true**: the e2b runs were subsequently done, and the current figures are the ones
> in the repo README's Evidence summary. Conclusions here about *architecture* (extract-and-verify
> beats abstractive summarization; the model must never re-emit the patient's words) still stand
> and are what the shipped design is built on. It is kept unedited so the reasoning that produced
> the design is visible.
>
> **The harness scripts named below** — `slice.py`, `hard_probe.py`, `compare.py`, `glue_compare.py`,
> `a3.py`, `batch_extract.py`, `build_explorer.py` — are **exploration tooling and are not shipped
> in this repository.** They were scratch instruments for the runs described here; the code that
> ships is in `gate/` and `understudy_app.py`.

Evidence backing the extractive / "curator not author" commitment. Harness:
`slice.py` (clean rambles) + `hard_probe.py` (paraphrase-tempting rambles) → `span_check.py` gate.

## Result 1 — FIDELITY IS SOLVED FOR CAPABLE MODELS (verified, not inferred)

When told "copy exact words, don't paraphrase," capable instruction-tuned models obey, and the
gate proves each span verbatim:

| Input set | Models | Spans | Fabrication (gate FLAGs) |
|---|---|---|---|
| Clean rambles (`slice.py`) | Qwen3-30B, Llama-3.3-70B | 34 | **0%** |
| Hard: dialect / non-native / folk terms (`hard_probe.py`) | Gemma-3-27b, Qwen3-30B, Llama-3.3-70B | 87 | **0%** |

Crucially, verified by EYEBALL (not just the count) that the hard phrasings were *selected and kept
exactly*, not quietly standardized:
`"is problem with leg"` · `"my sugar been running high"` (not "blood glucose") · `"the water pill"`
(not "diuretic") · `"took her feet in the end"` · `"too much hot in the body"` · `"I no sleep good"`.
The "helpful" instinct to clean up a non-native/folk speaker's words **did not fire** — the exact
voice-erosion Understudy exists to prevent. This is the strongest evidence for the extractive design.

## Result 2 — SELECTION QUALITY IS THE REAL OPEN PROBLEM (the gate is blind to it)

Perfect fidelity ≠ good letter. Same input, both 100% faithful, but the models **selected different
things** — and the gate cannot see omission:
- Gemma-3-27b dropped `"small thing maybe, no need doctor, but my son he book already"` (minimization +
  who-initiated context) and `"in my country they do test, here only talking"` (distrust signal). Qwen kept both.
- Gemma-3-27b consistently selected FEWER spans than Qwen.

"Did it pick the RIGHT spans, with meaning-preserving boundaries?" is unmeasured here. This is where the
reframed eval belongs (not the old adversarial "extract the secret" scorer).

## Caveats (do not over-read)

1. **Wrong model.** All three are big (27–70B). The DEMO model is `gemma4:e2b` (~2.3B eff) — smaller
   models paraphrase more, and Gemma-3-27b *already* drops more aggressively, which a smaller Gemma may
   amplify. The number that matters (e2b fidelity + selection) is **untested** — blocked on Ollama relaunch.
2. **Fidelity, not boundaries.** The gate guarantees a quote is verbatim, NOT that its edges preserve
   meaning (negation left *outside* a span is not caught). See PIPELINE_ARCHITECTURE.md commitment 8 note.
3. **Small N, author-written input.** 121 spans total, rambles hand-written for this probe.

## So what
- The extractive+gate spine is **validated for fidelity/voice** on capable models → safe to build on.
- Next real question is **selection quality**, and it needs the reframed (willing-but-struggling) eval,
  not the old scorer. Rerun both harnesses on `gemma4:e2b` once Ollama is back for the real demo-model number.

## Result 3 — "TRY BOTH" (Alex, 07-25): Arm A (recall+structure) vs Arm B (model curates)

`compare.py`, same 9 rambles, same model (Qwen3-30B, arm-vs-arm control), concerns hand-labeled.

| | Arm A (recall+structure) | Arm B (model curates) |
|---|---|---|
| Recall vs hand-labeled concerns | 33/33 (100%) | 33/33 (100%) |
| Fabrication | impossible — 0 gate flags, all verbatim | **COMMITTED IT** (see below) |
| Voice (dialect) | exact | **standardized** ("is problem with leg"→"My right leg has been swelling") |
| Gate can police it? | YES | **NO** (abstractive prose — nothing to verify) |
| Readability | choppy fragments (fixable via bracketed-glue, commitment 7) | strong prose (intrinsic) |
| Avg length | 81 words | 71 words |

**Recall tie is a CEILING ARTIFACT, not equivalence:** capable model + same-model lenient recall-judge +
prominent concerns → everything conveys. Real recall differentiation needs a WEAK model (e2b) where B's
curation drops and A's over-surface does not. Ollama-blocked.

**The decisive divergence is qualitative — Arm B's two cardinal sins, verified in the output:**
- **Fabrication:** `formal_head` B invented `"No other symptoms—no vision changes, headaches, or fever"`
  (patient said none); `mood` B invented `"It's not depression like I've seen on TV."` Fake history a
  doctor reads as real. The ~44%-invented-content failure, live — and **the gate structurally cannot
  catch it** because B is abstractive.
- **Voice erasure:** B standardized the dialect the extraction had preserved.

**Refined bet (not settled):** A's weakness is **cosmetic + fixable** (choppy → add the already-designed
bracketed glue); B's weakness is **safety-critical + unfixable + unguardable** (fabrication is intrinsic
to abstraction). Tilts toward **A-as-spine + a glue layer**. To settle: (1) e2b rerun for the real recall
gap; (2) a stricter, diverse-model recall judge + a fabrication-detection judge on B; (3) build A's glue
layer and re-compare readability. Cost so far: $0.003.

## Result 4 — GLUE LAYER stress test (`glue_compare.py`) — the naive fix FAILS, the guard CATCHES it

Fabrication judged by a DIFFERENT family (Llama-3.3-70B, not the Qwen generator) → removes the
same-model-judge concern and confirms the eyeball claim:

| | Arm A2 (model writes glued letter) | Arm B (model curates) |
|---|---|---|
| Fabrications (Llama-70B judge, 9 letters) | **0** | **20** (incl. dangerous: invented "no vision changes/no fever" ROS; "thing on shoulder"→"small lump") |
| Bracket-discipline (deterministic, no LLM) | **FAILED 4/9** | n/a |

**Key failure:** letting the model WRITE the whole glued letter reintroduced rewording — 4/9 failed the
deterministic bracket-discipline check, and on `nonnative_leg` it **standardized the dialect back**
("is problem with leg" → "I have a problem with my leg"): Arm B's voice-erosion sin, sneaking in through
the glue step. The fabrication judge MISSED these (not "new clinical content"); the **deterministic
bracket-discipline check caught every one**. Lesson: **the model must NEVER re-emit the patient's words as
free prose** — that is where rewording hides.

## Result 5 — THE FIX (`a3.py`): deterministic assembly + bracketed glue → safe by construction

Model does ONLY select (verbatim) + tag-tier (indices, never touches words); **CODE assembles**. Spans stay
verbatim + unbracketed; every structural/connective word code adds is bracketed. Result:
**9/9 letters PROVABLY verbatim + fabrication-impossible** via bracket-discipline — *no LLM in the
safety-critical path.* Voice preserved exactly ("Is problem with leg. Two months now." / "Too much hot in
the body, that is why."). Cost: $0.001.

**The architectural principle, now proven:** *the model selects + tags; code assembles; the model never
re-emits the patient's words.* This buys fabrication-impossibility with a deterministic (no-LLM) proof —
the thing Arm B can never have.

**The honest tradeoff A3 surfaces (Alex's call):** you CANNOT smooth "is problem with leg" into standard
grammar without rewording = voice erosion. For the accessibility population, **non-standard grammar IS the
voice, not a defect** — A3 reads as clean but staccato honest testimony. Whether that's "readable enough"
vs. a doctor's expectation is a values call, but smoothing it is the exact harm we're avoiding. (Minor
cosmetic bug: a span ending in a comma yields ",." — trivial strip-fix, not logged as important.)

## Result 6 — SWEEP over 50 real synthetic conversations × 4 cheap models (2026-07-25)

`batch_extract.py` (labeled dialogue as context; extract + gate-verify against PATIENT WORDS ONLY,
so agent text can't contaminate) → signals → `build_explorer.py` → `results/explorer.html` (offline).
Independent recall judge = Llama-70B. Cost $0.37, 18/200 cells errored (JSON parse, spread evenly).

| model | fidelity | finding recall | key |
|---|---|---|---|
| gemma-3-27b | 97% | **80%** | 76% |
| gpt-oss-120b | 97% | **83%** | 74% |
| Qwen3-30B | 98% | 74% | 70% |
| Qwen3-32B | 96% | 75% | 70% |

- **Fidelity 96–98% across all four** on messy/evasive/dialect input — verbatim extraction is robust. Clean win.
- **Recall 74–83% is a PESSIMISTIC FLOOR** — these are the OLD adversarial transcripts where the patient
  *hid* the finding, so many "misses" = patient never disclosed it (correct omission), not extractor failure.
  Real recall for a willing patient is higher. gemma-27b + gpt-oss led. **Metric conflates "not disclosed"
  vs "missed"** → disambiguate by eyeballing the explorer (is the finding actually in patient_source?).
- 6/18 must-escalate cells missed the finding (small N, same caveat) — watch, don't panic.
- Nebius has NO sub-27B chat models (floor = gemma-3-27b); true-small (e2b) test still needs local Ollama.
- Caveat: gemma-3-27b authored the patient speech AND is an extractor (in-family; tagged in the explorer).

## Still open (unchanged by this)
- **Recall** — A3 fixes fidelity/fabrication, NOT omission. Recall (did selection surface every concern?)
  is the remaining axis, needs the reframed eval + **e2b** (Ollama-blocked).
- **Tier-ranking quality** — the model's primary/secondary/aside call can be wrong (cosmetic, human-fixable).
- All on Qwen3-30B; **e2b (demo model) untested.**
