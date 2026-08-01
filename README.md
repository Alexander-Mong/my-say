# My Say

*"Nothing about me without me."*

**On-device visit-prep for people who find doctor visits hard.** Built for the Kaggle "Build with
Gemma" hackathon (On-Device Private Health track) on **Gemma 4 (e2b)** via **Ollama**.

---

## Run it yourself (3 commands)

This is the whole point: **it runs on your machine, offline.** No account, no key, no cloud.

```bash
ollama pull gemma4:e2b        # ~7GB, one time
git clone https://github.com/Alexander-Mong/my-say && cd my-say
python understudy_app.py      # then open http://localhost:8077
```

Python 3.10+ and [Ollama](https://ollama.com). **No `pip install` for the core path** — the server is
Python standard library only. First launch pre-warms the model (~78s on a laptop CPU); after that a
reply takes a few seconds and a sheet about 8.

Once it is running, **turn your wifi off.** Everything still works. That is not a feature of the
demo, it is the product.

*Optional:* `pip install faster-whisper` adds voice input (a mic button appears; without it, nothing
changes). Set `PORT=8081` to run alongside something else.

---

## What it is

Some people find it hard to say the thing that actually matters in a doctor's visit — because
they're neurodivergent, socially anxious, overwhelmed, or just bad at talking under pressure with a
clock running. My Say is a private conversation partner that draws the hard-to-say thing out gently,
one question at a time, and then turns what the person *actually said* into a one-page visit sheet
they can hand to their doctor: "Notes for my appointment."

The sheet is not a summary the model wrote about the person. It is built from their own words. Every
line on the sheet carries a receipt — click it, and the exact source sentence highlights in the
transcript. The patient controls everything downstream of that: delete a line and it disappears
everywhere it would have shown up; undo brings it back; a "still yours, just not shared" drawer holds
anything set aside without discarding it. A small, clearly-marked, non-deletable block of sourced
federal health text (NHLBI / MedlinePlus / NIDDK, quoted verbatim) can appear alongside the patient's
own words when a safety criterion is met — it is visually distinct because it is not the patient's
text, and it was not chosen by looking at their specific case.

The whole thing runs on a laptop, offline, in airplane mode, on a small local model (Gemma 4 e2b)
served by Ollama. There is also an optional cloud backend (a larger hosted model) selectable from a
dropdown in the UI, framed explicitly as a privacy trade-off rather than a silent fallback: local
stays on the device, cloud is a labeled, opt-in choice that trades privacy for speed/quality. Nothing
about which backend is "better" is hidden — the point of the toggle is that the choice is visible and
the person makes it.

## The core invariant

**The model cannot invent patient wording in the sheet.**

Mechanically: the conversation transcript is segmented into numbered candidate clauses (deterministic
code, not the model). The model's only job at assembly time is to return a JSON list of which clause
IDs to keep and how to tier them — `{"ids": [...]}` — never free text. The displayed sentence and its
character offsets both come from the original segmentation of the patient's own transcript, so a line
that was never in the transcript cannot appear on the sheet.

The guarantee is *structural*, not a filter: because the candidate list is produced by code and the
model returns only bounds-checked IDs, there is no path by which model-authored text can reach the
sheet. On top of that, every kept line is independently re-checked against the transcript at the
claimed offsets by a second, separate check (`span_check.verify_span`); a line that fails is carried
with a `verified: false` flag rather than silently trusted. That check is a redundant belt, not the
thing holding the guarantee up.

This is a narrower claim than "the system cannot make mistakes." Selection can be wrong (it can pick
the wrong sentence to represent a concern, or drop something that mattered). Tagging can be wrong (a
line filed under the wrong tier). Omission is possible. What is not possible, by construction, is the
model putting words in the patient's mouth that they did not say — because assembly is deterministic
code operating over an ID list, not free-form generation.

**The one bounded exception, stated plainly.** The optional "say it better" feature (see Evidence
summary) does produce model-drafted wording — and it is the only thing on the sheet that ever does.
It is off unless the patient asks for it, it is offered per line, and nothing appears on the sheet
until that patient approves that line. An approved draft is labeled on the sheet ("said with help —
approved by me"), introduction of new clinical vocabulary is suppressed, and the receipt still opens
the patient's verbatim original underneath. Everything else on the sheet is the patient's own text,
unmodified.

## Architecture sketch

```
patient <-> conversational model (Gemma 4, warm, one question at a time)
                    |
                    v
        transcript (patient turns only)
                    |
                    v
     deterministic segmentation into numbered candidate clauses
                    |
                    v
   model call #2: SELECT + TIER only  ->  returns {"ids": [...]}   <- never free text
                    |
                    v
   deterministic assembly (code) -----> per-line span re-verification
                    |
                    v
   patient-controlled sheet: delete / undo / add-back / set-aside,
   click-to-source highlighting, non-deletable sourced safety block
                    |
                    v
              print view / handoff
```

Key files:

- `understudy_app.py` — the whole running app: stdlib-only HTTP server, chat UI, backend switcher
  (local Ollama models vs. an optional labeled cloud backend), and the sheet-editing endpoints.
- `gate/pipeline.py` — **the join.** One function, `build_sheet`, that turns a transcript + a chat
  function into structured sheet data (not a formatted string), so the UI can show receipts and
  support editing. This is the file that encodes the guarantee above.
- `gate/safety_net.py` — the sourced, non-deletable federal safety-information block: deterministic,
  criteria-gated (detect-then-decide, not the model deciding to escalate on its own), never chosen by
  looking at the patient's specific case.
- `gate/span_check.py` — independent re-verification that every line the pipeline kept really is
  present in the source transcript at the offsets claimed.
- `gate/test_span_check.py` — unit tests for the span-verification logic.
- `gate/polish.py` — the optional, per-line, patient-approved "say it better" draft step described
  under The core invariant.
- `gate/FINDINGS.md`, `gate/results/` — working notes and raw output from the validation runs
  described below. These are raw, uncurated generator and evaluator output, kept as-is rather than
  tidied, so the record matches what actually ran.
- `diagrams/` — architecture diagrams (C4 levels 1–3, plus one on why the invariant holds), as both
  `.svg` and `.mmd`.

## How to run

Requirements: Python 3 standard library only (no `pip install`), plus [Ollama](https://ollama.com)
running locally with a Gemma 4 model pulled.

```bash
ollama pull gemma4:e2b
python understudy_app.py
```

Then open the `http://localhost:...` URL the script prints. `PORT` / `UNDERSTUDY_PORT` and
`UNDERSTUDY_BACKEND` environment variables can override the defaults (port 8077, backend `e2b`). The
optional cloud backend is off by default and requires its own credentials, supplied via a local
`.env` file (not part of this repo) or the `NEBIUS_ENV_PATH` environment variable — the local,
on-device path needs none of that.

## Evidence summary

Numbers below are from the local, quantized Gemma 4 e2b model unless noted, and are kept in this repo
for authenticity/traceability, not as a substitute for reading the code:

- **180-cell validation gate:** the deterministic segmentation + selection path was run across 180
  test cells against Gemma 4 e2b with **0 gate failures** and **0.00 restatement** (the model never
  reproduced patient text itself instead of returning IDs), at **55.9% coverage** on realistic
  synthetic transcripts. *Run log not included in this repo;* the figures are recorded against the
  code they describe as a standing assertion in `gate/pipeline.py` (see the block comment near
  line 34), which also states the condition under which they stop being true.
- **Rewrite test:** an 80-cell test across 4 model families, checking whether a "clean up my wording"
  rewrite step could be layered on without the model drifting into inventing or escalating clinical
  vocabulary, found **zero vocabulary escalations**. That result is the evidence base for the
  shipped, opt-in, per-line **"say it better"** feature (`gate/polish.py`, the `/polish` endpoint,
  and the sheet UI): the patient approves it line by line, and an approved draft is labeled as such
  with the verbatim original still one click away.

A separate 200-cell sweep across four cloud models (`gate/results/batch_run.log`, `results.json`,
`explorer.html`) is a **different, earlier run** from the 180-cell e2b gate above — it measured
fidelity and recall on cloud models, errored on 18 of its 200 cells, and none of the numbers in this
section come from it. It is included as a decision trail, not as evidence for the shipped path.

These numbers describe the pipeline mechanism (segmentation, selection, verification), not a clinical
outcome study. See Honest limits below.

## Honest limits

- **This is not a diagnostic or treatment tool.** It is decision-support for the conversation itself:
  documentation, navigation, and communication help, nothing more.
- **The synthesis is novel and untested end-to-end.** The individual pieces (segmentation, ID-only
  selection, independent span verification, sourced safety text) are each validated in isolation as
  above. The product built by combining them into a real conversational flow has not been tested on
  real patients or real clinical encounters.
- **Selection and tagging can still be wrong.** The invariant is about *wording*, not about *which*
  wording gets chosen or how it's tiered. A patient should always review the sheet before handing it
  over — the UI is built around that review (delete/undo/add-back/highlight) precisely because it is
  needed, not decorative.
- **This was built fast.** Assembled and validated over a few days of prep plus on-site hackathon
  time — a transparent, honest provenance, not a polished multi-month product.
- **The e2b model is small.** It is fast enough to run on a consumer laptop CPU offline, which is the
  point, but it is not a large frontier model; its conversational quality reflects that trade-off.

## Synthetic-data note

No real patient data, real patient transcripts, or real personal health information appears anywhere
in this repository — in the code, the validation artifacts, the evidence folder, or any example
text. All example transcripts and test cells used during development and validation are synthetic,
written for testing purposes only.
