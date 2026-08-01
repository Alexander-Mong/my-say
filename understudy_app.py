#!/usr/bin/env python3
"""
Understudy — live prototype (v0.2) for the Sunday role-play test + the hackathon demo.

A private health confidant: warmly draws out the hard-to-say thing, then assembles a sheet the
person can hand to their doctor -- built from their OWN words, with a receipt on every line.

The conversation is a model. The sheet is not: gate/pipeline.py sends the model numbered clauses
from the transcript and accepts only {"ids": [...]}, so code does the assembling and the model
never re-emits anything the person said. Warm on the way in, neutral on the way out.

Runs with ZERO pip installs (stdlib only). Three runtimes, switchable live in the UI dropdown —
this toggle IS the demo ("show the difference" between private-on-device and fast-cloud):
  - e2b    : local gemma4:e2b via Ollama  (DEFAULT; the on-device demo model — fast enough on CPU)
  - e4b    : local gemma4:e4b via Ollama  (bigger/better; fine on a GPU, slow on a laptop CPU)
  - nebius : cloud fallback               (fast, but the words LEAVE the device — the contrast)

Gemma 4 is a REASONING model: it "thinks" before answering. Thinking is the dominant latency cost
on CPU, so we run chat with thinking OFF (snappy) and expose a "deep reasoning" toggle for the
one-time hand-off (better clinical extraction, worth the wait on a fast machine).

Run:  python understudy_app.py   then open the http://localhost URL it prints.
"""
import io, json, os, sys, time, threading, urllib.request, http.server, socketserver
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "gate"))
import pipeline                     # the verified path: model selects IDs, CODE assembles
import polish                       # "say it better": tested drafter prompt + deterministic gates
import paste_parser                 # paste-in: deterministic (no model) turn/topic splitting
import probe                        # "ask me more about this": clinical-element question bank, zero model calls
import endround                     # end-of-conversation deep-reasoning round: questions + dropped threads
import speech                       # "read my sheet aloud": Windows SAPI via PowerShell, no model
import safety_net                   # deterministic self-harm net -- ONE pattern list shared with the sheet path

# Honour a PORT from the environment so a second instance can run alongside one already serving
# 8077 (an earlier build was still up when this was wired). 8077 stays the default.
PORT = int(os.environ.get("PORT") or os.environ.get("UNDERSTUDY_PORT") or 8077)
DEFAULT_BACKEND = os.environ.get("UNDERSTUDY_BACKEND", "e2b")
# HOSTED demo mode (UNDERSTUDY_HOSTED=1): the cloud-preview surface (QR link / subdomain).
# Forces the Nebius backend (there is no Ollama on the server, and "on-device" would be a false
# claim on a hosted page), shows the honesty + demo-only banners, and trims the runtime menu.
# The airplane-mode laptop demo runs this same file with the flag unset — one source of truth.
HOSTED = os.environ.get("UNDERSTUDY_HOSTED") == "1"
# Which backend the hosted surface serves. "nebius" = the original cloud fallback; on the
# Daytona deploy this is set to e2b/e4b so the hosted preview runs REAL Gemma via the
# sandbox's own Ollama (all-Gemma stack, Alex ruling 07-31 03:59 — Nebius = emergency only).
HOSTED_BACKEND = os.environ.get("UNDERSTUDY_HOSTED_BACKEND", "nebius")
if HOSTED:
    DEFAULT_BACKEND = HOSTED_BACKEND
# FEEDBACK build (UNDERSTUDY_FEEDBACK=1): adds an explicit, opt-in "share your thoughts" panel
# for the hackathon tinkering phase. NOTHING is logged ambiently — a conversation is stored only
# when the tester presses Share, which is the product's own thesis (patient-controlled
# disclosure) applied to feedback. Not part of the demo or the product claims; default OFF.
FEEDBACK = os.environ.get("UNDERSTUDY_FEEDBACK") == "1"
# Bind localhost by default. Set UNDERSTUDY_HOST=0.0.0.0 for phone-on-LAN or server deploys.
HOST = os.environ.get("UNDERSTUDY_HOST", "127.0.0.1")

# The app name is PENCILED (family consult in progress; may change day-of). Everything that
# shows or says the name derives from these two constants — a rename is a one-line edit.
APP_NAME = os.environ.get("UNDERSTUDY_NAME", "My Say")
TAGLINE = os.environ.get("UNDERSTUDY_TAGLINE", "nothing about me without me")

OLLAMA_CHAT = "http://localhost:11434/api/chat"
NEBIUS_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
NEBIUS_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # fast+stable cloud fallback: ~0.9s chat, ~5s handoff
# (measured 2026-07-24: Llama-3.3-70B was highly variable — median ~18s, spiked to 84s — bad for a live demo)
NEBIUS_ENV = os.environ.get("NEBIUS_ENV", ".env")

# The runtime menu. kind=local → Ollama on this device; kind=cloud → leaves the device.
BACKENDS = {
    "e2b":    {"kind": "local", "model": "gemma4:e2b", "label": "Gemma 4 E2B — on-device"},
    "e4b":    {"kind": "local", "model": "gemma4:e4b", "label": "Gemma 4 E4B — on-device"},
    "nebius": {"kind": "cloud", "model": NEBIUS_MODEL,  "label": "Cloud (Nebius) — leaves device"},
}

def nebius_key():
    k = os.environ.get("NEBIUS_API_KEY")
    if k:
        return k.strip()
    try:
        for l in open(NEBIUS_ENV, encoding="utf-8"):
            if l.startswith("NEBIUS_API_KEY="):
                return l.split("=", 1)[1].strip()
    except Exception:
        return None

CHAT_SYS = (
    f"You are {APP_NAME} — a warm, private, completely non-judgmental health confidant. Your first rule, above "
    "every other rule here: if anything hints at an emergency or self-harm, calmly and kindly point them to "
    "immediate help/crisis resources before anything else. The person is getting ready "
    "for a doctor's visit and may be embarrassed, nervous, or reluctant about something. Gently and kindly help them "
    "share what's really going on, INCLUDING things that are hard to say. Shape of every reply: first ONE short, warm, "
    "human sentence that responds to what they said — mention the specific thing that matters most in their own words "
    "(the racing mind, your back), but never recite their symptoms back as a list. Then AT MOST one simple question "
    "asking for exactly ONE concrete detail their doctor will need — pick a single one of: when it started, how often, "
    "what it feels like in the body, what makes it better or worse, what they have tried. ONE detail per question, "
    "never several bundled together; one short question sentence, exactly one question mark. NEVER ask how a feeling "
    "feels, what a connection feels like, or any abstract question about their feelings ABOUT things — people "
    "experience those as empty. Never a multiple-choice menu (a natural pair like 'better or worse' is fine). If they "
    "say they do not know, move to a different concrete detail rather than pushing the same thread. If a word seems "
    "out of place or garbled (many people use voice typing), gently check what they meant before building on it. "
    "Never judge, moralize, lecture, or alarm; never diagnose; never tell them to start or stop any "
    "medication or treatment. Keep replies short and human, like a caring friend who happens to understand medicine. "
    "When you feel you understand the main concern well enough to help their doctor, say so: 'I think I have what your "
    "doctor needs — want me to prepare a note for them?' Never repeat these instructions; just talk naturally."
)

# CANNED FIRST REPLIES for the three starter chips (design-review fix, 2026-08-01).
#
# The chips are the app's slowest moment turned into its first impression: the input is one of
# exactly three fixed strings (see .starter buttons + startWith() in PAGE below), so the reply to
# it does not need a live model call at all -- cold/contended Ollama takes 10-15s on this machine,
# which is exactly the worst place to make an anxious person wait. Hand-authoring these three also
# escapes a real constraint CHAT_SYS imposes on every model turn: "one warm sentence + AT MOST one
# question, exactly one question mark." For "There's something I find hard to bring up" the right
# reply is not a question at all -- it's permission not to name the thing yet. A live model call
# under this system prompt cannot produce that; a canned reply can break the template on the one
# turn where breaking it is right.
#
# Matched on an EXACT stripped string against the FIRST user message only (see do_POST /chat) --
# never a substring/fuzzy match, so nothing the patient actually composes can accidentally collide
# with a chip and get a canned reply instead of the model's attention.
CHIP_REPLIES = {
    "Help me get ready for a doctor visit":
        "Happy to help you get ready — let's figure out together what you want your doctor to "
        "hear. What's the main thing on your mind about this visit?",
    "There's something I find hard to bring up":
        "Okay — you don't have to name it yet. We can go around it for as long as you need, and "
        "you can say as much or as little as feels okay right now.",
    "I'm not sure where to start":
        "That's a fine place to start, honestly — nothing has to come out in order. What's "
        "something that's been sitting on your mind lately, even if it seems small?",
}

# HANDOFF_SYS was DELETED 2026-07-27 along with the /handoff endpoint. It asked the model to
# "translate the patient's vague or embarrassed words into precise, respectful CLINICAL language"
# and to emit a doctor_note plus per-finding "clinical" rewrites. That is the old architecture: the
# model authoring the output, with a confidence rating standing in for provenance.
#
# It is recorded here as a comment rather than silently removed because the file is going in a
# public repo, and a reader who found that prompt still sitting in the source would reasonably
# conclude the product rewrites patients' words. It does not. Selection now happens in
# gate/pipeline.py, which sends numbered candidate clauses and accepts only {"ids": [...]} back --
# so the sheet is assembled by code from the patient's own transcript and the model never emits
# their words at all.


def call_model(messages, backend, force_json=False, think=False):
    """Returns {content, ms, where} or {error}. `think` only applies to local Gemma."""
    cfg = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])
    t0 = time.time()
    if cfg["kind"] == "cloud":
        key = nebius_key()
        if not key:
            return {"error": "no Nebius key found — check the .env path"}
        body = json.dumps({"model": cfg["model"], "messages": messages,
                           "temperature": 0 if force_json else 0.5, "max_tokens": 900}).encode()
        req = urllib.request.Request(NEBIUS_URL, data=body,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return {"content": d["choices"][0]["message"]["content"] or "",
                "ms": int((time.time() - t0) * 1000), "where": "cloud"}
    # local Gemma via Ollama (offline). think=False keeps CPU latency low.
    body = {"model": cfg["model"], "messages": messages, "stream": False, "think": think,
            "keep_alive": "8h", "options": {"temperature": 0 if force_json else 0.5}}
    if force_json:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA_CHAT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return {"content": d.get("message", {}).get("content") or "",
            "ms": int((time.time() - t0) * 1000), "where": "device"}


def call_model_stream(messages, backend, think=False):
    """Streaming counterpart to call_model, used only by /chat. Yields small event dicts as they
    become available:
      {"content": "..."}                       -- one piece of assistant text, in order
      {"done": True, "ms": int, "where": str}   -- terminal event on success (no more content follows)
      {"error": "..."}                          -- terminal event on failure (no more events follow)
    Ollama's own streaming wire format is newline-delimited JSON objects, one per line, each
    carrying {"message": {"content": "<piece>"}, "done": bool, ...} -- urllib's file-like response
    object iterates by line, so this is a small translation, not a new protocol.

    Cloud (Nebius) is NOT actually streamed here -- that would mean parsing a second wire format
    (OpenAI-style `data: {...}` SSE framing) for a backend this build never measured against a real
    key/network path in the time available. It still goes through this same generator so /chat has
    one code path regardless of backend: the cloud call blocks exactly as before, then its full
    reply is yielded as a single "content" event followed by "done" -- so the client's streaming
    renderer degrades to "whole reply appears at once," never breaks.
    """
    cfg = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])
    t0 = time.time()
    if cfg["kind"] != "local":
        try:
            out = call_model(messages, backend, think=think)
        except Exception as e:
            yield {"error": str(e)}
            return
        if "error" in out:
            yield out
            return
        if out["content"]:
            yield {"content": out["content"]}
        yield {"done": True, "ms": out["ms"], "where": out["where"]}
        return
    body = {"model": cfg["model"], "messages": messages, "stream": True, "think": think,
            "keep_alive": "8h", "options": {"temperature": 0.5}}
    req = urllib.request.Request(OLLAMA_CHAT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw_line in r:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (d.get("message") or {}).get("content") or ""
                if piece:
                    yield {"content": piece}
                if d.get("done"):
                    yield {"done": True, "ms": int((time.time() - t0) * 1000), "where": "device"}
                    return
    except Exception as e:
        # Covers connection errors, timeouts, and a mid-stream drop from Ollama alike -- the
        # caller (H._handle_chat) turns this into one more ndjson line, never a raw traceback
        # reaching the browser.
        yield {"error": str(e)}
        return
    # The loop above returns from inside the `with` on the normal path (a line with done=true).
    # Reaching here means the connection ended WITHOUT one -- Ollama's process died, or something
    # reset the socket mid-generation. urllib's response iterator treats a closed connection as a
    # quiet end-of-iteration, not an exception, so without this guard a mid-stream drop would fall
    # through and get reported as a successful, silently-truncated reply. Caught by
    # test_stream_error.py (a dropped connection after 3 real content chunks) during the build.
    yield {"error": "the connection ended before the reply finished"}


def prewarm(backend=DEFAULT_BACKEND):
    """Load the default local model so the first real turn isn't a 78s cold-load."""
    cfg = BACKENDS.get(backend)
    if not cfg or cfg["kind"] != "local":
        return
    try:
        print(f"[understudy] pre-warming {cfg['model']} (avoids cold-load in the demo)...", flush=True)
        t0 = time.time()
        call_model([{"role": "user", "content": "hi"}], backend, think=False)
        print(f"[understudy] {cfg['model']} resident in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[understudy] pre-warm skipped: {e}", flush=True)


# --- Voice input (merged from the voice_mode spike, Alex ruling 07-31 04:25: "definitely
# merge the voice into the demo"). Local STT via faster-whisper: browser mic -> POST audio
# bytes -> transcript lands in the composer as an EDITABLE DRAFT (never auto-sent; Send stays
# the only confirm gate). Nothing is archived: audio is transcribed in memory and discarded.
# GRACEFUL DEGRADE: if faster-whisper is not installed (family laptop default, IONOS box),
# the mic button simply does not render and /stt declines politely — zero other impact.
try:
    import faster_whisper  # noqa: F401 -- availability probe only; the model loads lazily below
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

# --- Voice output ("Read my sheet aloud", deferred item, ~30min estimate). Server-side SAPI via
# PowerShell/System.Speech (see gate/speech.py's module docstring for the full comparison against
# browser speechSynthesis and why server-side won: the app's one promise is "nothing leaves this
# device", and that promise is NOT reliably true of the browser API -- some of Chrome's own bundled
# voices call a Google endpoint to synthesize, verified this session by checking
# speechSynthesis.getVoices()[i].localService rather than assuming). GRACEFUL DEGRADE, same pattern
# as STT_AVAILABLE: the button simply does not render when this is False.
TTS_AVAILABLE = speech.available()
STT_MODEL_NAME = os.environ.get("UNDERSTUDY_STT_MODEL", "tiny.en")
STT_DEVICE = os.environ.get("UNDERSTUDY_STT_DEVICE", "cpu")
STT_COMPUTE = os.environ.get("UNDERSTUDY_STT_COMPUTE", "int8")
_stt_model = None
_stt_lock = threading.Lock()


def get_stt_model():
    """Lazy-load faster-whisper on first use; keep the loaded model resident afterward."""
    global _stt_model
    if _stt_model is None:
        with _stt_lock:
            if _stt_model is None:
                from faster_whisper import WhisperModel
                print(f"[understudy] loading STT model '{STT_MODEL_NAME}' "
                      f"({STT_DEVICE}/{STT_COMPUTE}) ...", flush=True)
                t0 = time.time()
                _stt_model = WhisperModel(STT_MODEL_NAME, device=STT_DEVICE, compute_type=STT_COMPUTE)
                print(f"[understudy] STT ready in {time.time() - t0:.1f}s", flush=True)
    return _stt_model


PAGE = """<!doctype html><html data-theme=blue><head><meta charset=utf-8><title>__APP_NAME__</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20160%20160'%3E%3Cg%20fill='%232f6f8a'%3E%3Cpath%20d='M56,50%20C46,50%2040,58%2040,68%20C40,78%2047,85%2056,85%20C56,96%2048,104%2038,107%20L38,116%20C55,113%2068,101%2068,79%20C68,62%2063,50%2056,50%20Z'/%3E%3Cpath%20d='M104,110%20C114,110%20120,102%20120,92%20C120,82%20113,75%20104,75%20C104,64%20112,56%20122,53%20L122,44%20C105,47%2092,59%2092,81%20C92,98%2097,110%20104,110%20Z'/%3E%3C/g%3E%3C/svg%3E">
<style>
 /* MERGED HYBRID design system (round 2): C_quiet_editorial's foundation (the sheet is a document,
    not a card in an app -- so it gets the elevation and the typography; everything around it is
    quieter than it is; hairline rules replace most card borders; the serif is reserved for the
    patient's own words and carries real display weight; paper is layered warm-under-cool so the
    sheet reads as a page lifted off a desk) + A_precision_calm's interaction layer (a real
    "thinking" state, graceful error cards, staggered sheet-reveal, smooth-scroll receipts, a
    retriggered mobile tab fade) restyled into this palette/rule/serif language rather than A's
    own cool-paper + shadow-language look. Voice input (mic button/canvas/status), the busy-guard
    (oops()), the concreteness system prompt, and hosted-backend config are main-line features that
    postdate both forks -- carried forward and restyled here, not reintroduced. */
 /* THEME SYSTEM (v3): every color in this stylesheet flows from :root custom properties. A theme
    is just a named override block, `:root[data-theme=NAME]{...}`, selected by one attribute on
    <html> that the appearance picker's one-line JS swaps -- no localStorage, nothing persisted,
    default (Calm blue) on every load, exactly on-message with "nothing is stored." Plain :root
    below carries the CALM BLUE values directly (so the very first paint is already correct, no
    flash before JS runs); `[data-theme=warm]` restores the v2-approved palette byte-for-byte;
    `[data-theme=dark]` is the low-light slate variant.
    v3.1 CORRECTION (Alex, 13:21): the v2/v3 salience system used a HOT wayfinding family --
    terracotta rails, saturated-yellow highlighter receipts. On a nervous person's own words that
    reads as ALARMING, not comforting -- red/orange/correction-yellow signal "flagged" or "graded,"
    the opposite of this app's whole promise. New rule: HONORING, NOT FLAGGING. Emphasis on
    anything patient-authored has to feel like their words being treasured, never marked. So:
      - --tier-primary/--tier-secondary/--tier-context are now a CALM-TRUST family -- indigo, sage,
        soft stone -- distinct from each other and from --accent's own teal-blue, varying in
        STRENGTH by depth/saturation (primary is the deepest/most saturated) rather than by hue-heat.
      - --mark-bg/--mark-ink (the highlighter receipt mark) moved from a hot saturated yellow to a
        soft parchment/cream glow with a firmer ink-colored underline for definition -- "illuminated,"
        like a passage someone chose to underline in their own book, not "marked," like a teacher's
        red pen. Chosen over a blue wash specifically because a blue highlight would blend into the
        Calm Blue chrome and read as "more app," not "your words, treasured" -- the warm parchment
        family stays legible as its own thing against all three chromes.
      - The red/orange family is NOT banned outright -- it stays for the two places that are
        conventional and non-alarming precisely because they are not about the patient's words: the
        mic recording indicator (--danger-quiet/--danger-rgb -- universal "recording" red) and the
        destructive-hover states (--danger-text, on Start-over/Remove-from-sheet -- a normal warning
        affordance for a delete action, not a judgment on what was said).
      - The pinned card dropped its colored background wash entirely -- salience there now comes
        from TYPOGRAPHY + SPACE + ELEVATION (a bigger roman-weight serif line, more padding, a
        stronger soft shadow lifting it off the sheet, and a large low-opacity hanging quotation mark
        echoing the who-glyph monogram language) rather than a tinted fill. See `.pinned` below.
    Two token families never move between themes -- this is the salience system, and it is frozen:
    --mark-bg/--mark-ink/--mark-bg-faint (the highlighter receipt family) and --tier-primary/
    --tier-secondary/--tier-context (+ their tints), plus --pulse-warm/--cloud/--danger-quiet/
    --danger-rgb. These stay constant across chrome so the wayfinding rails and the receipt marks
    read as landmarks against ANY background, cold or warm, light or dark. The one exception the
    dark theme needs is --mark-bg-faint and the tier *tints*: those are large background washes with
    theme-following ink text sitting on top of them (the source panel, the pinned card used to be),
    so they still have to darken with the stage -- only the small self-contained chip/glyph/rule
    colors stay identical everywhere.
    Two roles were split out of tokens that used to double-book a "surface" and a "text-on-a-filled-
    control" meaning at once -- fine when paper/ink are both light (warm, blue) but broken once a
    dark theme makes the surface dark while a button's label still needs to read light-on-accent:
      --on-fill     : label/icon color for anything sitting on a solid --accent or --danger-quiet
                      fill (was hardcoded to --paper-sheet, which only worked because paper-sheet
                      happened to be light in every theme built so far).
      --accent-hover : the darker fill a filled button turns on hover (was --accent-ink, which also
                      had to double as the TEXT color used everywhere else -- a dark theme wants
                      that text role light-on-dark and the hover-fill role staying a filled color,
                      so hover gets its own token).
      --danger-text  : caution/warning TEXT color (draft cautions, destructive-hover states) --
                      split from --danger-quiet, which keeps its original job as a fill/rule color.
    --shadow-rgb is the r,g,b triple the elevation box-shadows key off of (rgba(var(--shadow-rgb),
    alpha) instead of a literal rgba(35,32,25,...)) -- in the dark theme this is a pale warm triple
    instead of a dark one, so the raised sheet/pinned card get a soft glow off the dark stage
    instead of an invisible dark-on-dark shadow ("paper cards glow"), while light themes keep the
    original ink-tinted drop shadow. --danger-rgb is the equivalent triple for the mic-recording
    pulse ring; it is one of the constants above and never changes with theme. */
 :root{
  /* ---- Calm blue (DEFAULT) ---- */
  --paper:#e7ecf1; --paper-recessed:#dde4ea; --paper-sheet:#f9fbfc; --paper-sheet-2:#eef2f6;
  --ink:#1c2b3a; --ink-soft:#4c5c6c; --ink-faint:#8592a0;
  --rule:#d2dbe3; --rule-strong:#b7c4d0;
  --accent:#2f6f8a; --accent-ink:#1c4c5e; --accent-tint:#e1eef1; --accent-tint-2:#c7dee3;
  --on-fill:#f9fbfc; --accent-hover:#1c4c5e; --danger-text:#7c433c; --shadow-rgb:27,43,58;
  --focus:var(--accent);
  /* ---- constants: never redefined per theme (see note above) ---- */
  /* Receipt mark, softened (v3.1): a gentle parchment/cream glow, not a saturated highlighter --
     "illuminated," not "flagged." --mark-ink stays a warm sepia ink (it was never the alarming
     part -- ink-on-paper is calm; the hot part was the bright background it sat on). */
  --mark-bg:#f1e6cc; --mark-ink:#5c4a2e; --mark-bg-faint:#f8f2e2;
  --danger-quiet:#7c433c; --danger-rgb:124,67,60; --cloud:#6b4a8a;
  /* CALM-TRUST SALIENCE (v3.1): a wayfinding family for the sheet's three tiers -- indigo, sage,
     soft stone, each a distinct hue so the three tiers stay a scannable map, but all drawn from the
     same calm-trust family as the rest of the app (never red/orange/hot-yellow -- these are the
     patient's own words, and emphasis on them should read as care, not correction). Strength varies
     by DEPTH/SATURATION (primary is the deepest) rather than by hue-heat -- same "same family, lower
     volume" principle as before, just recolored. */
  --tier-primary:#4a5a8f; --tier-primary-tint:#e6e8f2;
  --tier-secondary:#6e7f5c; --tier-secondary-tint:#e9ede4;
  --tier-context:#8a8f8a; --tier-context-tint:#eaeaea;
  /* A warm pulse for "the app is working" states (thinking dots, the on-device status dot) --
     distinct from --accent's cool teal (which still means "this is the confidant/primary action"),
     so waiting reads as warm/alive rather than clinical. */
  --pulse-warm:#ba7239;
  --font-sans:'Segoe UI',Optima,Candara,Calibri,'Noto Sans',Arial,sans-serif;
  --font-serif:Constantia,'Iowan Old Style','Palatino Linotype',Cambria,Georgia,serif;
  --sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:20px; --sp-6:24px; --sp-7:32px;
  --r-card:3px; --r-line:2px; --r-sm:3px; --r-pill:5px;
  --ease:cubic-bezier(.2,.65,.25,1); --dur:180ms;
 }
 /* ---- Warm paper: the exact v2-approved palette, byte-identical, as the alternate theme ---- */
 :root[data-theme="warm"]{
  --paper:#eae7de; --paper-recessed:#e0ddd2; --paper-sheet:#fbfaf4; --paper-sheet-2:#f4f2e8;
  --ink:#232019; --ink-soft:#575145; --ink-faint:#948c7c;
  --rule:#dbd5c4; --rule-strong:#c3b9a3;
  --accent:#2e4a52; --accent-ink:#1a2f34; --accent-tint:#e7ece9; --accent-tint-2:#d3ddd6;
  --on-fill:#fbfaf4; --accent-hover:#1a2f34; --danger-text:#7c433c; --shadow-rgb:35,32,25;
 }
 /* ---- Quiet dark: low-light slate stage, paper cards glow (E2 deep-ink heritage). The tier
    colors and the pinned-card/source-panel washes are brightened/darkened as a set here (not left
    at their light-theme values) because real text sits directly on them -- see the long note above
    for why that is the one place the "salience never moves" rule needs a per-theme adaptation
    rather than a literal freeze. */
 :root[data-theme="dark"]{
  --paper:#191b20; --paper-recessed:#141519; --paper-sheet:#24272e; --paper-sheet-2:#2a2d34;
  --ink:#e9e6de; --ink-soft:#b7b2a4; --ink-faint:#89836f;
  --rule:#34373d; --rule-strong:#474b53;
  --accent:#2f7a92; --accent-ink:#7cc3d6; --accent-tint:#20343d; --accent-tint-2:#2a4650;
  --on-fill:#ffffff; --accent-hover:#1f5a6d; --danger-text:#e2988c; --shadow-rgb:233,230,222;
  --mark-bg-faint:#29241d;
  --tier-primary:#8f9adb; --tier-primary-tint:#262a3d;
  --tier-secondary:#a8bb8f; --tier-secondary-tint:#262a20;
  --tier-context:#b7bcc2; --tier-context-tint:#26282b;
 }
 *{box-sizing:border-box}
 html,body{margin:0;padding:0}
 body{background:var(--paper);color:var(--ink);font-family:var(--font-sans);font-size:14.5px;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
 button,input,select,textarea{font-family:var(--font-sans)}
 button:focus-visible,a:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible,
 textarea:focus-visible,[tabindex]:focus-visible{
  outline:none;box-shadow:0 0 0 2px var(--paper-sheet),0 0 0 4px var(--accent);border-radius:3px}
 ::selection{background:var(--mark-bg);color:var(--mark-ink)}
 *{scrollbar-width:thin;scrollbar-color:var(--rule-strong) transparent}
 ::-webkit-scrollbar{width:10px;height:10px}
 ::-webkit-scrollbar-track{background:transparent}
 ::-webkit-scrollbar-thumb{background:var(--rule-strong);border-radius:6px;border:2px solid var(--paper)}
 ::-webkit-scrollbar-thumb:hover{background:var(--ink-faint)}
 @keyframes riseIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
 @keyframes ruleDraw{from{transform:scaleX(0)}to{transform:scaleX(1)}}
 @keyframes unfold{from{opacity:0;max-height:0}to{opacity:1;max-height:150px}}
 @media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;
   transition-duration:.001ms!important;scroll-behavior:auto!important}
 }

 /* Masthead. A double hairline (the box-shadow is the second, fainter rule) reads like the rule
    under a newspaper nameplate; the wordmark takes the serif, the tagline sits under it in italic
    like a subtitle line under a title. */
 header.chrome{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:var(--sp-4);
  padding:var(--sp-5) var(--sp-7) var(--sp-4);background:var(--paper-sheet-2);
  border-bottom:1px solid var(--rule-strong);box-shadow:0 5px 0 -4px var(--rule)}
 .brand{display:flex;flex-direction:row;align-items:center;gap:10px}
 .brand-mark{width:30px;height:30px;flex:none;color:var(--accent)}
 .brand-text{display:flex;flex-direction:column;gap:1px}
 .wordmark{font-family:var(--font-serif);font-size:21px;font-weight:600;letter-spacing:-.005em;color:var(--ink)}
 .tagline{font-family:var(--font-serif);font-style:italic;font-size:12.5px;color:var(--ink-faint);letter-spacing:.005em}
 .ctrls{display:flex;align-items:center;flex-wrap:wrap;gap:var(--sp-5);font-size:12px;color:var(--ink-soft)}
 .ctrl-select{display:flex;align-items:center;gap:var(--sp-2);font-variant-caps:all-small-caps;letter-spacing:.02em}
 .ctrl-select select{font:inherit;font-size:12.5px;color:var(--ink-soft);background:transparent;
  border:none;border-bottom:1px dotted var(--rule-strong);padding:2px 2px 3px 4px;cursor:pointer}
 .ctrl-select select:hover{color:var(--ink);border-bottom-color:var(--ink-soft)}
 .ctrl-toggle{display:flex;align-items:center;gap:6px;cursor:pointer;font-variant-caps:all-small-caps;letter-spacing:.02em}
 .ctrl-toggle input{accent-color:var(--accent)}

 .tabbar{display:none}
 .wrap{display:flex;gap:var(--sp-6);padding:var(--sp-6) var(--sp-7) var(--sp-7);
  max-width:1180px;margin:0 auto;align-items:flex-start}
 .col{flex:1;min-width:0}

 /* Conversation is the recessed surface -- a sunken tray, not a bordered card -- so the sheet
    reads as the one thing on the page that is actually raised off the paper. */
 .chat{background:var(--paper-recessed);border:none;border-radius:var(--r-card);
  box-shadow:inset 0 1px 4px rgba(var(--shadow-rgb),.06);
  padding:var(--sp-4) var(--sp-4) var(--sp-2);height:60vh;overflow:auto}
 .msg{margin:0 0 var(--sp-4);max-width:88%;line-height:1.55;font-size:14.5px;
  animation:riseIn var(--dur) var(--ease) both}
 .msg p{margin:0;padding:var(--sp-3) var(--sp-4);border-radius:10px}
 .msg.you{margin-left:auto}
 .msg.you p{background:var(--paper-sheet);border:1px solid var(--rule);border-bottom-right-radius:3px}
 .msg.u p{background:transparent;border:1px solid transparent;padding-left:2px;color:var(--ink-soft);
  font-family:var(--font-serif);font-size:15.5px;line-height:1.6}
 /* Anchor moment: the confidant's identity is a small warm monogram glyph, not just a caps label --
    a fixed visual landmark the eye can find at the top of every reply instead of re-reading text
    each time to know who's speaking. */
 .msg.u .who{display:flex;align-items:center;gap:6px;font-variant-caps:all-small-caps;font-size:12px;
  letter-spacing:.08em;color:var(--ink-faint);margin:0 0 var(--sp-1) 2px;font-family:var(--font-sans)}
 .who-glyph{display:inline-flex;flex:none;align-items:center;justify-content:center;width:15px;height:15px;
  border-radius:50%;background:var(--accent-tint);color:var(--accent-ink);border:1px solid var(--accent-tint-2);
  font-family:var(--font-serif);font-size:10.5px;line-height:1}
 .msg mark,.sheet mark{background:var(--mark-bg);color:var(--mark-ink);border-bottom:2px solid var(--accent);
  border-radius:1px;padding:0 2px;font-weight:600;box-decoration-break:clone}

 /* Graceful error presentation (grafted from A_precision_calm, restyled): a quiet notice in the
    danger ink -- left rule + a small round marker, same marginalia family as .refusal/.draft-card
    below -- instead of a raw bracketed string dropped into a patient-style bubble. The real detail
    stays visible, just kept small underneath. */
 .msg.err p{display:flex;align-items:flex-start;gap:var(--sp-2);background:none;
  border:none;border-left:2px solid var(--danger-quiet);padding-left:var(--sp-3);color:var(--ink-soft);
  animation:riseIn var(--dur) var(--ease) both}
 /* display:inline-flex makes this self-contained regardless of the parent's own display -- it
    renders inside a flex parent (.msg.err p, .sheet-error) AND a plain block/inline parent
    (.refusal, which is not display:flex), where a bare inline span would ignore width/height/
    border-radius and collapse to text instead of a circle. */
 .err-ic{display:inline-flex;align-items:center;justify-content:center;vertical-align:middle;
  flex:none;width:15px;height:15px;border-radius:50%;background:var(--danger-quiet);
  color:var(--on-fill);font-size:10.5px;font-weight:700;font-style:normal}
 .err-detail{display:block;font-family:var(--font-sans);font-size:11.5px;color:var(--ink-faint);
  margin-top:2px}

 /* A real thinking state (grafted from A_precision_calm) -- three softly pulsing dots in the ink
    accent, replacing the static "..." ellipsis, used for every wait state. Frozen by the global
    prefers-reduced-motion rule above like every other animation here. */
 .thinking{display:inline-flex;align-items:center;gap:6px}
 .thinking .td{width:5px;height:5px;border-radius:50%;background:var(--pulse-warm);display:inline-block;
  animation:tdPulse 1.1s var(--ease) infinite}
 .thinking .td:nth-child(2){animation-delay:.15s}
 .thinking .td:nth-child(3){animation-delay:.3s}
 @keyframes tdPulse{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

 .composer{display:flex;gap:var(--sp-3);margin-top:var(--sp-4)}
 .composer textarea{flex:1;font:inherit;font-size:14px;padding:var(--sp-3) var(--sp-4);border-radius:var(--r-sm);
  border:1px solid var(--rule-strong);background:var(--paper-sheet);color:var(--ink);resize:none;line-height:1.5;
  max-height:9em;overflow-y:auto;transition:border-color var(--dur) var(--ease)}
 .composer textarea:hover{border-color:var(--ink-faint)}
 .composer textarea::placeholder{color:var(--ink-faint)}
 .composer .btn-primary{align-self:flex-end}
 /* Honest priority: this is the one filled, full-color control in most rows -- Send in the
    composer, and (below) "Make my sheet" in the actions row, which is the actual primary action
    of the whole screen and should look like it, not share a ghost-outline with two lesser actions. */
 .btn-primary{font:inherit;font-size:14px;font-weight:600;letter-spacing:.01em;padding:var(--sp-3) var(--sp-6);
  border-radius:var(--r-sm);border:1px solid var(--accent);background:var(--accent);color:var(--on-fill);
  cursor:pointer;transition:background var(--dur) var(--ease),border-color var(--dur) var(--ease)}
 .btn-primary:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
 /* Voice input (merged from the voice_mode spike, Alex ruling 07-31 04:25). One button, never a
    default, never auto-listening. Recording state is loud (solid pulsing button + outlined
    composer + live waveform) so the person is always certain whether the mic is live. Restyled
    onto quiet-editorial's ink/rule/danger tokens -- same mechanics as the cool-paper original. */
 .btn-mic{align-self:flex-end;font:inherit;font-size:17px;line-height:1;padding:var(--sp-3) var(--sp-4);
  border-radius:var(--r-sm);border:1px solid var(--rule-strong);background:var(--paper-sheet);
  color:var(--ink-soft);cursor:pointer;transition:color var(--dur) var(--ease),
  border-color var(--dur) var(--ease),background var(--dur) var(--ease)}
 .btn-mic:hover{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-tint)}
 .btn-mic[aria-pressed=true]{background:var(--danger-quiet);color:var(--on-fill);
  border-color:var(--danger-quiet);animation:micpulse 1.4s ease-in-out infinite}
 @keyframes micpulse{0%,100%{box-shadow:0 0 0 0 rgba(var(--danger-rgb),.4)}50%{box-shadow:0 0 0 7px rgba(var(--danger-rgb),0)}}
 .composer.recording{outline:2px solid var(--danger-quiet);outline-offset:3px}
 .mic-status{font-size:12px;color:var(--ink-soft);margin-top:var(--sp-1);min-height:15px}
 #micwave{display:none;align-self:flex-end;width:120px;height:30px;background:var(--paper-recessed);
  border:1px solid var(--rule);border-radius:var(--r-sm)}
 .composer.recording #micwave{display:block}
 @media print{.mic-status,#micwave,.btn-mic,.btn-resay{display:none!important}}
 /* "Clear and re-say" (deferred item, folded back in): sits next to the mic, not styled as a
    peer of it -- this is a quiet utility action, not a second primary voice control, so it gets
    the neutral ghost treatment (same family as .btn-ghost) rather than the mic's accent hover. */
 .btn-resay{align-self:flex-end;font:inherit;font-size:13px;line-height:1;padding:var(--sp-3) var(--sp-3);
  border-radius:var(--r-sm);border:1px solid var(--rule-strong);background:var(--paper-sheet);
  color:var(--ink-soft);cursor:pointer;white-space:nowrap;transition:color var(--dur) var(--ease),
  border-color var(--dur) var(--ease)}
 .btn-resay:hover{border-color:var(--ink);color:var(--ink)}
 .actions{display:flex;gap:var(--sp-2);margin-top:var(--sp-3);flex-wrap:wrap}
 /* Ghost actions carry no fill at rest or on hover -- hover darkens ink and border only, so the
    row never turns into three little colored boxes. */
 .btn-ghost{font:inherit;font-size:13px;padding:var(--sp-2) var(--sp-4);border-radius:var(--r-sm);
  border:1px solid var(--rule-strong);background:none;color:var(--ink-soft);cursor:pointer;
  transition:color var(--dur) var(--ease),border-color var(--dur) var(--ease)}
 .btn-ghost:hover{border-color:var(--ink);color:var(--ink)}
 /* Lowest priority of the three actions row buttons -- a text link, not a bordered button, so
    "Start over" (destructive, resets everything) never competes visually with the two actions
    that move the conversation forward. */
 .btn-quiet{font:inherit;font-size:12.5px;color:var(--ink-soft);background:none;border:none;
  cursor:pointer;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px;
  padding:var(--sp-2) var(--sp-2);border-radius:2px;transition:color var(--dur) var(--ease)}
 .btn-quiet:hover{color:var(--danger-text)}
 .starters{display:flex;flex-wrap:wrap;gap:var(--sp-2);margin-top:var(--sp-3)}
 /* Starter chips are the very first thing an undecided visitor sees -- they need to look like an
    invitation, not a menu of small print: a size-up over the body copy, a warm tinted fill (same
    accent-tint used elsewhere so it reads as "this app's warm interactive color", not a new hue),
    and a small directional glyph instead of relying on italics alone to say "tap me". */
 .starter{display:inline-flex;align-items:center;gap:7px;font:inherit;font-family:var(--font-serif);
  font-style:italic;font-size:14.5px;padding:11px var(--sp-5);border-radius:var(--r-pill);
  border:1px solid var(--accent-tint-2);background:var(--accent-tint);color:var(--accent-ink);cursor:pointer;
  transition:border-color var(--dur) var(--ease),color var(--dur) var(--ease),background var(--dur) var(--ease)}
 .starter:hover{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-tint-2)}
 .starter-ic{flex:none;width:13px;height:13px;opacity:.65;transition:opacity var(--dur) var(--ease)}
 .starter:hover .starter-ic{opacity:1}
 .status{display:flex;align-items:center;gap:6px;margin-top:var(--sp-3);font-size:12px;
  color:var(--ink-faint);min-height:16px}
 .status .dot,#where .dot{width:6px;height:6px;border-radius:50%;background:var(--pulse-warm);
  display:inline-block;flex:none}
 .status .dot.cloud,#where .dot.cloud{background:var(--cloud)}
 .refusal{background:none;border:none;border-left:2px solid var(--accent);
  border-radius:0;padding:var(--sp-2) var(--sp-4);font-size:13px;font-style:italic;font-family:var(--font-serif);
  color:var(--ink-soft);margin-top:var(--sp-2);animation:riseIn var(--dur) var(--ease) both}

 /* The sheet is the object of the app: a real page, lifted. Layered shadow instead of a flat
    border does the lifting; a warmer paper tone than everything around it does the "this is
    paper, that is chrome" distinction. */
 .sheet{background:var(--paper-sheet);border:1px solid var(--rule);border-radius:var(--r-card);
  padding:var(--sp-6) var(--sp-7) var(--sp-6);
  box-shadow:0 1px 2px rgba(var(--shadow-rgb),.05),0 10px 28px -14px rgba(var(--shadow-rgb),.22),0 28px 56px -30px rgba(var(--shadow-rgb),.14)}
 .sheet-title{margin:0;font-family:var(--font-serif);font-size:19px;font-weight:600;
  letter-spacing:-.005em;color:var(--ink)}
 .sheet-sub{display:block;margin:2px 0 var(--sp-5);padding-bottom:var(--sp-4);font-size:12px;
  font-style:italic;font-family:var(--font-serif);color:var(--ink-faint);border-bottom:1px solid var(--rule)}
 .placeholder{color:var(--ink-faint);font-family:var(--font-serif);font-style:italic;font-size:15.5px;
  line-height:1.6;padding:var(--sp-5) 0}

 /* ACCOMMODATIONS -- "how I need this visit to go", the patient's own statement of what the
    office should know before anything else. Sits above the sheet title's own body (#accom is
    placed before #body in the markup, and is INDEPENDENT of whether a sheet has been made yet --
    a person can fill this in before ever talking, or never talk at all). Real-world precedent:
    AASPIRE Healthcare Toolkit's accommodations report (Nicolaidis et al. 2016) cut clinician-
    reported communication barriers 4.07->2.82 (p<0.0001, n=259 patients + 51 PCPs) and was read
    by ~97% of clinicians -- this is a cite-and-extend of that finding, not a new idea. Uses the
    same tier-primary rail as "What matters most" (this IS the most-important-to-read thing, by
    the cited evidence) but its own label so it never reads as a fourth tier of clinical content --
    it isn't one. `.is-empty` (set in JS when nothing has been added yet) drops the whole block
    from PRINT ONLY, so an unused feature never prints a bare heading; on screen it always shows,
    including the invitation to add something, so it can be discovered before first use. */
 .accom{margin:0 0 var(--sp-6);padding:0 0 var(--sp-5);border-bottom:1px solid var(--rule)}
 .accom-title{font-variant-caps:all-small-caps;font-size:12.5px;letter-spacing:.08em;
  color:var(--tier-primary);font-weight:600}
 .accom-sub{display:block;margin-top:2px;font-family:var(--font-serif);font-style:italic;
  font-size:12px;color:var(--ink-faint)}
 .accom-items{display:flex;flex-direction:column;gap:var(--sp-2);margin-top:var(--sp-3)}
 .accom-item{display:flex;align-items:baseline;gap:var(--sp-3);justify-content:space-between;
  padding:7px 2px 7px var(--sp-4);border-left:2px solid var(--tier-primary);
  animation:riseIn var(--dur) var(--ease) both}
 .accom-item .txt{font-family:var(--font-serif);font-size:15.5px;line-height:1.5;color:var(--ink);
  flex:1;max-width:58ch}
 .accom-add{margin-top:var(--sp-3)}
 .accom-add summary{list-style:none;cursor:pointer;font-size:12.5px;color:var(--ink-soft);
  display:flex;align-items:center;gap:7px;border-radius:3px;padding:2px 4px;margin:-2px -4px;
  transition:color var(--dur) var(--ease)}
 .accom-add summary::-webkit-details-marker{display:none}
 .accom-add summary:hover{color:var(--ink)}
 .accom-add summary .chev{display:inline-block;width:8px;height:8px;
  border-right:1.5px solid var(--ink-faint);border-bottom:1.5px solid var(--ink-faint);
  transform:rotate(-45deg);transition:transform .15s}
 .accom-add[open] summary .chev{transform:rotate(45deg)}
 .accom-picker{display:flex;flex-wrap:wrap;gap:var(--sp-2);margin:var(--sp-3) 0}
 /* Chips are the CHOICE surface: selecting one is authoring, per the chosen_by:"patient"
    convention applied everywhere else a patient picks rather than types (see the starter-chip
    comment in <script> above) -- so a chip toggles a real add/remove, not a filter. */
 .accom-fam{border-top:1px solid var(--rule)}
 .accom-fam:first-child{border-top:none}
 .accom-fam>summary{cursor:pointer;list-style:none;padding:10px 2px;font-family:var(--font-sans);
   font-size:12.5px;letter-spacing:.02em;color:var(--ink-soft);display:flex;align-items:center;gap:8px}
 .accom-fam>summary::-webkit-details-marker{display:none}
 .accom-fam[open]>summary{color:var(--ink)}
 .accom-fam>summary:hover{color:var(--ink)}
 .accom-count{font-size:11px;color:var(--accent)}
 .accom-fam-in{display:flex;flex-direction:column;gap:6px;padding:0 0 10px 0}
 @media (max-width:700px){
  /* THE FOLD (measured 375x812, 2026-08-01): the hosted surface -- the one a QR code opens --
     ran 397px past the fold and put the composer at y=915, i.e. a stranger landed on this app
     and could not see the box they were meant to type into. 311px of that was chrome and
     caveat stacked above a single word of the patient's. Phones only; the demo laptop is
     untouched. */
  .ctrl-more{position:relative}
  .ctrl-more>summary{list-style:none;cursor:pointer;font-variant-caps:all-small-caps;
    letter-spacing:.02em;font-size:12px;color:var(--ink-soft);padding:4px 0;min-height:44px;
    display:flex;align-items:center}
  .ctrl-more>summary::-webkit-details-marker{display:none}
  .ctrl-more[open]>.ctrl-more-in{display:flex;flex-direction:column;gap:10px;padding:6px 0 2px}
  .ctrl-more:not([open])>.ctrl-more-in{display:none}
  /* `more` is a <details>, which is block-level and was taking a row of its own -- that made the
     header TALLER (143 -> 157px) than the flat controls it replaced. Force one row and let it
     wrap only if it must. */
  .ctrls{flex-direction:row;flex-wrap:wrap;align-items:center;gap:14px;row-gap:4px}
  .ctrl-more{display:inline-flex;align-items:center}
  .ctrl-more[open]{display:block;width:100%}
  .ctrl-more>summary{padding:0;min-height:32px}
  .ctrl-select{min-height:32px;max-width:100%}
  /* The "my words" caption is redundant once the options themselves say the consequence
     ("Stay on this device" / "Send my words to the cloud"). Dropping it on phones stops the
     wider honest labels from wrapping the control strip onto a second row -- the labels were
     the right change, the caption was the thing that could go. */
  .ctrl-select .ctrl-lab{display:none}
  /* A <select> sizes itself to its LONGEST option, and the honest labels are long -- it measured
     380px of a 462px strip, which pushed `more` onto a second row and made the header taller than
     the flat controls it replaced. Let it shrink instead: the chosen option still reads in full,
     and the list opens at full width when tapped. */
  .ctrl-select{flex:1 1 auto;min-width:0}
  .ctrl-select select{width:100%;min-width:0;max-width:100%;text-overflow:ellipsis}
  .ctrl-more{flex:0 0 auto}
  .tagline{display:none}            /* brand, not information */
  .brand-mark{width:28px;height:28px}
  header.chrome{padding-top:var(--sp-2);padding-bottom:var(--sp-2)}
  /* the cloud caveat COLLAPSES rather than disappears: it is the honest disclosure that this
     link is not the on-device product, and it appears on the one surface where strangers meet
     the app. Furniture, not content -- always present, never what the eye lands on first. */
  .hosted-note{padding:0}
  .hosted-note>details>summary{list-style:none;cursor:pointer;padding:10px 14px;min-height:44px;
    display:flex;align-items:center;font-size:12.5px}
  .hosted-note>details>summary::-webkit-details-marker{display:none}
  .hosted-note>details>div{padding:0 14px 12px}
.accom-fam>summary{min-height:44px}}
 .accom-chip{font:inherit;font-family:var(--font-serif);font-size:13px;text-align:left;
  padding:9px 13px;border-radius:var(--r-pill);border:1px solid var(--rule-strong);
  background:var(--paper-sheet-2);color:var(--ink-soft);cursor:pointer;max-width:34ch;
  transition:border-color var(--dur) var(--ease),color var(--dur) var(--ease),background var(--dur) var(--ease)}
 .accom-chip:hover{border-color:var(--accent);color:var(--ink)}
 .accom-chip[aria-pressed=true]{background:var(--accent-tint);border-color:var(--accent-tint-2);
  color:var(--accent-ink);font-weight:600}
 .accom-custom{display:flex;gap:var(--sp-2);margin-top:var(--sp-2)}
 .accom-custom textarea{flex:1;font:inherit;font-size:13.5px;padding:var(--sp-2) var(--sp-3);
  border-radius:var(--r-sm);border:1px solid var(--rule-strong);background:var(--paper-sheet);
  color:var(--ink);resize:vertical;min-height:2.4em}
 /* MY WEEK diary inputs -- same tokens as the accom textarea; the diary block itself reuses the
    .accom classes wholesale so screen, print and break rules apply unchanged. */
 .diary-in{font:inherit;font-size:13.5px;padding:var(--sp-2) var(--sp-3);border-radius:var(--r-sm);
  border:1px solid var(--rule-strong);background:var(--paper-sheet);color:var(--ink);min-width:0}
 #diaryDay{flex:0 1 140px}
 #diaryScore{flex:0 0 auto}
 #diaryNote{flex:1 1 130px}
 @media (max-width:700px){.accom-chip,.accom-add summary{min-height:44px}
  .accom-item{padding-top:10px;padding-bottom:10px}}

 /* A sheet-build failure gets the same quiet marginalia treatment as the chat error card, not a
    bare bracketed string overwriting the sheet body (grafted from A_precision_calm, restyled). */
 .sheet-error{display:flex;gap:var(--sp-3);align-items:flex-start;padding:var(--sp-2) 0 var(--sp-2) var(--sp-4);
  background:none;border:none;border-left:2px solid var(--danger-quiet);border-radius:0;
  animation:riseIn var(--dur) var(--ease) both}
 .sheet-error b{display:block;font-family:var(--font-serif);color:var(--ink);margin-bottom:4px;font-size:15.5px}
 .sheet-error span{color:var(--ink-soft);font-size:13px}

 /* Pull-quote treatment (v3.1 correction): the pin is the patient's own choice of the one thing
    that matters most, but a tinted color fill on their own words read as "flagged," not "treasured"
    (Alex's honoring-not-flagging ruling). Salience now comes from PRESENCE instead of hue: a
    neutral paper surface identical to the sheet's own (so nothing about it looks corrected), a
    stronger two-layer soft shadow lifting it further off the page than an ordinary tier line, more
    generous padding, and a noticeably bigger roman-weight serif line (23px vs. the ordinary line's
    16.5px) -- plus a large, low-opacity hanging quotation mark (echoing the same serif-glyph
    monogram language as the confidant's `.who-glyph`) as a quiet "these words are being quoted"
    flourish. The rail stays a thin colored line -- rails are still explicitly fine per the ruling,
    it is the big background wash that had to go. */
 .pinned{padding:var(--sp-4) var(--sp-5) var(--sp-5) var(--sp-6);margin:0 0 var(--sp-6);
  background:var(--paper-sheet);border-left:3px solid var(--tier-primary);border-radius:var(--r-card);
  box-shadow:0 2px 5px rgba(var(--shadow-rgb),.07),0 16px 34px -18px rgba(var(--shadow-rgb),.22);
  animation:riseIn 220ms var(--ease) both;
  display:flex;gap:var(--sp-3);align-items:flex-start;position:relative;overflow:hidden}
 /* The hanging quotation mark sits behind the flex content (z-index:-1 inside .pinned's own
    stacking context) so it can never obscure the text it is quietly decorating; overflow:hidden on
    .pinned keeps its oversized glyph from ever poking past the card's own rounded corner. */
 .pinned::before{content:'\\201C';position:absolute;left:8px;top:-14px;font-family:var(--font-serif);
  font-size:56px;line-height:1;color:var(--tier-primary);opacity:.14;pointer-events:none;z-index:-1}
 .pin-icon{flex:none;width:14px;height:14px;margin-top:4px;color:var(--tier-primary)}
 .pin-body{flex:1;min-width:0}
 .pin-label{font-variant-caps:all-small-caps;font-size:12.5px;letter-spacing:.08em;color:var(--tier-primary);
  margin-bottom:var(--sp-2);font-weight:600}
 .pinned .row{display:flex;align-items:baseline;gap:var(--sp-3);justify-content:space-between}
 .pinned .txt{font-family:var(--font-serif);font-weight:400;font-size:23px;line-height:1.48;
  color:var(--ink);flex:1;max-width:58ch}

 .tier{margin:0 0 var(--sp-5);animation:riseIn var(--dur) var(--ease) both}
 /* Color as wayfinding (v2): each tier carries its own warm color (set via --tier-color inline,
    see renderSheet()) on a small glyph and its underline rule, so the sheet reads as a scannable,
    color-coded map -- not three sections of identical gray small-caps that only differ by the
    words in the label. Label text itself stays neutral ink -- the color lives in the glyph + rule,
    not the copy -- so this stays a wayfinding system, not a coat of paint on the words. */
 .tierlab{font-variant-caps:all-small-caps;font-size:12.5px;letter-spacing:.09em;color:var(--ink-soft);
  margin-bottom:var(--sp-2);position:relative}
 /* .tlrow (icon+text) stays a separate inline flex row nested inside .tierlab so the ::after rule
    below keeps behaving as a block-level line under the whole label, exactly as before -- making
    .tierlab itself the flex container would turn ::after into a flex item instead. */
 .tierlab .tlrow{display:flex;align-items:center;gap:7px}
 .tierlab::after{content:'';display:block;height:1px;margin-top:6px;background:var(--tier-color,var(--rule-strong));
  opacity:.6;transform-origin:left;animation:ruleDraw 280ms var(--ease) both;animation-delay:50ms}
 .tricon{display:inline-flex;flex:none;width:11px;height:11px;color:var(--tier-color,var(--ink-faint))}
 .line{padding:9px 2px;border-radius:var(--r-line);cursor:pointer;
  transition:box-shadow var(--dur) var(--ease);animation:riseIn var(--dur) var(--ease) both}
 .line .row{display:flex;align-items:baseline;gap:var(--sp-3);justify-content:space-between}
 .line .txt{font-family:var(--font-serif);font-size:16.5px;line-height:1.55;color:var(--ink);flex:1;max-width:58ch}
 /* Chips read as quiet text labels, not buttons -- no border, no fill, just tracked caps that
    darken on interaction. The highlighter family (v2): a small warm dot travels with every
    "source" chip as a standing badge -- the same warm mark used for the receipt itself -- so the
    app's single most important interaction (see your own words verified) is recognizable at a
    glance everywhere it appears, and lights up fully once the receipt is actually open. */
 .chip{display:inline-flex;align-items:center;gap:5px;flex:none;text-align:right;font-family:var(--font-sans);
  font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-faint);border:none;
  border-radius:2px;padding:4px 2px;background:none;cursor:pointer;transition:color var(--dur) var(--ease)}
 .chip:hover{color:var(--accent-ink)}
 .chip-dot{width:6px;height:6px;border-radius:50%;flex:none;background:var(--mark-bg);
  border:1px solid var(--mark-ink);opacity:.6;transition:opacity var(--dur) var(--ease)}
 .chip:hover .chip-dot{opacity:.85}
 .pinbtn{flex:none;text-align:right;font-family:var(--font-sans);font-size:10.5px;
  letter-spacing:.07em;text-transform:uppercase;color:var(--ink-faint);border:none;
  border-radius:2px;padding:4px 2px;background:none;cursor:pointer;transition:color var(--dur) var(--ease);opacity:1}
 @media (hover:hover){.pinbtn{opacity:.5}.line:hover .pinbtn{opacity:1}}
 .pinbtn:hover{color:var(--accent-ink)}
 /* Hover is a margin mark, not a highlight wash: a hairline tick grows in from the left, the same
    visual language the "open" state already uses, just fainter. */
 .line:hover{box-shadow:inset 2px 0 0 var(--rule-strong)}
 /* Receipt-open state (v3.1 correction): the RAIL now uses --accent (a calm-trust wayfinding
    color, same family as the tier rails) rather than the highlighter's own mark-ink -- a v2 choice
    that put a hot underline on every open line. The CHIP still goes full highlighter-ink/-bg (below)
    since chips are explicitly still allowed the receipt-mark family: "this line's receipt is
    showing" reads as the (now-softened, parchment) receipt mark itself, everywhere it appears. */
 .line.open{background:none;box-shadow:inset 3px 0 0 var(--accent)}
 .line.open .txt{color:var(--ink)}
 .line.open:hover{box-shadow:inset 3px 0 0 var(--accent)}
 /* .pinned.open is included in every one of these -- the pinned line isn't a .line (it has its
    own markup), so it needs the same hook repeated or its own receipt-open state would silently
    fall back to the plain, unhighlighted chip while every other line's did not. */
 .line.open .chip,.pinned.open .chip{background:var(--mark-bg);color:var(--mark-ink);font-weight:700;
  border-radius:3px;padding:4px 7px}
 .line.open .chip-dot,.pinned.open .chip-dot{opacity:1;border-color:var(--mark-ink)}
 /* Footnote treatment: the receipt unfolds below the line like a note expanding, not a modal
    popping open -- a left rule instead of a full border box, and an "unfold" animation that grows
    the panel open rather than snapping it into place. Highlighter family (v2): the rule and the
    faint background wash pick up the same warm mark-ink/mark-bg tokens as the <mark> receipt
    itself, so the panel reads as "the highlighted passage's own frame" rather than a neutral box
    that happens to contain one. */
 .source-panel{margin:var(--sp-1) 4px 0;padding:var(--sp-2) var(--sp-4);background:var(--mark-bg-faint);
  border:none;border-left:2px solid var(--mark-ink);border-radius:0 var(--r-sm) var(--r-sm) 0;
  font-family:var(--font-serif);font-size:14.5px;line-height:1.65;color:var(--ink-soft);max-height:150px;
  overflow-y:auto;animation:unfold 220ms var(--ease) both;transform-origin:top}
 .source-label{font-family:var(--font-sans);font-variant-caps:all-small-caps;font-size:11.5px;
  letter-spacing:.08em;color:var(--ink-faint);margin-bottom:var(--sp-2);display:block}
 .line-tools{display:flex;justify-content:flex-end;margin-top:var(--sp-2)}
 .quiet-remove{font-family:var(--font-sans);font-size:12px;color:var(--ink-soft);background:none;border:none;
  cursor:pointer;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px;
  padding:2px 4px;border-radius:2px;transition:color var(--dur) var(--ease)}
 .quiet-remove:hover{color:var(--danger-text)}
 /* "Ask me more about this" sits in the same quiet-text-link family as Remove-from-sheet (same
    row, same weight) but hovers toward --accent-ink rather than --danger-text: this action adds,
    it does not destroy, and should never borrow a delete affordance's hover color. */
 .quiet-more{font-family:var(--font-sans);font-size:12px;color:var(--ink-soft);background:none;
  border:none;cursor:pointer;text-decoration:underline;text-decoration-style:dotted;
  text-underline-offset:3px;padding:2px 4px;border-radius:2px;margin-left:var(--sp-3);
  transition:color var(--dur) var(--ease)}
 .quiet-more:hover{color:var(--accent-ink)}
 /* A probe-answer line carries a small badge naming which clinical element it answers ("Onset",
    "Duration"...) so its relationship to the line it was asked about stays legible even though it
    renders in the flat tier list rather than nested under its parent (a known simplification --
    see BUILD_REPORT.md). Colored off --tier-secondary since these lines always land in that tier. */
 .probe-tag{display:inline-block;font-family:var(--font-sans);font-size:10px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--tier-secondary);background:var(--tier-secondary-tint);
  border-radius:3px;padding:1px 6px;margin-right:7px;white-space:nowrap;vertical-align:1px}
 .probe-ta{display:block;width:100%;font:inherit;font-size:13.5px;padding:8px 9px;
  border:1px solid var(--rule-strong);border-radius:var(--r-sm);background:var(--paper-sheet);
  color:var(--ink);resize:vertical;min-height:2.8em;margin-top:6px}
 .probe-ta:hover,.probe-ta:focus{border-color:var(--accent)}
 .mine{font-family:var(--font-sans);font-size:11px;color:var(--accent-ink);letter-spacing:.02em;white-space:nowrap}
 .helped{font-family:var(--font-sans);font-size:11px;color:var(--accent-ink);letter-spacing:.02em;white-space:nowrap}
 /* Marginalia, not a dashed sticky-note box: a left rule aside, same family as the source panel. */
 .draft-card{margin:var(--sp-2) 4px 0;padding:var(--sp-1) 0 var(--sp-2) var(--sp-4);background:none;
  border:none;border-left:2px solid var(--accent-tint-2);border-radius:0;
  animation:riseIn var(--dur) var(--ease) both}
 .draft-label{display:block;font-family:var(--font-sans);font-variant-caps:all-small-caps;font-size:11.5px;
  letter-spacing:.08em;color:var(--accent-ink);margin-bottom:var(--sp-2)}
 .draft-text{font-family:var(--font-serif);font-size:15.5px;line-height:1.5;color:var(--ink);max-width:58ch}
 .draft-cautions{font-family:var(--font-sans);font-size:12px;color:var(--danger-text);
  margin-top:var(--sp-2);line-height:1.4}
 .draft-acts{display:flex;gap:var(--sp-2);margin-top:var(--sp-3)}
 .draft-acts button{font:inherit;font-size:12.5px;padding:var(--sp-2) var(--sp-4);
  border-radius:var(--r-sm);cursor:pointer;transition:background var(--dur) var(--ease),border-color var(--dur) var(--ease),color var(--dur) var(--ease)}
 .draft-accept{background:var(--accent);color:var(--on-fill);border:1px solid var(--accent)}
 .draft-accept:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
 .draft-keep{background:none;color:var(--ink-soft);border:1px solid var(--rule-strong)}
 .draft-keep:hover{border-color:var(--ink);color:var(--ink)}

 details.drawer{margin:var(--sp-4) 0 var(--sp-5);border-top:1px solid var(--rule);padding-top:var(--sp-3)}
 details.drawer summary{list-style:none;cursor:pointer;font-size:12.5px;color:var(--ink-soft);
  display:flex;align-items:center;gap:7px;border-radius:3px;padding:2px 4px;margin:-2px -4px;
  transition:color var(--dur) var(--ease)}
 details.drawer summary::-webkit-details-marker{display:none}
 details.drawer summary:hover{color:var(--ink)}
 details.drawer summary .chev{display:inline-block;width:8px;height:8px;border-right:1.5px solid var(--ink-faint);
  border-bottom:1.5px solid var(--ink-faint);transform:rotate(-45deg);transition:transform .15s}
 details.drawer[open] summary .chev{transform:rotate(45deg)}
 /* The set-aside list is a manuscript margin list -- each item divided by a dotted rule, not
    boxed as a group. */
 .drawer-body{margin:var(--sp-2) 2px 0;padding:var(--sp-1) 0 0;background:none;
  border:none;border-radius:0}
 .drawer-body .row{display:flex;justify-content:space-between;align-items:baseline;gap:var(--sp-3);
  padding:var(--sp-2) 2px;border-bottom:1px dotted var(--rule)}
 .drawer-body .row:last-child{border-bottom:none}
 .drawer-body .txt{font-family:var(--font-serif);font-size:15px;color:var(--ink-soft)}
 .addback{font-family:var(--font-sans);font-size:12px;color:var(--accent-ink);background:none;border:none;
  cursor:pointer;text-decoration:underline;text-decoration-style:dotted;white-space:nowrap;border-radius:2px;
  transition:color var(--dur) var(--ease)}
 .addback:hover{color:var(--accent-ink);text-decoration-style:solid}

 /* Depth variation (v2): the safety block is furniture, not content -- it now recedes a step
    further than before (quieter label color, a touch of opacity) so that, next to the newly
    colorful tiers and the warmer pinned/CTA treatment above it, this section visibly reads as the
    LEAST salient thing on the page, matching what it actually is: generic reference text that is
    the same for everyone, included every time, never the thing an ADHD-scanning eye should land
    on first. */
 .safety-lab{display:flex;justify-content:space-between;align-items:baseline;gap:var(--sp-3);
  font-variant-caps:all-small-caps;font-size:12px;letter-spacing:.08em;color:var(--ink-faint);
  margin:18px 0 var(--sp-2);border-top:1px solid var(--rule);padding-top:var(--sp-3);opacity:.85}
 .locked{font-variant-caps:normal;font-size:11px;color:var(--ink-faint);letter-spacing:0}
 .safety-lab a{color:var(--accent-ink);text-decoration:none;border-bottom:1px dotted var(--accent-ink)}
 /* Safety copy reads as reference material -- an endnote, not a warning card: no border box, just
    quieter type below its rule. */
 .safety{background:none;border:none;border-radius:0;max-width:64ch;opacity:.85;
  padding:0;font-size:12.5px;line-height:1.6;color:var(--ink-soft);white-space:pre-wrap;margin-top:var(--sp-1)}

 .stats{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--ink-faint);margin-top:var(--sp-3)}
 .sheetacts{display:flex;gap:var(--sp-3);margin-top:var(--sp-6);flex-wrap:wrap;align-items:center}
 .sheetacts button{font-size:12px;letter-spacing:.03em;text-transform:uppercase;padding:var(--sp-3) var(--sp-5);
  border-radius:var(--r-sm);border:1px solid transparent;cursor:pointer;font-family:var(--font-sans);
  transition:background var(--dur) var(--ease),border-color var(--dur) var(--ease),color var(--dur) var(--ease)}
 .speak-wrap{margin-top:var(--sp-2)}
 @media print{.speak-wrap{display:none!important}}
 .sheetacts .primary{background:var(--accent);color:var(--on-fill);border-color:var(--accent)}
 .sheetacts .primary:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
 .sheetacts .secondary{background:none;color:var(--ink);border-color:var(--rule-strong)}
 .sheetacts .secondary:hover{border-color:var(--ink);color:var(--ink)}
 .copied{font-size:12px;font-style:italic;font-family:var(--font-serif);color:var(--accent-ink);align-self:center}

 /* A colophon line, like the printer's note at the foot of a book page. */
 footer.app-footer{max-width:1180px;margin:0 auto;padding:var(--sp-5) var(--sp-7) var(--sp-7);
  border-top:1px solid var(--rule);font-family:var(--font-serif);font-style:italic;
  font-size:12px;color:var(--ink-faint);text-align:center}
 .hosted-note{background:var(--accent-tint);border-bottom:1px solid var(--accent-tint-2);
  padding:var(--sp-2) var(--sp-4);font-size:12px;line-height:1.5;color:var(--accent-ink);text-align:center}
 .fb{margin-top:var(--sp-3)}
 .fb-open{font:inherit;font-size:12px;color:var(--ink-soft);background:none;border:none;cursor:pointer;
  text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px;padding:2px 4px;border-radius:2px;
  transition:color var(--dur) var(--ease)}
 .fb-open:hover{color:var(--ink)}
 .fb-panel{margin-top:var(--sp-2);padding:var(--sp-3);background:var(--paper-sheet-2);border:1px solid var(--rule);
  border-radius:var(--r-sm)}
 .fb-panel textarea{width:100%;font:inherit;font-size:13px;padding:var(--sp-2);border:1px solid var(--rule);
  border-radius:var(--r-sm);background:var(--paper-sheet);color:var(--ink);resize:vertical}
 .fb-row{display:flex;gap:var(--sp-3);align-items:center;margin-top:var(--sp-2)}
 .fb-consent{font-size:11px;color:var(--ink-faint);flex:1;line-height:1.4}
 .fb-send{font:inherit;font-size:12.5px;padding:var(--sp-2) var(--sp-4);border-radius:var(--r-sm);border:none;
  background:var(--accent);color:var(--on-fill);cursor:pointer;transition:background var(--dur) var(--ease)}
 .fb-send:hover{background:var(--accent-hover)}
 .fb-status{font-size:12px;color:var(--accent-ink);margin-top:var(--sp-1);min-height:14px}

 /* PASTE-IN: bring in a conversation someone else had about/with the patient. Same quiet-panel
    family as .fb-panel above (a box that recedes until opened), but its own namespace since the
    content inside carries real stakes -- a wrong role guess here would put someone else's words
    on the sheet wearing the patient's receipt, so every visual decision favors legibility over
    decoration: the two role groups get a plain tint + rail (no cleverness), the confirm question
    is the single largest thing in the panel, and nothing in here uses the mark/highlighter family
    (that is reserved for words already confirmed as the patient's own). */
 .paste-panel{margin-top:var(--sp-4);padding:var(--sp-5);background:var(--paper-sheet-2);
  border:1px solid var(--rule);border-radius:var(--r-card);animation:riseIn var(--dur) var(--ease) both}
 .paste-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--sp-2)}
 .paste-title{font-family:var(--font-serif);font-weight:600;font-size:15.5px;color:var(--ink)}
 .paste-hint{font-size:12.5px;color:var(--ink-soft);line-height:1.5;margin:0 0 var(--sp-3)}
 .paste-kind{display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--ink-soft);
  margin-bottom:var(--sp-3)}
 .paste-kind input{accent-color:var(--accent);margin-right:6px}
 .paste-name{width:100%;font:inherit;font-size:13px;padding:var(--sp-2) var(--sp-3);
  border:1px solid var(--rule-strong);border-radius:var(--r-sm);background:var(--paper-sheet);
  color:var(--ink);margin-bottom:var(--sp-3)}
 .paste-panel textarea{width:100%;font:inherit;font-size:13.5px;padding:var(--sp-3);
  border:1px solid var(--rule-strong);border-radius:var(--r-sm);background:var(--paper-sheet);
  color:var(--ink);resize:vertical;line-height:1.5}
 .paste-acts{margin-top:var(--sp-3)}
 .paste-turns{margin-top:var(--sp-4)}
 /* Role groups: a plain tint + left rail, deliberately NOT the mark/highlighter family (that
    means "confirmed as yours"; this is still a proposal). Patient-guessed turns get the app's own
    accent tint (same family as a starter chip); assistant-guessed and unattributed turns share a
    quieter neutral -- the point is patient-vs-not, not a three-way palette. */
 .pt-row{padding:var(--sp-2) var(--sp-3);margin-bottom:6px;border-radius:var(--r-sm);
  font-family:var(--font-serif);font-size:14px;line-height:1.5}
 .pt-patient{background:var(--accent-tint);color:var(--ink);border-left:3px solid var(--accent)}
 .pt-assistant{background:var(--paper-recessed);color:var(--ink-soft);border-left:3px solid var(--rule-strong)}
 .pt-lab{display:block;font-family:var(--font-sans);font-variant-caps:all-small-caps;font-size:10.5px;
  letter-spacing:.07em;color:var(--ink-faint);margin-bottom:3px}
 .paste-confirm{margin-top:var(--sp-4);padding-top:var(--sp-3);border-top:1px solid var(--rule)}
 .paste-confirm-q{font-family:var(--font-serif);font-size:15px;color:var(--ink);margin-bottom:var(--sp-3)}
 .paste-confirm-acts{display:flex;gap:var(--sp-3);flex-wrap:wrap}
 /* Recovery list: the patient's job here is RECOVERY not approval -- everything starts excluded
    (no checkbox pre-checked, nothing tinted as if already accepted) and "This is mine" is the only
    action, one click per line they want to pull back in. */
 .recover-item{display:flex;justify-content:space-between;align-items:flex-start;gap:var(--sp-3);
  padding:var(--sp-2) 2px;border-bottom:1px dotted var(--rule)}
 .recover-item:last-child{border-bottom:none}
 .recover-txt{font-family:var(--font-serif);font-size:14.5px;color:var(--ink-soft);line-height:1.5;flex:1}
 .recover-added{color:var(--accent-ink);font-size:12px;white-space:nowrap;align-self:center}
 .topic-tray{display:flex;flex-wrap:wrap;gap:var(--sp-2);margin-top:var(--sp-3)}
 .paste-added-note{font-size:12.5px;color:var(--accent-ink);margin-top:var(--sp-3);font-style:italic;
  font-family:var(--font-serif)}
 /* Small caption over a pasted-in patient bubble, mirroring .msg.u .who but right-aligned to sit
    over the right-aligned "you" bubble it labels -- the one visible trace, in the transcript
    itself, that a line arrived by paste rather than by typing, so nothing about its origin is
    hidden from the person it belongs to. */
 .msg.you .who.src-tag{display:block;text-align:right;font-variant-caps:all-small-caps;
  font-size:11px;letter-spacing:.06em;color:var(--ink-faint);margin:0 2px 4px;font-family:var(--font-sans)}
 @media print{.hosted-note,.fb,.paste-panel{display:none!important}}

 /* MOBILE. Below 700px the two columns become two tabs (Conversation / My sheet); making a
    sheet auto-switches to the sheet tab. Relative sizing throughout — no hardcoded artboard
    width (the 390px pin in the mocks was a headless-Chrome screenshot workaround, not design). */
 @media (max-width:700px){
  .tabbar{display:flex;position:sticky;top:0;z-index:5;background:var(--paper-sheet-2);
   border-bottom:1px solid var(--rule-strong);padding:var(--sp-2) var(--sp-3);gap:var(--sp-2)}
  .tabbtn{flex:1;font-size:13.5px;padding:10px 0;border-radius:var(--r-pill);border:1px solid transparent;
   background:none;color:var(--ink-soft);cursor:pointer;min-height:44px;transition:color var(--dur) var(--ease)}
  .tabbtn[aria-selected=true]{background:var(--accent-tint);color:var(--accent-ink);border-color:var(--accent-tint-2);font-weight:600}
  header.chrome{padding:var(--sp-3) var(--sp-4)}
  .ctrls{font-size:11.5px;gap:var(--sp-3)}
  .wrap{flex-direction:column;padding:var(--sp-3) var(--sp-3) var(--sp-6);gap:var(--sp-4)}
  .chat{height:52vh}
  .sheet{padding:var(--sp-4) var(--sp-4) var(--sp-5)}
  /* Touch targets: the quieter, text-only chip/pin labels still need >=44px of tappable height,
     and the pin toggle can't rely on hover to reveal itself on a touch screen. */
  .btn-primary,.btn-ghost,.btn-quiet{padding-top:12px;padding-bottom:12px;min-height:44px;
   display:inline-flex;align-items:center}
  .starter{padding-top:11px;padding-bottom:11px}
  .chip,.pinbtn{padding:12px 4px;opacity:1}
  .line{padding:6px 2px}
  /* Tab-switch fade (grafted from A_precision_calm): showTab() removes+re-adds this class so the
     CSS animation replays on every Conversation/My sheet switch, not just the first. */
  #chatcol.tab-in,#sheetcol.tab-in{animation:tabIn var(--dur) var(--ease) both}
  @keyframes tabIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
  body.show-sheet #chatcol{display:none}
  body:not(.show-sheet) #sheetcol{display:none}
 }

 /* PRINT. What comes out of the printer must be the document, not a screenshot of an app.
    On-screen affordances drop; the patient's words take the visual majority; every word of the
    safety information still prints, set as reference material (small, two columns, below a rule).
    Type does the ranking, not deletion. */
 #printhead{display:none}
 /* ?screen=min variant (see the UX-VARIANTS note in the script): the two sheet-stage actions
    stay hidden until a sheet exists. Inert unless the html element carries screen-min. */
 html.screen-min:not(.has-sheet) .needs-sheet{display:none}
 @media print{
  header.chrome,.tabbar,#chatcol,.sheetacts,.source-panel,.source-label,.stats,.chip,.pinbtn,
  .line-tools,#refusal,.status,details.drawer,footer.app-footer,.mine,.sheet-title,.sheet-sub,
  .draft-card,.sheet-error,.accom-add{display:none!important}
  /* Two print leaks found 2026-08-01 by exercising the print path directly (both PRE-EXISTING,
     both only visible on paper, which is why neither showed up on screen):
     1. `.quiet-remove` was never in the hide-list above. `.line-tools` hides the remove control
        for a normal sheet line, but the safety-cue block's remove button sits bare in `.row`
        with no `.line-tools` wrapper -- so a cue's "remove" link PRINTED on the page a clinician
        receives. A control, on the doctor's copy.
     2. `.placeholder` is the on-screen instruction to the PATIENT ("Talk for a bit, then press
        Make my sheet."). A Ctrl+P before a sheet exists printed that instruction into the
        document. Not reachable from the sheet-actions Print button, but a bare browser print is. */
  .quiet-remove,.placeholder{display:none!important}
  /* .accom.is-empty is set client-side (renderAccom()) when nothing has been added yet -- an
     unused block should not print a bare "How I need this visit to go" heading with nothing
     under it; on screen it still shows (with the add-affordance) so the feature is discoverable
     before first use. */
  .accom.is-empty{display:none!important}
  /* "said with help" DOES print — the sheet is honest about which lines were worded with help */
  .helped{font-size:7.5pt;color:#333;font-style:italic}
  body{background:#fff;color:#000}
  .wrap{display:block;padding:0;max-width:none;margin:0}
  .col{width:100%}
  #sheetcol{display:block!important}
  #printhead{display:block;border-bottom:1.5pt solid #000;padding-bottom:5pt;margin-bottom:12pt}
  #printhead b{font-size:13pt;letter-spacing:.01em}
  #printhead span{display:block;font-size:8.5pt;color:#333;margin-top:2pt}
  .sheet{border:none;border-radius:0;padding:0;background:#fff;box-shadow:none}
  /* Accommodations print ABOVE everything else in #sheet (DOM order: #printhead, #accom, #body --
     #accom comes before the pinned quote and every tier), matching "the part a clinician actually
     reads first" -- this is the AASPIRE-precedent block, see the <style> comment above .accom. */
  .accom{border-bottom:1pt solid #000;padding-bottom:8pt;margin-bottom:14pt}
  .accom-title{font-size:8pt;letter-spacing:.1em;color:#000}
  .accom-sub{display:none}
  .accom-items{margin-top:6pt}
  .accom-item{border-left:2pt solid #000;padding:2pt 0 2pt 8pt;animation:none}
  .accom-item .txt{font-size:12.5pt;line-height:1.5;color:#000;max-width:none}
  .pinned{background:none;border:none;border-left:2pt solid #000;border-radius:0;padding:2pt 0 2pt 8pt;
   margin-bottom:10pt;box-shadow:none}
  .pin-icon,.tricon,.chip-dot,.who-glyph,.starter-ic{display:none}
  .pinned::before{content:none}
  .pin-label{font-size:8pt;letter-spacing:.1em;color:#000}
  .pinned .txt{font-size:13.5pt;color:#000;max-width:none}
  .tierlab{font-size:8pt;letter-spacing:.1em;color:#000;margin-bottom:3pt}
  .tierlab::after{animation:none;background:#000}
  .tier{margin-bottom:13pt}
  .line{padding:1pt 0;cursor:default;border:none;animation:none}
  .line .txt{font-size:12.5pt;line-height:1.5;color:#000;max-width:none}
  .line.open,.line:hover{background:none;box-shadow:none}
  /* Real bug found in this pass, unrelated to the animation fix above: the screen rule
     `.line.open .txt{color:var(--ink)}` (three simple selectors) OUTRANKS the plain print rule just
     above (`.line .txt`, two selectors) on specificity alone -- source order never gets consulted,
     specificity is compared first. Harmless every theme built before this pass, since --ink was
     always dark; the Quiet Dark theme's --ink is a near-white warm tone for on-screen legibility,
     so printing a line whose receipt happened to be open while Quiet Dark was the active theme
     printed that line's text in near-white on white paper -- confirmed by rendering the PDF to PNG
     and seeing the open line's text nearly vanish, while the pinned line (no equivalent `.pinned.
     open .txt` screen override exists) printed solid black correctly. Match the specificity here so
     print always wins regardless of which theme happened to be on screen. */
  .line.open .txt{color:#000}
  /* opacity:1 here overrides the screen-only .85 recede (v2) -- print accessibility must not
     inherit a screen-scanning affordance; this population includes people with visual impairment,
     so full-opacity #333/#222 ink is the floor regardless of how the on-screen hierarchy reads. */
  .safety-lab{font-variant-caps:normal;font-size:7.5pt;margin:16pt 0 4pt;border-top:.75pt solid #666;padding-top:5pt;color:#333;opacity:1}
  .safety-lab .locked{font-size:7.5pt}
  /* 8.5pt: small enough to recede, large enough to actually read — this population includes
     people with visual impairment. Two columns does the height reduction, not smaller type. */
  .safety{background:#fff;border:none;padding:0;font-size:8.5pt;line-height:1.35;color:#222;opacity:1;
          columns:2;column-gap:16pt;max-width:none}
  a{color:#000;text-decoration:none}
  /* Screen-only entrance motion (riseIn/unfold) must not survive into the printed page: a
     fill-mode:both animation holds an element at its FROM keyframe (opacity:0) until it plays,
     and print rendering does not run it — so without this reset the pinned line and every tier
     would print blank. `.line` (each individual clause, nested inside `.tier`) carries its OWN
     independent riseIn animation -- opacity does not inherit, so resetting only the parent `.tier`
     would leave the lines inside it unprotected; added here defensively for the same reason,
     alongside the (separately confirmed, see above) `.line.open .txt` specificity fix -- that one
     was the actual bug this pass turned up; this animation reset is consistency/hygiene so `.line`
     gets the identical guarantee `.pinned`/`.tier` already had. */
  .pinned,.tier,.line,.accom-item{animation:none;opacity:1;transform:none}
  /* No page/column break may orphan a heading from its content or split a single line/item --
     the print layout had never been human-checked before today (review find 2026-08-01). */
  .tierlab,.safety-lab,.accom-title{break-after:avoid}
  .line,.pinned,.accom-item{break-inside:avoid}
 }
</style></head><body>
<div class=tabbar id=tabbar>
 <button class=tabbtn id=tabchat aria-selected=true onclick="showTab(false)">Conversation</button>
 <button class=tabbtn id=tabsheet aria-selected=false onclick="showTab(true)">My sheet</button>
</div>
<header class=chrome>
 <div class=brand><svg class=brand-mark viewBox="0 0 160 160" aria-hidden="true"><g fill="currentColor"><path d="M56,50 C46,50 40,58 40,68 C40,78 47,85 56,85 C56,96 48,104 38,107 L38,116 C55,113 68,101 68,79 C68,62 63,50 56,50 Z"/><path d="M104,110 C114,110 120,102 120,92 C120,82 113,75 104,75 C104,64 112,56 122,53 L122,44 C105,47 92,59 92,81 C92,98 97,110 104,110 Z"/></g></svg><span class=brand-text><span class=wordmark>__APP_NAME__</span><span class=tagline>__TAGLINE__</span></span></div>
 <span class=ctrls>
   <!-- Sorted by WHO EACH CONTROL SERVES (Alex, 2026-08-01 01:30: "all of the controls should
        serve the user, not just be arbitrarily placed"). The runtime picker is the ONE control
        here that is genuinely the patient's: "where do my words go" is the product's whole
        promise, made operable. It was wearing developer clothes -- model names are for us, the
        consequence is for them -- so it stays visible and gets labelled by consequence. Theme is
        also the patient's (light/visual load is real for this population) but nobody needs it in
        the first second, so it moves behind `more`. "deep reasoning" serves US: a patient cannot
        know what it means or when to want it, and pays 190s+ for it -- it lives behind `more`
        too, reachable for demoing, not sitting on a frightened person's phone screen. -->
   <label class=ctrl-select><span class=ctrl-lab>my words</span>
   <select id=backend>__BACKEND_OPTIONS__</select></label>
   <details class=ctrl-more><summary aria-label="More settings">more</summary>
    <div class=ctrl-more-in>
     <label class=ctrl-select><span class=ctrl-lab>appearance</span>
     <select id=theme onchange="document.documentElement.dataset.theme=this.value">
      <option value=blue selected>Calm blue</option>
      <option value=warm>Warm paper</option>
      <option value=dark>Quiet dark</option>
     </select></label>
     <label class=ctrl-toggle><input type=checkbox id=deep> deep reasoning (note)</label>
    </div>
   </details>
 </span>
</header>
__HOSTED_BANNER__
<div class=wrap>
 <div class=col id=chatcol>
  <div class=chat id=chat></div>
  <div class=starters id=starters>
   <button class=starter onclick=startWith(this)><svg class=starter-ic viewBox="0 0 24 24" aria-hidden=true><path d="M4 12h13M12 6l7 6-7 6" fill=none stroke=currentColor stroke-width=2 stroke-linecap=round stroke-linejoin="round"/></svg>Help me get ready for a doctor visit</button>
   <button class=starter onclick=startWith(this)><svg class=starter-ic viewBox="0 0 24 24" aria-hidden=true><path d="M4 12h13M12 6l7 6-7 6" fill=none stroke=currentColor stroke-width=2 stroke-linecap=round stroke-linejoin="round"/></svg>There's something I find hard to bring up</button>
   <button class=starter onclick=startWith(this)><svg class=starter-ic viewBox="0 0 24 24" aria-hidden=true><path d="M4 12h13M12 6l7 6-7 6" fill=none stroke=currentColor stroke-width=2 stroke-linecap=round stroke-linejoin="round"/></svg>I'm not sure where to start</button>
  </div>
  <div class=composer id=composerRow><textarea id=inp rows=1 placeholder="Tell me what's on your mind…"></textarea>__MIC_UI__<button class=btn-primary onclick=send()>Send</button></div>
  <div class=mic-status id=micstatus aria-live=polite></div>
  <div class=actions><button class=btn-primary onclick=makeSheet()>Make my sheet →</button><button class="btn-ghost needs-sheet" onclick=betterWords()>Say it better for me</button><button class=btn-ghost onclick=togglePaste()>Paste a conversation</button><button class="btn-ghost needs-sheet" onclick=runEndRound()>One more look — before you go</button><button class=btn-quiet onclick=reset()>Start over</button></div>
  <div class=status id=hint></div>
  <div id=refusal></div>
  <div id=pastePanel class=paste-panel style=display:none>
   <div class=paste-head><span class=paste-title>Bring in something you already have</span>
    <button class=btn-quiet onclick=closePaste()>Close</button></div>
   <p class=paste-hint>Paste a conversation you had with an AI assistant (ChatGPT, Claude, Gemini…)
    about this, or a note from someone who helps you. Nothing here reaches your sheet until you
    say so.</p>
   <div class=paste-kind>
    <label><input type=radio name=pastekind value=conversation checked onchange=switchPasteKind()> A conversation with an AI</label>
    <label><input type=radio name=pastekind value=advocate onchange=switchPasteKind()> A note from someone who helps me</label>
   </div>
   <div id=advocateName style=display:none><input id=advName type=text placeholder="Their name (e.g. Maria)" class=paste-name></div>
   <textarea id=pasteText rows=7 placeholder="Paste here…"></textarea>
   <div class=paste-acts><button class=btn-primary onclick=submitPaste()>Look at this</button></div>
   <div id=pasteResult></div>
  </div>
__FEEDBACK_UI__
 </div>
 <div class=col id=sheetcol>
  <div class=sheet id=sheet>
   <div id=printhead><b>Notes for my appointment</b>
    <span>Written by me before this visit, in my own words. Prepared with __APP_NAME__ + Gemma 4.</span>
    <span id=printver></span></div>
   <h2 class=sheet-title>Notes for my appointment</h2>
   <span class=sheet-sub>Built from your own words — nothing added <span id=where></span></span>
   <div id=accom></div>
   <div id=diary></div>
   <div id=body class=placeholder>Talk for a bit, then press <b>Make my sheet</b>.</div></div>
 </div>
</div>
<footer class=app-footer>__FOOTER_CLAIM__</footer>
<script>
const NAME='__APP_NAME__';
// Whether the server can render speech (Windows + PowerShell/System.Speech, see gate/speech.py).
// Same graceful-degrade pattern as the mic: __MIC_UI__ is only injected when STT_AVAILABLE, and
// the read-aloud button below only renders when this is true.
const TTS_AVAILABLE=__TTS_AVAILABLE__;
let msgs=[];
// Patient bubble <p> nodes in send order — the in-bubble receipt highlight maps transcript
// offsets back onto these. Raw text is kept in dataset so marks can be cleared losslessly.
let YOUB=[];
// prefers-reduced-motion is read live (not cached) so a mid-session OS toggle takes effect on the
// very next scroll or thinking-dot render (grafted from A_precision_calm).
function reduceMotion(){return matchMedia('(prefers-reduced-motion: reduce)').matches;}
function scrollChat(){chat.scrollTo({top:chat.scrollHeight,behavior:reduceMotion()?'auto':'smooth'});}
function bubble(t,who,opts){opts=opts||{};let d=document.createElement('div');d.className='msg '+who;
 let p=document.createElement('p');p.textContent=t;
 // Anchor moment (v2): the confidant's replies get a small warm monogram glyph next to the
 // NAME label -- a fixed visual landmark, not just re-reading small caps every time to know
 // who's speaking. Built with textContent/appendChild (not innerHTML) so NAME can never be
 // interpreted as markup.
 if(who==='u'){let w=document.createElement('span');w.className='who';
  let g=document.createElement('span');g.className='who-glyph';g.setAttribute('aria-hidden','true');
  g.textContent='”';w.appendChild(g);w.appendChild(document.createTextNode(NAME));d.appendChild(w);}
 else{p.dataset.raw=t;YOUB.push(p);
  // PASTE-IN provenance stays visible in the transcript itself, not just in the paste panel that
  // is about to close: a small caption over the bubble says this line arrived by paste, same
  // textContent-only construction as NAME above so pasted text can never be read as markup.
  if(opts.pasted){let w=document.createElement('span');w.className='who src-tag';
   w.textContent='from a pasted conversation';d.appendChild(w);}}
 d.appendChild(p);chat.appendChild(d);scrollChat();return p;}
// Graceful error presentation in the conversation (grafted from A_precision_calm): a quiet card
// (icon + calm phrasing, the real detail kept small underneath) instead of a raw bracketed string
// inside a patient-style bubble. Styling is in <style> under .msg.err/.err-ic/.err-detail.
function errBubble(msg){let d=document.createElement('div');d.className='msg err';
 let p=document.createElement('p');
 p.innerHTML='<span class=err-ic aria-hidden=true>!</span><span>Something didn&#8217;t come through.'
   +'<span class=err-detail>'+esc_(msg)+'</span></span>';
 d.appendChild(p);chat.appendChild(d);scrollChat();}
function badge(where,ms){let s=(ms/1000).toFixed(1)+'s';return where==='cloud'
  ?'<span class="dot cloud"></span> left your device · '+s
  :'<span class=dot></span> stayed on this device · '+s;}
// Animated "thinking" hint (grafted from A_precision_calm) — three pulsing dots + the label,
// replacing a static "..." ellipsis, used for every wait state (chat reply, sheet build, polish
// draft). Reduced-motion freezes the dots via the global media query in <style>.
function thinkingHTML(label){return '<span class=thinking><span class=td></span><span class=td></span>'
  +'<span class=td></span> '+esc_(label)+'…</span>';}
// One model call at a time: an anxious double-tap during a slow local build must never fire
// concurrent requests (replies land out of order and read as glitches). BUSY gates every
// model-touching action; errors surface as the app's own gentle presentation, never as raw
// exception text spoken in the confidant's voice.
let BUSY=false;
// oops() stays the single entry point every model-touching action calls on failure -- only its
// PRESENTATION was upgraded (A_precision_calm's error-card pattern, generalized to every surface
// an error can land on): a quiet card with a small icon in the chat list, the sheet body, or the
// refusal panel, with the raw detail kept small underneath, instead of one shared status-line
// message. `where` picks the surface; omit it for the old generic status-line fallback.
function oops(detail,where){console.warn('model call failed:',detail);
 const msg=(detail&&detail.message)?detail.message:String(detail);
 hint.textContent='';
 if(where==='chat'){errBubble(msg);}
 else if(where==='sheet'){body.className='';
  body.innerHTML='<div class=sheet-error><span class=err-ic aria-hidden=true>!</span>'
   +'<div><b>Couldn&#8217;t build your sheet</b><span>'+esc_(msg)+'</span></div></div>';}
 else if(where==='refusal'){
  refusal.innerHTML='<div class=refusal><span class=err-ic aria-hidden=true>!</span> '+esc_(msg)+'</div>';}
 else{hint.textContent='Something went wrong on my end — nothing was lost. Please try again.';}}
// Quick-start ramps (WebMD-AISC-inspired): for the person who cannot get the first words out,
// tapping a chip sends that sentence as THEIR chosen opener. Chips vanish once talking starts.
// Provenance stamp (design review, 2026-08-01): a chip fills the composer with app-authored
// wording, not something the patient composed -- send(true) marks the resulting msgs[] entry
// `origin:'chip'` so it can be told apart from typed text all the way to the sheet (see the
// server's /sheet and /export handlers, and l.chip_origin in lineRow() below). Compare addLine(),
// which already stamps `chosen_by:'patient'` on a line the patient puts back after dropping it --
// this is the same idea, for how a line's WORDS got into the transcript in the first place, not
// who kept it on the sheet.
function startWith(b){inp.value=b.textContent;send(true);}
// Streaming /chat: the server responds with one JSON object per newline ("ndjson") as tokens
// arrive, ending in exactly one terminal line -- {done:true, reply:<full text>, ms, where} on
// success or {error} on failure. BUSY is set BEFORE the fetch and only cleared in `finally`, so
// it still gates the entire in-flight stream, not just the initial request -- an anxious
// double-tap mid-stream hits the `if(BUSY)return;` guard exactly as it did against the old
// blocking call. msgs[] only ever receives the FULL reply, and only once the terminal {done:...}
// line has actually arrived -- a reply cut short by a mid-stream error never lands there partial,
// matching the old all-or-nothing behavior of the blocking call.
// MERGE NOTE (2026-08-01): the canned starter-chip replies also come back over this same ndjson
// channel (see _handle_chat) rather than as a plain JSON body -- one wire format for /chat, so
// this reader never has to branch on which kind of reply it is getting.
async function send(fromChip){if(BUSY)return;let t=inp.value.trim();if(!t)return;inp.value='';inp.style.height='auto';
 starters.style.display='none';
 bubble(t,'you');
 let m={role:'user',content:t};if(fromChip)m.origin='chip';
 msgs.push(m);hint.innerHTML=thinkingHTML(NAME+' is thinking');
 BUSY=true;
 let bub=null, acc='', buf='';
 try{
  let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs,backend:backend.value,stream:true})});
  if(!r.ok||!r.body) throw new Error('request failed ('+r.status+')');
  const reader=r.body.getReader(), dec=new TextDecoder();
  let finished=false;
  while(true){
   const {done,value}=await reader.read();
   if(done)break;
   buf+=dec.decode(value,{stream:true});
   let lines=buf.split('\\n'); buf=lines.pop();
   for(const line of lines){
    if(!line.trim())continue;
    let ev; try{ev=JSON.parse(line);}catch(e){continue;}
    if(ev.error){throw new Error(ev.error);}
    if(ev.content){
     acc+=ev.content;
     if(!bub){hint.innerHTML='';bub=bubble(acc,'u');} else {bub.textContent=acc;}
     scrollChat();
    }
    if(ev.done){
     acc=(typeof ev.reply==='string'?ev.reply:acc).trim();
     if(!bub)bub=bubble(acc,'u'); else bub.textContent=acc;
     msgs.push({role:'assistant',content:acc});
     hint.innerHTML=badge(ev.where,ev.ms);
     finished=true;
    }
   }
  }
  if(!finished) throw new Error('the connection ended before a full reply arrived');
 }catch(e){
  // A partial reply that never reached its terminal {done:...} line (network drop, model error
  // mid-stream) must not sit in the transcript looking finished -- pull the in-progress bubble
  // back out so the only trace of this turn is the same gentle error card the old blocking path
  // showed, and msgs[] stays exactly as if the call had failed immediately.
  if(bub && bub.parentElement) bub.parentElement.remove();
  oops(e,'chat');
 }finally{BUSY=false;}}
// SIDE-BY-SIDE UX VARIANTS (2026-08-01, review pass -- Alex judges UX by eye, so both ways are
// live at once behind URL params; no rebuild to compare, judges never see the switch):
//   ?drawer=open  -> the add-back drawer starts OPEN with an inviting label ("more things you
//                    said -- add any of these"), promoting it from recovery bin to co-selection
//                    surface. Default stays the current quiet closed drawer.
//   ?screen=min   -> "Say it better for me" and "One more look" stay hidden until a sheet
//                    exists (their pre-sheet tap is a refusal by design; this variant simply
//                    doesn't offer them yet). Default keeps all five actions visible.
const QS=new URLSearchParams(location.search);
const DRAWER_B=QS.get('drawer')==='open';
if(QS.get('screen')==='min')document.documentElement.classList.add('screen-min');
// Composer focus is desktop-only: an unconditional autofocus pops the keyboard over the first
// screen on a phone -- the judge's QR path -- before they have read a word.
if(!('ontouchstart' in window)){const _ta=document.getElementById('inp');if(_ta)_ta.focus();}
let SHEET=null, SEL=-1, DROPPED=[], ADDED=[], HIST=[], SHOWMORE=DRAWER_B, PINNED=-1;
// END-ROUND: the "look back over everything" finale (Alex's own idea, three_pane/
// BRAINSTORM_ALEX_2026-07-31.md §2). ENDROUND holds the CANDIDATES the model suggested this pass
// (never shown as if they were the patient's words); EAPPROVED holds only what the patient has
// actively approved onto the sheet -- the one thing that ever reaches /export. An unapproved
// candidate lives only in ENDROUND and is never read by copySheet().
let ENDROUND=null, EAPPROVED={questions:[],dropped:[]};
// True only for the render right after "Make my sheet" — the one time the whole list should
// stagger in on top of the sheet's own always-on entrance animation (see .line/.pinned/.tier in
// <style>). Set here (not per-render) so clicking a line open, dropping one, pinning one, or
// accepting a draft re-renders the sheet WITHOUT re-staggering every other line (grafted from
// A_precision_calm; see renderSheet()'s nextDelay() for how it's wired to C's own riseIn instead
// of a separate stagger keyframe).
let SHEET_ANIM=false;
// DROPPED / ADDED: line ids the patient removed or put back, sent on export.
// PINNED: the id of the line the patient chose as "your first thing to say". Patient-chosen by
// definition — the model never ranks it. Display + print only for now: the emailed export still
// comes off the server's assemble path, which has no pin concept (known limit, logged).
// HIST: snapshots for undo. Snapshot-based rather than per-operation inverses -- delete a line,
// delete a cue and add a line all mutate different arrays, and hand-written undo for each is where
// the off-by-one bugs live. Copying a small object is cheap and cannot get out of step.
// Color as wayfinding (v2): each tier carries its own warm color + a small glyph (see .tierlab/
// .tricon in <style>) -- primary is the strongest/warmest (this is literally "what matters
// most"), context is the quietest but still warm, never a cold gray, so the three read as one
// family at different volumes rather than three unrelated sections.
const TIERS=[
 {key:'primary',label:'What matters most',color:'var(--tier-primary)',
  icon:'<svg viewBox="0 0 24 24"><circle cx=12 cy=12 r=9 fill="currentColor"/></svg>'},
 {key:'secondary',label:'Also on my mind',color:'var(--tier-secondary)',
  icon:'<svg viewBox="0 0 24 24"><circle cx=12 cy=12 r=6.5 fill="currentColor"/></svg>'},
 {key:'context',label:'Background — family history',color:'var(--tier-context)',
  icon:'<svg viewBox="0 0 24 24"><circle cx=12 cy=12 r=8 fill=none stroke=currentColor stroke-width="2.4"/></svg>'}
];
function esc_(s){let d=document.createElement('div');d.textContent=s;return d.innerHTML;}
// Mirrors gate/probe.py's ELEMENTS labels — display-only, so a mismatch here is cosmetic (the
// server's own copy is what next_question() actually returns and PROBE stores per line).
const PROBE_LABELS={onset:'Onset',location:'Location',quality:'Quality',severity:'Severity',
 duration:'Duration',timing:'Timing / frequency',modifying:'What changes it',
 associated:'Associated symptoms',context:'Context'};
function probeLabel(key){return PROBE_LABELS[key]||key||'';}

// ACCOMMODATIONS -- "how I need this visit to go", the patient's own statement, independent of
// the conversation and of any sheet build. NO MODEL IS EVER CALLED FOR THIS: the only two sources
// are a fixed curated list (below -- selecting one is CHOOSING, which this codebase already
// treats as authoring; see chosen_by:"patient" in gate/pipeline.py and the starter-chip comment
// on startWith() above) and the patient's own typing. Cite-and-extend of the AASPIRE Healthcare
// Toolkit's accommodations report (Nicolaidis et al. 2016): a similar report cut clinician-
// reported communication barriers 4.07->2.82 (p<0.0001, n=259 patients + 51 PCPs) and was read by
// ~97% of clinicians -- the best-evidenced idea available here, so the wording below extends that
// precedent rather than inventing new claims. Curated, not model-authored, so a human can judge
// every word of it; see BUILD_REPORT.md for the rationale on each one.
// Grouped into FAMILIES with progressive disclosure (Alex, 2026-08-01 00:58): a flat list of
// eighteen asks a person with executive-function difficulty to scan eighteen options and choose,
// which is the exact load this block exists to remove. Five doors, then a handful behind whichever
// one you open. Family names are in the patient's register, not clinical categories -- you should
// recognise your own situation in the door rather than translate.
//
// [H] items were brainstormed by a human in this population (2026-08-01) when asked for examples
// of things a patient might want a doctor to know. They are NOT testimony about her own needs and
// must never be presented as such; they are simply better candidates than a framework
// extrapolation. Their wording and punctuation are kept EXACTLY as written -- including the
// inconsistent trailing periods -- because the register is right and it was arrived at without us.
// [W] items were written from the AASPIRE categories. Neither set has been validated by a real
// patient using it.
const ACCOM_FAMILIES=[
 {h:"How I take things in", items:[
  "I process info at slightly slower pace so please pause so I can keep up.",   // [H]
  "English is not my first language",                                          // [H]
  "Please use plain language, not medical jargon.",
  "Write things down for me — I won’t retain it if it’s only spoken.",
  "I may need you to repeat or rephrase something more than once.",
 ]},
 {h:"How I say things", items:[
  "I hear well, but have trouble replying.",                                    // [H]
  "I get anxious at appointments and forget my thoughts.",                      // [H]
  "If I go quiet, I’m still working on the answer — please wait rather than moving on.",
  "Please ask me one question at a time.",
  "Please don’t rush me — I need extra time to answer.",
 ]},
 {h:"What appointments do to me", items:[
  "Medical appointments make my blood pressure higher",                         // [H]
  "I mask when I’m anxious, so I may look fine when I’m not.",
  "I may go quiet or shut down if I’m overwhelmed — that doesn’t mean I’m fine.",
  "I have a hard time with eye contact — it doesn’t mean I’m not listening.",
 ]},
 {h:"The room and my body", items:[
  "Please tell me what you’re about to do before you touch me.",
  "Bright lights or loud sounds are hard for me — please keep things low-key if you can.",
 ]},
 {h:"What helps me", items:[
  "I’d like to have a support person with me during the visit.",
  "I may need a minute partway through.",
  "Please check I’ve understood before we move on.",
 ]},
];
// Flattened once so optIdx (and therefore every stored ACCOM entry) keeps meaning a single stable
// index, unchanged from the flat-list version. FAM_OF maps that index back to its family for
// rendering; nothing downstream needs to know families exist.
const ACCOM_OPTIONS=[], ACCOM_FAM_OF=[];
ACCOM_FAMILIES.forEach(function(f,fi){f.items.forEach(function(s){
 ACCOM_FAM_OF.push(fi); ACCOM_OPTIONS.push(s);});});
// {id, text, source:'curated'|'typed', optIdx?}. Deletable/editable on the same terms as every
// other line on the sheet -- re-clicking a chosen chip removes it; each item has its own remove
// button. Persists across "Make my sheet" (it is not conversation output) but is cleared by
// "Start over", same as every other piece of session state.
let ACCOM=[], ACCOM_SEQ=1000;
function accomSelected(){return new Set(ACCOM.filter(a=>a.source==='curated').map(a=>a.optIdx));}
function toggleAccomOption(idx){
 const at=ACCOM.findIndex(a=>a.source==='curated'&&a.optIdx===idx);
 if(at>=0){ACCOM.splice(at,1);}
 else{ACCOM.push({id:ACCOM_SEQ++,text:ACCOM_OPTIONS[idx],source:'curated',optIdx:idx});}
 renderAccom();}
function addAccomCustom(){const t=(accomText?accomText.value:'').trim();if(!t)return;
 ACCOM.push({id:ACCOM_SEQ++,text:t,source:'typed'});renderAccom();
 if(accomText)accomText.focus();}
function removeAccom(id){ACCOM=ACCOM.filter(a=>a.id!==id);renderAccom();}
function renderAccom(){
 const sel=accomSelected();
 let h='<span class=accom-title>How I need this visit to go</span>'
   +'<span class=accom-sub>My own words, shown first — add or remove anytime.</span>';
 if(ACCOM.length){
  h+='<div class=accom-items>';
  for(const a of ACCOM){
   h+='<div class=accom-item><span class=txt>'+esc_(a.text)+'</span>'
     +(a.source==='curated'?'<span class=mine>chosen by me</span>':'')
     +'<button class=quiet-remove onclick="removeAccom('+a.id+')">remove</button></div>';
  }
  h+='</div>';
 }
 h+='<details class=accom-add'+(ACCOM.length?'':' open')+'>'
   +'<summary><span class=chev></span> Add how I need this to go</summary>'
   +'<div class=accom-picker>';
 let idx=0;
 for(let fi=0;fi<ACCOM_FAMILIES.length;fi++){
  const fam=ACCOM_FAMILIES[fi];
  const n=fam.items.filter(function(_,k){return sel.has(idx+k);}).length;
  h+='<details class=accom-fam'+(n?' open':'')+'><summary><span class=chev></span>'
    +esc_(fam.h)+(n?' <span class=accom-count>'+n+'</span>':'')+'</summary><div class=accom-fam-in>';
  for(let k=0;k<fam.items.length;k++,idx++){
   h+='<button type=button class=accom-chip aria-pressed="'+(sel.has(idx)?'true':'false')+'" '
     +'onclick="toggleAccomOption('+idx+')">'+esc_(ACCOM_OPTIONS[idx])+'</button>';
  }
  h+='</div></details>';
 }
 h+='</div><div class=accom-custom>'
   +'<textarea id=accomText rows=2 placeholder="Or write your own, in your own words…" '
   +'onkeydown="if(event.key===&#39;Enter&#39;&&!event.shiftKey){event.preventDefault();addAccomCustom();}"></textarea>'
   +'<button class=btn-ghost type=button onclick=addAccomCustom()>Add</button></div></details>';
 accom.innerHTML=h;
 accom.className='accom'+(ACCOM.length?'':' is-empty');}

// MY WEEK -- the symptom diary (added 2026-08-01, from a family design conversation). Same provenance class as the
// accommodations block above it: nothing here is extracted or modeled. Every entry is a number
// the patient CHOSE (0-10) and words they typed, so it is patient-authored by construction and
// carries no receipt because there is nothing to verify it against. Day-by-day severity is the
// range question a clinician actually asks ("how bad does it get? what's a good day like?"),
// answered in the patient's own record. Renders with the .accom classes so screen, print and
// page-break treatment are identical.
let DIARY=[], DIARY_SEQ=5000;
function addDiaryEntry(){
 const d=(diaryDay?diaryDay.value:'').trim();
 const sc=diaryScore?diaryScore.value:'';
 const n=(diaryNote?diaryNote.value:'').trim();
 if(!d&&!n&&sc==='')return;
 let t=d;
 if(sc!=='')t+=(t?' — ':'')+sc+'/10';
 if(n)t+=(t?' — ':'')+n;
 DIARY.push({id:DIARY_SEQ++,text:t});renderDiary();
 const dd=document.getElementById('diaryDay');if(dd){dd.value='';dd.focus();}}
function removeDiary(id){DIARY=DIARY.filter(x=>x.id!==id);renderDiary();}
function renderDiary(){
 let h='<span class=accom-title>My week — day by day</span>'
   +'<span class=accom-sub>My own numbers, 0–10 — I add and remove these myself.</span>';
 if(DIARY.length){
  h+='<div class=accom-items>';
  for(const e of DIARY){
   h+='<div class=accom-item><span class=txt>'+esc_(e.text)+'</span>'
     +'<button class=quiet-remove onclick="removeDiary('+e.id+')">remove</button></div>';
  }
  h+='</div>';
 }
 let opts='<option value="">0–10</option>';
 for(let i=0;i<=10;i++)opts+='<option>'+i+'</option>';
 h+='<details class=accom-add'+(DIARY.length?'':' open')+'>'
   +'<summary><span class=chev></span> Add a day</summary>'
   +'<div class=accom-custom>'
   +'<input id=diaryDay class=diary-in placeholder="Day (Wed, Jul 30…)" maxlength=24>'
   +'<select id=diaryScore class=diary-in>'+opts+'</select>'
   +'<input id=diaryNote class=diary-in placeholder="What happened, in my words" maxlength=120 '
   +'onkeydown="if(event.key===&#39;Enter&#39;){event.preventDefault();addDiaryEntry();}">'
   +'<button class=btn-ghost type=button onclick=addDiaryEntry()>Add</button></div></details>';
 diary.innerHTML=h;
 diary.className='accom'+(DIARY.length?'':' is-empty');}

async function makeSheet(){if(BUSY)return;hint.innerHTML=thinkingHTML('Putting your words in order');refusal.innerHTML='';
 BUSY=true;
 try{
  let r=await fetch('/sheet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs,backend:backend.value,think:deep.checked,accommodations:ACCOM.map(a=>a.text)})});
  let j=await r.json();if(j.error){oops(j.error,'sheet');return;}
  hint.innerHTML=badge(j.where,j.ms);where.innerHTML=badge(j.where,j.ms);
  SHEET=j;SEL=-1;DROPPED=[];ADDED=[];HIST=[];SHOWMORE=DRAWER_B;PINNED=-1;POLISH={};PROBE={};SHEET_ANIM=true;
  document.documentElement.classList.add('has-sheet');
  ENDROUND=null;EAPPROVED={questions:[],dropped:[]};renderSheet();
  if(window.innerWidth<=700)showTab(true);
 }catch(e){oops(e,'sheet');}finally{BUSY=false;}}

// Every line is one of the patient's own segmented clauses, carrying the character offsets the
// gate verified. Deleting one removes an entry from a list -- the ID architecture IS the edit
// model, so this needs no server round-trip and cannot alter anyone's wording.
function lineRow(l,i){
 return '<div class=row><span class=txt>'
   +(l.probe_of!=null?'<span class=probe-tag>'+esc_(probeLabel(l.probe_element))+'</span>':'')
   +esc_(l.display)
   +(l.polished?' <span class=helped>said with help — approved by me</span>':'')
   +(l.chosen_by==='patient'?' <span class=mine>you added this</span>':'')
   +(l.chip_origin?' <span class=mine>from a suggested opener, not typed</span>':'')+'</span>'
   +'<button class=chip title="chars '+l.start+'–'+l.end+' of your words" '
   +'onclick="event.stopPropagation();showSource('+i+')"><span class=chip-dot aria-hidden=true></span>'
   +(i===SEL?'hide':'source')+'</button>'
   +'<button class=pinbtn title="Make this my first thing to say" '
   +'onclick="event.stopPropagation();pinLine('+l.id+')">'+(l.id===PINNED?'unpin':'pin')+'</button></div>';}
function srcPanel(l){const t=SHEET.transcript;
 return '<div class=source-panel><span class=source-label>Where this came from — your own words</span>'
   +esc_(t.slice(0,l.start))+'<mark>'+esc_(t.slice(l.start,l.end))+'</mark>'+esc_(t.slice(l.end))+'</div>';}
function renderSheet(){
 if(!SHEET){return;}
 body.className='';
 let h='';
 // First render right after "Make my sheet" staggers each line's entrance delay on top of the
 // sheet's own always-on riseIn animation (see .line/.pinned/.tier in <style>) — grafted from
 // A_precision_calm, wired to C_quiet_editorial's existing per-render entrance motion instead of a
 // separate stagger keyframe: nextDelay() is a no-op (always 0ms) on every render except the one
 // right after makeSheet(), so later re-renders (open a line, drop one, pin, undo, accept a draft)
 // keep C's original behavior of the remaining lines rising in together.
 let animIdx=0;
 const nextDelay=()=>SHEET_ANIM?Math.min(animIdx++,14)*22:0;
 const pi=SHEET.lines.findIndex(l=>l.id===PINNED);
 if(pi>=0){const l=SHEET.lines[pi];
  // "open" class carried here too (pinned isn't a .line, so it needs its own hook) so the
  // highlighter-family chip treatment (.pinned.open .chip, see <style>) also lights up when the
  // pinned line's own receipt is the one showing -- not just for lines inside a tier.
  h+='<div class="pinned'+(SEL===pi?' open':'')+'" style="animation-delay:'+nextDelay()+'ms"><svg class=pin-icon viewBox="0 0 24 24" aria-hidden=true><path fill=currentColor d="M12 2c-4.4 0-8 3.6-8 8 0 5.4 7 12 8 12s8-6.6 8-12c0-4.4-3.6-8-8-8zm0 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/></svg>'
    +'<div class=pin-body><div class=pin-label>Your first thing to say</div>'+lineRow(l,pi);
  if(SEL===pi)h+=srcPanel(l)
    +'<div class=line-tools><button class=quiet-more onclick="event.stopPropagation();askMore('+l.id+')">Ask me more about this</button></div>';
  h+=draftCard(l);
  h+=probeCard(l);
  h+='</div></div>';}
 for(const T of TIERS){
  const rows=SHEET.lines.filter(l=>l.tier===T.key&&l.id!==PINNED);
  if(!rows.length) continue;
  h+='<div class=tier style="--tier-color:'+T.color+'"><div class=tierlab>'
    +'<span class=tlrow><span class=tricon aria-hidden=true>'+T.icon+'</span>'+T.label+'</span></div>';
  for(const l of rows){
   const i=SHEET.lines.indexOf(l);
   h+='<div class="line'+(i===SEL?' open':'')+'" style="animation-delay:'+nextDelay()+'ms" onclick="showSource('+i+')">'+lineRow(l,i);
   if(i===SEL){h+=srcPanel(l)
     +'<div class=line-tools><button class=quiet-remove onclick="event.stopPropagation();dropLine('+i+')">Remove from sheet</button>'
     +'<button class=quiet-more onclick="event.stopPropagation();askMore('+l.id+')">Ask me more about this</button></div>';}
   h+=draftCard(l);
   h+=probeCard(l);
   h+='</div>';
  }
  h+='</div>';
 }
 if(!SHEET.lines.length) h+='<div class=placeholder>Every line has been removed.</div>';

 // Cues are the patient's OWN sentences, so they are deletable on exactly the same terms as any
 // other line. "The patient has full control" (Alex, 2026-07-27) does not survive a section that
 // is their words but cannot be removed.
 if(SHEET.cues&&SHEET.cues.length){
  h+='<div class=safety-lab><span>Things I mentioned</span><span class=locked>my words — remove any of these too</span></div>';
  h+='<div class=tier>';
  for(const c of SHEET.cues){
   h+='<div class=line style="cursor:default;animation-delay:'+nextDelay()+'ms"><div class=row><span class=txt>'+esc_(c.text)+'</span>'
     +'<button class=quiet-remove title="Remove this line" onclick="dropCue('+c.id+')">remove</button></div></div>';
  }
  h+='</div>';
 }
 h+=drawerHtml();
 h+=endRoundHtml();

 // The safety block keeps no delete control -- not as an exception to the patient's control, but
 // because it is not their text. It is generic, identical for everyone whose transcript touches the
 // same topic, and says nothing about them. Removing it hides nothing they said.
 h+='<div class=safety-lab><span>Safety information</span><span class=locked>included every time · not about you</span></div>';
 h+='<div class=safety>'+esc_(SHEET.safety)+'</div>';
 const s=SHEET.stats;
 h+='<div class=stats>'+s.n_selected+' of '+s.n_candidates+' pieces chosen · '
   +s.n_verified+'/'+s.n_selected+' verified against your recording</div>';
 h+='<div class=sheetacts><button class=primary onclick=printSheet()>Print / save as PDF</button>'
   +'<button class=secondary onclick=copySheet()>Copy for email</button>'
   +(TTS_AVAILABLE?'<button class=secondary onclick=readAloud()>Read my sheet aloud</button>':'')
   +(HIST.length?'<button class=secondary onclick=undo()>Undo</button>':'')
   +'<span class=copied id=copied></span></div>';
 // Read-aloud player: hidden until there is something to play, so it never sits on the page as
 // dead chrome for the (default) STT-unavailable install or before the button is ever pressed.
 if(TTS_AVAILABLE){
  h+='<div class=speak-wrap><audio id=sheetaudio controls style="width:100%;display:none"></audio>'
    +'<span class=copied id=speakStatus></span></div>';
 }
 body.innerHTML=h;
 // The printed page's provenance line derives from THIS sheet, never asserted statically: the
 // paper claims on-device processing only when the sheet actually came back where=device (the
 // old static line claimed it even for hosted/cloud sheets -- review find 2026-08-01). Same
 // spot carries the verified count and, only when brackets are actually on the sheet, the
 // one-line bracket legend -- the whole trust story, on the artifact a doctor holds.
 const pv=document.getElementById('printver');
 if(pv){let t=(s.n_selected?s.n_verified+' of '+s.n_selected+' lines checked word-for-word against my recording. ':'');
  t+=(SHEET.where==='device')?'Processed on my own device — my words never left it.':'Processed by the hosted preview of this app.';
  if(SHEET.lines.some(l=>((l.display||l.text)||'').indexOf('[')>=0))
   t+=' Plain text is word-for-word mine; anything in [brackets] was added by the app as a question to check, not as my claim.';
  pv.textContent=t;}
 SHEET_ANIM=false;}

// The set-aside drawer replaces the old "show N" list: removed lines stay visibly recoverable —
// "still yours, just not shared." For someone anxious about control, seeing where a removed line
// WENT is what makes removing one feel safe.
function drawerHtml(){
 const u=SHEET.unselected||[];
 if(!u.length) return '';
 let h='<details class=drawer'+(SHOWMORE?' open':'')+' ontoggle="SHOWMORE=this.open">'
   +'<summary><span class=chev></span> '+(DRAWER_B
     ?u.length+' more thing'+(u.length>1?'s':'')+' you said — add any of these to your sheet.'
     :u.length+' item'+(u.length>1?'s':'')+' set aside — still yours, just not shared.')
   +'</summary><div class=drawer-body>';
 for(const c of u){
  h+='<div class=row><span class=txt>'+esc_(c.display)+'</span>'
    +'<button class=addback onclick="addLine('+c.id+')">put back on sheet</button></div>';
 }
 h+='</div></details>';
 return h;}

function printSheet(){window.print();}

// Export asks the SERVER to rebuild the sheet from the transcript plus the ids still on screen.
// Rendering it here from the DOM would be quicker and would be the wrong thing: the exported
// artifact must come off the same code path as the on-screen one, and must reflect what is left
// after the patient's deletions rather than what was generated before them.
async function copySheet(){copied.textContent='';
 // chip_spans rides along so a re-derived export still knows which transcript ranges came from a
 // starter chip -- see the /sheet handler, which computes it once from msgs[].origin and hands it
 // back on the sheet object precisely so this round trip is possible.
 const r=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({transcript:SHEET.transcript,
                        ids:SHEET.lines.map(l=>l.id),
                        dropped:DROPPED,
                        added:ADDED,
                        keep_cues:SHEET.cues.map(c=>c.id),
                        chip_spans:SHEET.chip_spans||[],
                        accommodations:ACCOM.map(a=>a.text),
                        diary:DIARY.map(d=>d.text),
                        // Only what the patient APPROVED, never the full candidate list — see
                        // ENDROUND/EAPPROVED note above. The server re-verifies each item again
                        // before it can land in the printed/emailed letter (endround.verify_export_item).
                        endround:{questions:EAPPROVED.questions.map(q=>({text:q.text})),
                                  dropped:EAPPROVED.dropped.map(d=>({quote:d.quote,prompt:d.prompt}))}})});
 const j=await r.json();
 try{await navigator.clipboard.writeText(j.text);copied.textContent='Copied — paste it into an email.';}
 catch(e){copied.textContent='Copy blocked by the browser; the text is in the console.';
   console.log(j.text);}}

// "Read my sheet aloud" (deferred item, ~30min estimate). Same re-derive-on-the-server
// discipline as copySheet(): sends the transcript plus the ids/drops/adds/kept-cues still on
// screen, so what gets READ is what the patient currently has -- after their deletions -- not a
// stale version. The server (gate/speech.py, via Windows SAPI/System.Speech) reads section
// headings and safety information in one voice and the patient's own words in another, so the
// sheet's print-time [bracket] convention ("this is Understudy's, not yours") survives as an
// audible distinction instead of a typographic one. Nothing here changes the sheet; this is
// read-only, like Print or Copy.
let SPEAK_BUSY=false;
async function readAloud(){
 if(SPEAK_BUSY||!SHEET)return;
 SPEAK_BUSY=true;
 const st=document.getElementById('speakStatus'), au=document.getElementById('sheetaudio');
 st.innerHTML=thinkingHTML('Recording your sheet as speech');
 try{
  const r=await fetch('/speak',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({transcript:SHEET.transcript,
                         ids:SHEET.lines.map(l=>l.id),
                         dropped:DROPPED,
                         added:ADDED,
                         keep_cues:SHEET.cues.map(c=>c.id)})});
  const ct=(r.headers.get('Content-Type')||'');
  if(!r.ok || ct.indexOf('audio')<0){
   let msg='Something went wrong turning your sheet into speech.';
   try{const j=await r.json();if(j.error)msg=j.error;}catch(e){}
   st.textContent='['+msg+']';return;
  }
  const blob=await r.blob();
  if(au.dataset.url)URL.revokeObjectURL(au.dataset.url);
  const url=URL.createObjectURL(blob);
  au.src=url;au.dataset.url=url;au.style.display='block';st.textContent='';
  au.play().catch(()=>{st.textContent='Audio ready — press play.';});
 }catch(e){st.textContent='[Could not reach the read-aloud service.]';}
 finally{SPEAK_BUSY=false;}}

// UNDO. Every mutation snapshots first. Deletion being reversible is not a convenience: for someone
// anxious about control, an irreversible action is the thing that stops you using it at all. Making
// it safe to remove a line is what makes the control real rather than nominal.
function snap(){HIST.push(JSON.stringify({lines:SHEET.lines,cues:SHEET.cues,uns:SHEET.unselected,
  dropped:DROPPED,added:ADDED,sel:SEL,pinned:PINNED}));if(HIST.length>50)HIST.shift();}
function undo(){if(!HIST.length)return;const s=JSON.parse(HIST.pop());
 SHEET.lines=s.lines;SHEET.cues=s.cues;SHEET.unselected=s.uns;
 DROPPED=s.dropped;ADDED=s.added;SEL=s.sel;PINNED=(s.pinned===undefined?-1:s.pinned);
 renderSheet();highlightBubble(SEL);}

// Deleting a line REMOVES it. It used to move it: the cue block re-surfaced the same sentence
// under another heading once it was no longer represented among the lines. The dropped id is
// remembered and sent with every export so the server suppresses its cue too.
function dropLine(i){const l=SHEET.lines[i];if(!l)return;snap();
 DROPPED.push(l.id);ADDED=ADDED.filter(x=>x!==l.id);
 if(l.id===PINNED)PINNED=-1;
 SHEET.lines.splice(i,1);if(SEL===i){SEL=-1;}else if(SEL>i){SEL--;}
 SHEET.cues=SHEET.cues.filter(c=>!sameWords(c.text,l.text));
 SHEET.unselected.push({id:l.id,text:l.text,display:l.display,start:l.start,end:l.end});
 SHEET.unselected.sort((a,b)=>a.start-b.start);
 renderSheet();highlightBubble(SEL);}

function dropCue(id){snap();SHEET.cues=SHEET.cues.filter(c=>c.id!==id);renderSheet();}

// The pin is the patient's call and only the patient's: one click promotes their own top concern
// to a locked first position; the model is never asked which line matters most.
function pinLine(id){snap();PINNED=(PINNED===id?-1:id);renderSheet();}

// Putting a line back. Same verified words, but chosen by the person they belong to -- so it is
// marked patient-chosen, which is stronger provenance than model-chosen, not weaker.
function addLine(id){const u=SHEET.unselected.find(x=>x.id===id);if(!u)return;snap();
 SHEET.unselected=SHEET.unselected.filter(x=>x.id!==id);
 DROPPED=DROPPED.filter(x=>x!==id);if(ADDED.indexOf(id)<0)ADDED.push(id);
 SHEET.lines.push({id:u.id,text:u.text,display:u.display,tier:'secondary',
   start:u.start,end:u.end,source_text:u.text,verified:true,chosen_by:'patient'});
 renderSheet();}

// Loose match, mirroring pipeline._norm: lowercase, collapse whitespace, drop punctuation, then
// compare in both directions because a cue sentence often CONTAINS a selected clause.
function sameWords(a,b){
 const n=s=>s.toLowerCase().replace(/[^a-z0-9 ]/g,'').replace(/\\s+/g,' ').trim();
 const x=n(a),y=n(b);
 return !!x&&!!y&&(x.indexOf(y)>=0||y.indexOf(x)>=0);}

// The receipt, made visible in BOTH places: the source panel inside the sheet, and the exact
// characters lit up in the original chat bubble on the left. Clicking the same line again closes
// both, so nothing is ever stuck open.
function showSource(i){SEL=(SEL===i?-1:i);renderSheet();
 const m=body.querySelector('.source-panel mark');
 if(m)m.scrollIntoView({block:'center',behavior:reduceMotion()?'auto':'smooth'});
 highlightBubble(SEL);}

// Map a line's transcript offsets back to the patient bubble it came from. The transcript is the
// server's join of the patient turns with a newline (understudy_app.py /sheet), and YOUB holds the
// bubble <p> nodes in the same order — so walk the turns, find the one containing the span, and
// mark the local slice. Guards: if the bubble list and transcript disagree (stale state), do
// nothing rather than mark the wrong words — a wrong receipt is worse than no receipt.
function clearMarks(){for(const p of YOUB){if(p.dataset.raw!==undefined)p.textContent=p.dataset.raw;}}
function highlightBubble(sel){clearMarks();
 if(sel<0||!SHEET||!SHEET.lines[sel])return;
 const l=SHEET.lines[sel];
 const turns=SHEET.transcript.split('\\n');
 let off=0;
 for(let i=0;i<turns.length;i++){
  const end=off+turns[i].length;
  if(l.start>=off&&l.start<end){
   const p=YOUB[i];
   if(!p||p.dataset.raw!==turns[i])return;
   const ls=l.start-off, le=Math.min(l.end-off,turns[i].length);
   p.innerHTML=esc_(turns[i].slice(0,ls))+'<mark>'+esc_(turns[i].slice(ls,le))+'</mark>'+esc_(turns[i].slice(le));
   p.scrollIntoView({block:'nearest',behavior:reduceMotion()?'auto':'smooth'});
   return;
  }
  off=end+1;
 }}

// "Say it better for me" — the delegation path (was a hardcoded refusal until 07-30; Alex's
// pluralism ruling: some people need help getting words out, not maximal control). The guarantee
// transforms rather than breaks: the model still cannot put words in your mouth SILENTLY. It
// offers a labeled draft per line; you approve or reject each one; an accepted draft is tagged
// on the sheet and its receipt still opens your verbatim original. Server-side deterministic
// gates suppress any draft that introduces a clinical word you did not use.
let POLISH={};
async function betterWords(){if(BUSY)return;
 if(!SHEET||!SHEET.lines.length){
  refusal.innerHTML='<div class=refusal>Make your sheet first — then I can offer a clearer '
   +'wording for each line. You approve every word before anything changes.</div>';return;}
 refusal.innerHTML='';
 const targets=SHEET.lines.filter(l=>!l.polished).map(l=>({id:l.id,text:l.text}));
 if(!targets.length){hint.textContent='Every line already has wording you approved.';return;}
 hint.innerHTML=thinkingHTML('Drafting clearer wording — nothing changes unless you approve it');
 BUSY=true;
 try{
  const r=await fetch('/polish',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({lines:targets,backend:backend.value,think:deep.checked})});
  const j=await r.json();
  if(j.error){oops(j.error,'refusal');return;}
  hint.innerHTML=badge(j.where,j.ms);
  POLISH={};
  for(const d of (j.drafts||[])){
   if(d.error)continue;
   POLISH[d.id]={draft:d.draft||null,cautions:d.cautions||[]};
  }
  renderSheet();
  if(window.innerWidth<=700)showTab(true);
 }catch(e){oops(e,'refusal');}finally{BUSY=false;}}

function draftCard(l){const p=POLISH[l.id];if(!p)return '';
 if(!p.draft){return '<div class=draft-card><span class=draft-label>No safe draft for this line</span>'
  +'<div class=draft-cautions>'+esc_(p.cautions.join(' · '))+'</div></div>';}
 let h='<div class=draft-card><span class=draft-label>Clearer wording — yours only if you approve it</span>'
  +'<div class=draft-text>'+esc_(p.draft)+'</div>';
 if(p.cautions.length)h+='<div class=draft-cautions>&#9888; '+esc_(p.cautions.join(' · '))+'</div>';
 h+='<div class=draft-acts>'
  +'<button class=draft-accept onclick="event.stopPropagation();acceptDraft('+l.id+')">Use this wording</button>'
  +'<button class=draft-keep onclick="event.stopPropagation();rejectDraft('+l.id+')">Keep my words</button>'
  +'</div></div>';
 return h;}
function acceptDraft(id){const l=SHEET.lines.find(x=>x.id===id);const p=POLISH[id];
 if(!l||!p||!p.draft)return;snap();
 l.polished=true;l.display=p.draft;delete POLISH[id];renderSheet();}
function rejectDraft(id){delete POLISH[id];renderSheet();}

// "Ask me more about this" — the patient clicks a line; the app asks ONE deepening question
// shaped by the standard clinical history-taking elements (see gate/probe.py), sibling of
// POLISH{} above: a side-map keyed by line id, rendered as a card under the line. Unlike POLISH
// this never calls the model for its own output — the question is templated (see probe.py for
// why) and landing the answer is pure code (see pipeline.append_answer) — so the whole feature
// costs zero extra Ollama calls.
// PROBE[lineId] = {asked:[element keys already asked for this line], current:{element,label,
// question}|null, done:bool (every element asked or already evident in the line's own words)}.
let PROBE={};
async function askMore(id){
 if(BUSY)return;
 const l=SHEET.lines.find(x=>x.id===id); if(!l)return;
 const st=PROBE[id]||(PROBE[id]={asked:[],current:null,done:false});
 refusal.innerHTML='';
 BUSY=true;
 try{
  const r=await fetch('/probe_question',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:l.text,asked:st.asked})});
  const j=await r.json();
  if(j.error){oops(j.error,'refusal');return;}
  if(j.question){st.current=j.question;st.done=false;}
  else{st.current=null;st.done=true;}
  renderSheet();
 }catch(e){oops(e,'refusal');}finally{BUSY=false;}}

// Skipping moves to the next element without an answer — same "if they say they do not know,
// move to a different concrete detail" discipline the chat's own system prompt already follows.
function skipProbe(id){const st=PROBE[id];if(!st||!st.current)return;
 st.asked.push(st.current.element);st.current=null;renderSheet();}

// The answer is appended to the transcript and re-segmented server-side (pipeline.append_answer)
// — no model call, and every existing line's id is provably unaffected (append-only growth; see
// BUILD_REPORT.md). It also becomes a real "you" bubble in the conversation (bubble()+msgs.push,
// same call a typed chat message makes) so the receipt-highlight-in-chat and a later "Make my
// sheet" rebuild both see it exactly like anything else the patient said.
async function submitProbe(id){
 if(BUSY)return;
 const st=PROBE[id];if(!st||!st.current)return;
 const ta=document.getElementById('probe-ta-'+id);
 const ans=((ta&&ta.value)||'').trim();
 if(!ans){if(ta)ta.focus();return;}
 BUSY=true;
 try{
  const r=await fetch('/probe_answer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({transcript:SHEET.transcript,answer:ans,parent_id:id,element:st.current.element})});
  const j=await r.json();
  if(j.error){oops(j.error,'refusal');return;}
  snap();
  SHEET.transcript=j.transcript;
  for(const ln of (j.lines||[])){
   SHEET.lines.push(ln);
   if(ADDED.indexOf(ln.id)<0)ADDED.push(ln.id);
   SHEET.stats.n_selected++;SHEET.stats.n_candidates++;
   if(ln.verified)SHEET.stats.n_verified++;
  }
  bubble(ans,'you');msgs.push({role:'user',content:ans});
  st.asked.push(st.current.element);
  st.current=null;
  refusal.innerHTML='';
  renderSheet();
 }catch(e){oops(e,'refusal');}finally{BUSY=false;}}

function probeCard(l){const st=PROBE[l.id];if(!st)return '';
 if(st.done&&!st.current)return '<div class=draft-card><span class=draft-label>Ask me more</span>'
  +'<div class=draft-cautions>You have covered the standard details for this one.</div></div>';
 if(!st.current)return '';
 const q=st.current;
 return '<div class=draft-card><span class=draft-label>Ask me more — '+esc_(q.label)+'</span>'
  +'<div class=draft-text>'+esc_(q.question)+'</div>'
  +'<textarea id="probe-ta-'+l.id+'" class=probe-ta rows=2 placeholder="Type your answer…" '
  +'onclick="event.stopPropagation()"></textarea>'
  +'<div class=draft-acts>'
  +'<button class=draft-accept onclick="event.stopPropagation();submitProbe('+l.id+')">Add to my sheet</button>'
  +'<button class=draft-keep onclick="event.stopPropagation();skipProbe('+l.id+')">Skip this one</button>'
  +'</div></div>';}

// END-ROUND — "look back over everything". One deliberate pass, deep reasoning ON, over the
// WHOLE conversation (both sides — the model needs the flow to see what got dropped), after the
// sheet already exists. Reuses the exact approve-or-reject shape "say it better" uses (a labelled
// card per candidate, Approve/Not-now, nothing changes until the patient acts) rather than
// inventing a new interaction — see draftCard()/POLISH{} above.
//
// THE INVARIANT: a candidate in ENDROUND is never patient words and is never shown as if it were.
// It only ever reaches the sheet-shaped display (and /export) after landing in EAPPROVED, one
// item at a time, by the patient's own click — never silently, never in bulk.
async function runEndRound(){if(BUSY)return;
 if(!SHEET||!SHEET.lines.length){
  refusal.innerHTML='<div class=refusal>Make your sheet first — then I can read back over the '
   +'whole conversation for questions and things we didn\\'t get back to.</div>';return;}
 refusal.innerHTML='';
 hint.innerHTML=thinkingHTML('Taking one more look — reading everything back over (deep reasoning, this takes longer)');
 BUSY=true;
 try{
  const r=await fetch('/endround',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({messages:msgs,backend:backend.value})});
  const j=await r.json();
  if(j.error){oops(j.error,'refusal');return;}
  hint.innerHTML=badge(j.where,j.ms);
  ENDROUND={questions:j.questions||[],dropped:j.dropped||[]};
  renderSheet();
  if(window.innerWidth<=700)showTab(true);
 }catch(e){oops(e,'refusal');}finally{BUSY=false;}}

function eqCard(q){
 return '<div class=draft-card><span class=draft-label>Suggested by Understudy — not your words, yours to approve</span>'
  +'<div class=draft-text>'+esc_(q.text)+'</div>'
  +'<div class=draft-acts>'
  +'<button class=draft-accept onclick="approveQuestion(\\''+q.id+'\\')">Add to my sheet</button>'
  +'<button class=draft-keep onclick="skipQuestion(\\''+q.id+'\\')">Not now</button>'
  +'</div></div>';}
function edCard(d){
 return '<div class=draft-card><span class=draft-label>Something you mentioned, in your own words — never followed up</span>'
  +'<div class=source-panel style="margin-left:0"><span class=source-label>You said</span>'+esc_(d.quote)+'</div>'
  +'<div class=draft-text style="margin-top:var(--sp-2)">'+esc_(d.prompt)+'</div>'
  +'<div class=draft-cautions>Suggested by Understudy, not your words — approve to add it to your sheet</div>'
  +'<div class=draft-acts>'
  +'<button class=draft-accept onclick="approveDropped(\\''+d.id+'\\')">Add to my sheet</button>'
  +'<button class=draft-keep onclick="skipDropped(\\''+d.id+'\\')">Not now</button>'
  +'</div></div>';}

function approveQuestion(id){const q=(ENDROUND.questions||[]).find(x=>x.id===id);if(!q)return;
 ENDROUND.questions=ENDROUND.questions.filter(x=>x.id!==id);
 EAPPROVED.questions.push(q);renderSheet();}
function skipQuestion(id){ENDROUND.questions=(ENDROUND.questions||[]).filter(x=>x.id!==id);renderSheet();}
function removeApprovedQuestion(id){EAPPROVED.questions=EAPPROVED.questions.filter(x=>x.id!==id);renderSheet();}
function approveDropped(id){const d=(ENDROUND.dropped||[]).find(x=>x.id===id);if(!d)return;
 ENDROUND.dropped=ENDROUND.dropped.filter(x=>x.id!==id);
 EAPPROVED.dropped.push(d);renderSheet();}
function skipDropped(id){ENDROUND.dropped=(ENDROUND.dropped||[]).filter(x=>x.id!==id);renderSheet();}
function removeApprovedDropped(id){EAPPROVED.dropped=EAPPROVED.dropped.filter(x=>x.id!==id);renderSheet();}

// The section itself: pending candidates (approve/reject cards) above, an approved mini-tier
// below — same "safety-lab" locked-badge convention the safety block uses for "this is app
// content, not yours" (see .safety-lab/.locked in <style>), so a reader learns the same visual
// cue means the same thing everywhere on the page: NOT patient words.
function endRoundHtml(){
 const hasPending=ENDROUND&&((ENDROUND.questions||[]).length||(ENDROUND.dropped||[]).length);
 const hasApproved=(EAPPROVED.questions.length||EAPPROVED.dropped.length);
 if(!ENDROUND&&!hasApproved) return '';
 if(ENDROUND&&!hasPending&&!hasApproved){
  return '<div class=safety-lab><span>One more look</span>'
    +'<span class=locked>read back over everything -- nothing new to flag this time</span></div>';}
 let h='';
 if(hasApproved){
  if(EAPPROVED.questions.length){
   h+='<div class=tier style="--tier-color:var(--accent)"><div class=tierlab><span class=tlrow>'
     +'Questions I want to ask my doctor</span></div>';
   for(const q of EAPPROVED.questions){
    h+='<div class=line style="cursor:default"><div class=row><span class=txt>'+esc_(q.text)+'</span>'
      +'<button class=quiet-remove onclick="removeApprovedQuestion(\\''+q.id+'\\')">Remove</button></div></div>';
   }
   h+='</div>';
  }
  if(EAPPROVED.dropped.length){
   h+='<div class=tier style="--tier-color:var(--accent)"><div class=tierlab><span class=tlrow>'
     +'Threads I want to follow up on</span></div>';
   for(const d of EAPPROVED.dropped){
    h+='<div class=line style="cursor:default"><div class=row><span class=txt>'+esc_(d.prompt)+'</span>'
      +'<button class=quiet-remove onclick="removeApprovedDropped(\\''+d.id+'\\')">Remove</button></div></div>';
   }
   h+='</div>';
  }
  h+='<div class=safety-lab><span>One more look</span><span class=locked>added above by me, from Understudy\\'s suggestions</span></div>';
 }
 if(hasPending){
  h+='<div class=safety-lab><span>One more look — suggestions to review</span>'
    +'<span class=locked>not your words · approve or skip each one</span></div>';
  for(const q of (ENDROUND.questions||[])) h+=eqCard(q);
  for(const d of (ENDROUND.dropped||[])) h+=edCard(d);
 }
 return h;}

// VOICE INPUT. Design rule: transcribed text lands in the composer as an EDITABLE DRAFT,
// never as a sent message -- Send stays the only confirm gate, exactly as if the person had
// typed it. One button, click to start, click again to stop (record-then-transcribe; holding
// a button through a long disclosure is a real objection). Recording state is loud so the
// person is never unsure whether the mic is live. Nothing here writes audio to disk; the blob
// lives in the browser tab only, is POSTed once, and is discarded by both sides after
// transcription. (The mic button only renders when the server has local STT available.)
let MIC_STREAM=null, MIC_REC=null, MIC_CHUNKS=[];
// Set by clearAndResay() when it has to interrupt an IN-PROGRESS recording: onstop still fires,
// but it should discard that (now-aborted) clip instead of transcribing and inserting it, then
// go straight into a fresh recording rather than leaving the person to click the mic again.
let MIC_DISCARD_NEXT=false;
// "Clear and re-say": empty the draft, then either restart a recording that was already running
// (discarding the interrupted clip) or start a brand-new one. Never touches Send -- the next
// transcript still lands as an editable draft, same as any other recording.
function clearAndResay(){
 inp.value=''; inp.style.height='auto';
 if(MIC_REC && MIC_REC.state==='recording'){ MIC_DISCARD_NEXT=true; MIC_REC.stop(); return; }
 micstatus.textContent='';
 toggleMic();
}
async function toggleMic(){
 if(MIC_REC && MIC_REC.state==='recording'){MIC_REC.stop();return;}
 if(!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
  micstatus.textContent='Voice input needs a browser with microphone support (MediaRecorder).';return;}
 if(!window.MediaRecorder){
  micstatus.textContent='This browser does not support in-page audio recording.';return;}
 micstatus.textContent='Requesting microphone…';
 try{ MIC_STREAM = await navigator.mediaDevices.getUserMedia({audio:true}); }
 catch(e){ micstatus.textContent='Microphone access was blocked or is unavailable.'; return; }
 MIC_CHUNKS=[];
 let mime='';
 for(const cand of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus']){
  if(window.MediaRecorder.isTypeSupported && window.MediaRecorder.isTypeSupported(cand)){mime=cand;break;}
 }
 try{ MIC_REC = mime ? new MediaRecorder(MIC_STREAM,{mimeType:mime}) : new MediaRecorder(MIC_STREAM); }
 catch(e){ micstatus.textContent='Could not start the recorder.'; stopMicTracks(); return; }
 MIC_REC.ondataavailable=e=>{ if(e.data && e.data.size>0) MIC_CHUNKS.push(e.data); };
 MIC_REC.onstop=async()=>{
  setMicVisual(false);
  stopWave();
  stopMicTracks();
  if(MIC_DISCARD_NEXT){ MIC_DISCARD_NEXT=false; micstatus.textContent=''; toggleMic(); return; }
  micstatus.textContent='Transcribing…';
  const blob=new Blob(MIC_CHUNKS, {type: MIC_REC.mimeType || 'audio/webm'});
  if(!blob.size){ micstatus.textContent='No audio captured — try again.'; return; }
  try{
   const r=await fetch('/stt',{method:'POST',
    headers:{'Content-Type': blob.type || 'application/octet-stream'}, body: blob});
   const j=await r.json();
   if(j.error){ micstatus.textContent='['+j.error+']'; return; }
   if(!j.text){ micstatus.textContent='No speech detected — try again.'; return; }
   insertTranscript(j.text);
   micstatus.textContent='Added to the draft — read it over, then press Send when it is right.';
  }catch(e){ micstatus.textContent='Transcription request failed.'; }
 };
 MIC_REC.start();
 setMicVisual(true);
 startWave(MIC_STREAM);
 micstatus.textContent='Listening… click the mic again when you are done.';
}
// Live level squiggle (Claude/ChatGPT-style): an AnalyserNode drives a small canvas so the
// person can SEE the app hearing them. Purely visual; audio still goes nowhere but /stt.
let MIC_AC=null, MIC_RAF=null;
function startWave(stream){
 try{
  MIC_AC=new (window.AudioContext||window.webkitAudioContext)();
  const src=MIC_AC.createMediaStreamSource(stream);
  const an=MIC_AC.createAnalyser();an.fftSize=512;
  src.connect(an);
  const data=new Uint8Array(an.fftSize);
  const cv=document.getElementById('micwave');
  const cx=cv.getContext('2d');
  const stroke=(getComputedStyle(document.documentElement).getPropertyValue('--danger-quiet')||'#7c433c').trim();
  function draw(){
   MIC_RAF=requestAnimationFrame(draw);
   an.getByteTimeDomainData(data);
   cx.clearRect(0,0,cv.width,cv.height);
   cx.beginPath();
   const step=cv.width/data.length;
   for(let i=0;i<data.length;i++){
    const y=(data[i]/255)*cv.height;
    if(i===0)cx.moveTo(0,y);else cx.lineTo(i*step,y);
   }
   cx.strokeStyle=stroke;cx.lineWidth=1.5;cx.stroke();
  }
  draw();
 }catch(e){}
}
function stopWave(){
 if(MIC_RAF){cancelAnimationFrame(MIC_RAF);MIC_RAF=null;}
 if(MIC_AC){try{MIC_AC.close();}catch(e){}MIC_AC=null;}
 const cv=document.getElementById('micwave');
 if(cv)cv.getContext('2d').clearRect(0,0,cv.width,cv.height);
}
function stopMicTracks(){ if(MIC_STREAM){ MIC_STREAM.getTracks().forEach(t=>t.stop()); MIC_STREAM=null; } }
function setMicVisual(on){
 mic.setAttribute('aria-pressed', String(on));
 mic.title = on ? 'Stop recording' : 'Record voice — click to start, click again to stop';
 composerRow.classList.toggle('recording', on);
}
// Insert at the end of whatever is already drafted, then select just the new span so a
// misheard word is one keystroke away from correction (review-before-send, made concrete).
function insertTranscript(text){
 let prefix=inp.value;
 if(prefix && !/\\s$/.test(prefix)) prefix+=' ';
 const startIdx=prefix.length;
 inp.value=prefix+text;
 inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,144)+'px';
 inp.focus();
 inp.setSelectionRange(startIdx, inp.value.length);
}

// PASTE-IN: bring in a conversation someone else had ABOUT the patient or WITH an AI, and turn
// it into material the patient confirms in their own words. Two source shapes, handled
// differently on purpose (see paste_parser.py and understudy_app.py's /parse_paste):
//
//   (a) A conversation with a general AI assistant. The PATIENT'S OWN turns in it are the
//       patient's own words, exactly as if typed live -- confirming them just adds them to msgs
//       as role:'user', so they flow through the UNCHANGED /sheet pipeline: segmented, offered to
//       the model as candidate clauses, span-verified. Nothing about this feature bypasses the
//       gate; it only gets more of the patient's own words IN FRONT of it. The assistant's own
//       turns can never become sheet content, confirmed or not -- an AI has no observations of
//       its own, only what the patient already told it -- so they can only become a QUESTION
//       offered back (askTopic()), never a claim.
//
//   (b) A note from a trusted advocate. This is never the patient's own words, so it NEVER enters
//       msgs as role:'user', no matter what the patient confirms -- confirming an observation
//       does not make it something the patient said. It can only seed candidate QUESTIONS
//       (askTopic(), same mechanism as (a)'s leftover AI turns); only the patient's own typed
//       answer, afterward, ever reaches the sheet.
//
// THE ONE DECISION: role attribution is the one place an auto-parse could put someone else's
// words on the sheet wearing the patient's receipt (see paste_parser.py's docstring) -- so instead
// of asking the patient to approve every line, this asks ONE question at the seam ("these look
// like your messages, these look like the assistant's — right?") when the parse is confident, and
// falls back to manual one-click-per-line RECOVERY (default: excluded) when it is not. Recovering
// a line the parser missed costs one click; nothing is ever added without one.
let PASTE_PARSED=null;
// Suggested-question queue. Chip text lives here, referenced by index in onclick handlers, so a
// pasted sentence containing a quote character can never break the HTML attribute it would
// otherwise have to sit inside (the same reason NAME/pasted text are built via textContent
// elsewhere in this file, never string-concatenated into markup).
let TOPIC_Q=[];
function queueTopic(q){TOPIC_Q.push(q);return TOPIC_Q.length-1;}
function togglePaste(){const open=pastePanel.style.display==='none';
 pastePanel.style.display=open?'block':'none';
 if(open)pasteText.focus();else closePaste();}
function closePaste(){pastePanel.style.display='none';pasteResult.innerHTML='';pasteText.value='';PASTE_PARSED=null;}
function switchPasteKind(){const isAdv=document.querySelector('input[name=pastekind]:checked').value==='advocate';
 advocateName.style.display=isAdv?'block':'none';
 pasteText.placeholder=isAdv?"Paste or type what they told you…":"Paste the conversation here…";
 pasteResult.innerHTML='';PASTE_PARSED=null;}

async function submitPaste(){
 const kind=document.querySelector('input[name=pastekind]:checked').value;
 const text=pasteText.value.trim();
 if(!text){pasteResult.innerHTML='<div class=refusal>Paste something first.</div>';return;}
 pasteResult.innerHTML='<div class=status>'+thinkingHTML('Reading it over')+'</div>';
 try{
  const r=await fetch('/parse_paste',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({text,kind})});
  const j=await r.json();
  if(j.error){pasteResult.innerHTML='<div class=refusal>'+esc_(j.error)+'</div>';return;}
  PASTE_PARSED=j;
  if(kind==='advocate')renderAdvocateResult(j);else renderConversationResult(j);
 }catch(e){pasteResult.innerHTML='<div class=refusal>Could not read that just now — nothing was added.</div>';}
}

// Flattening internal newlines: a pasted turn is often several sentences/paragraphs long. The
// server joins each kept patient message with '\\n' to build the transcript, and the client maps
// a sheet line's offsets back to a bubble by splitting that SAME transcript on '\\n' and indexing
// into YOUB (see highlightBubble() below) -- one array entry per message. A message that itself
// contains '\\n' would silently break that 1:1 correspondence (this is a pre-existing assumption
// of the live-chat path too, just far less likely to be hit there than by a multi-paragraph
// paste), so every added line is flattened to one line before it becomes a message.
function addPastedPatientTurn(text){
 const flat=text.replace(/\\s*\\n\\s*/g,' ').trim();
 if(!flat)return;
 bubble(flat,'you',{pasted:true});
 msgs.push({role:'user',content:flat});
}

function renderConversationResult(j){
 if(j.confidence==='high'){
  let h='<div class=paste-turns>';
  for(const t of j.turns){
   const cls=t.role_guess==='patient'?'pt-patient':'pt-assistant';
   const lab=t.role_guess==='patient'?'Looks like you':'Looks like the assistant';
   h+='<div class="pt-row '+cls+'"><span class=pt-lab>'+esc_(lab)+'</span>'+esc_(t.text)+'</div>';
  }
  h+='</div><div class=paste-confirm><div class=paste-confirm-q>These look like your messages, '
    +'and these look like the assistant&#8217;s — is that right?</div>'
    +'<div class=paste-confirm-acts>'
    +'<button class=btn-primary onclick="confirmConversation(true)">Yes, that&#8217;s right</button>'
    +'<button class=btn-ghost onclick="confirmConversation(false)">No, let me pick myself</button>'
    +'</div></div>';
  pasteResult.innerHTML=h;
 }else{
  renderRecovery(j.turns);
 }
}

function renderRecovery(turns){
 let h='<p class=paste-hint>I couldn&#8217;t tell who said what here, so nothing is included yet. '
  +'Click <b>This is mine</b> on anything that&#8217;s actually your own words.</p><div class=paste-turns>';
 turns.forEach((t,i)=>{
  h+='<div class=recover-item id="rec'+i+'"><span class=recover-txt>'+esc_(t.text)+'</span>'
   +'<button class=btn-ghost onclick="recoverTurn('+i+')">This is mine</button></div>';
 });
 h+='</div>';
 pasteResult.innerHTML=h;
 window._RECOVER_TURNS=turns;
}
function recoverTurn(i){
 const t=window._RECOVER_TURNS[i];
 if(!t)return;
 addPastedPatientTurn(t.text);
 const row=document.getElementById('rec'+i);
 if(row)row.innerHTML='<span class=recover-txt>'+esc_(t.text)+'</span><span class=recover-added>Added &#10003;</span>';
}

function confirmConversation(yes){
 if(!PASTE_PARSED)return;
 if(!yes){renderRecovery(PASTE_PARSED.turns);return;}
 const mine=PASTE_PARSED.turns.filter(t=>t.role_guess==='patient');
 const theirs=PASTE_PARSED.turns.filter(t=>t.role_guess==='assistant');
 for(const t of mine)addPastedPatientTurn(t.text);
 let h='<div class=paste-added-note>Added '+mine.length+' of your own message'
  +(mine.length===1?'':'s')+' to what your sheet can draw from.</div>';
 if(theirs.length){
  h+='<p class=paste-hint style="margin-top:10px">The assistant&#8217;s own words never go on '
   +'your sheet — an AI only knows what you already told it. If any of these are worth answering '
   +'here, tap one:</p><div class=topic-tray>';
  theirs.forEach(t=>{
   const i=queueTopic(t.text);
   const short=t.text.length>90?t.text.slice(0,87)+'…':t.text;
   h+='<button class=starter onclick="askTopic('+i+',this)">'+esc_(short)+'</button>';
  });
  h+='</div>';
 }
 pasteResult.innerHTML=h;
 if(window.innerWidth<=700)showTab(false);
}

function renderAdvocateResult(j){
 const topics=j.topics||[];
 if(!topics.length){pasteResult.innerHTML='<div class=refusal>Nothing to work with there.</div>';return;}
 const name=(advName.value||'').trim()||'They';
 pasteResult.innerHTML='<div class=paste-confirm-q>'+esc_(name)+' mentioned these things — want '
  +NAME+' to offer them to you as things to talk about?</div><div class=paste-confirm-acts>'
  +'<button class=btn-primary onclick="confirmAdvocate(true)">Yes, offer them to me</button>'
  +'<button class=btn-ghost onclick="confirmAdvocate(false)">No, leave it</button></div>';
 window._ADV_TOPICS=topics;window._ADV_NAME=name;
}
function confirmAdvocate(yes){
 if(!yes){pasteResult.innerHTML='<div class=paste-added-note>Nothing was added.</div>';return;}
 const topics=window._ADV_TOPICS||[],name=window._ADV_NAME||'They';
 let h='<p class=paste-hint>Tap anything you want to talk about — nothing is said for you, this '
  +'only opens the topic so you can answer in your own words.</p><div class=topic-tray>';
 topics.forEach(t=>{
  const q=name+' mentioned: "'+t.text+'" — is that something you want to tell '+NAME+' about?';
  const i=queueTopic(q);
  const short=t.text.length>70?t.text.slice(0,67)+'…':t.text;
  h+='<button class=starter onclick="askTopic('+i+',this)">'+esc_(short)+'</button>';
 });
 h+='</div>';
 pasteResult.innerHTML=h;
}

// Offering a suggested question is identical in kind to the confidant asking one live (send()'s
// own bubble(j.reply,'u') + msgs.push({role:'assistant',...})) -- no model call, because nothing
// here needed one: the question text was already fixed the moment it was queued. Landing it as an
// 'assistant' message means /sheet's existing role:'user' filter already keeps it off the sheet;
// only what the patient types NEXT, in their own words, can ever become a line.
function askTopic(i,btn){
 const q=TOPIC_Q[i];if(q===undefined)return;
 bubble(q,'u');msgs.push({role:'assistant',content:q});
 if(btn){btn.disabled=true;btn.style.opacity=.5;}
 starters.style.display='none';
 if(window.innerWidth<=700)showTab(false);
 inp.focus();
}

// FEEDBACK build only (the button below only exists when the server injects it). Explicit
// opt-in share: nothing leaves unless the tester presses Share — the product's own disclosure
// model, applied to feedback.
function fbToggle(){const p=document.getElementById('fbpanel');
 p.style.display=(p.style.display==='none'?'block':'none');}
async function sendFeedback(){const st=document.getElementById('fbstatus');st.textContent='Sending…';
 try{
  const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({note:document.getElementById('fbnote').value,messages:msgs,
    sheet:SHEET?{lines:SHEET.lines,stats:SHEET.stats,unselected:SHEET.unselected,cues:SHEET.cues}:null})});
  const j=await r.json();
  st.textContent=(j.ok?'Sent — thank you for helping build this.':'['+(j.error||'send failed')+']');
 }catch(e){st.textContent='[send failed]';}}

function showTab(sheet){document.body.classList.toggle('show-sheet',sheet);
 tabchat.setAttribute('aria-selected',String(!sheet));
 tabsheet.setAttribute('aria-selected',String(sheet));
 // Retrigger the mobile tab-switch fade (grafted from A_precision_calm): remove+reflow+re-add so
 // the CSS animation replays each switch. A no-op on desktop — .tab-in only animates inside the
 // 700px mobile media query.
 const active=sheet?sheetcol:chatcol;
 active.classList.remove('tab-in');void active.offsetWidth;active.classList.add('tab-in');}

function reset(){msgs=[];chat.innerHTML='';YOUB=[];SHEET=null;SEL=-1;PINNED=-1;POLISH={};PROBE={};where.innerHTML='';
 document.documentElement.classList.remove('has-sheet');
 ENDROUND=null;EAPPROVED={questions:[],dropped:[]};
 hint.textContent='';refusal.innerHTML='';
 ACCOM=[];renderAccom();
 DIARY=[];renderDiary();
 pastePanel.style.display='none';pasteResult.innerHTML='';pasteText.value='';PASTE_PARSED=null;TOPIC_Q=[];
 body.innerHTML='Talk for a bit, then press <b>Make my sheet</b>.';body.className='placeholder';
 starters.style.display='flex';
 // Fix (design review, 2026-08-01): this greeting used to be rendered by bubble() only -- never
 // pushed to msgs -- so on turn one the model got [system, user] with no record of the very
 // question it had just asked, and the patient's first reply looked to the model like it came out
 // of nowhere. Pushing it here costs nothing (the /sheet transcript join only reads role==='user'
 // messages, so this assistant turn never touches the gate/pipeline path) and gives the model the
 // conversational context it's implicitly answering into.
 const GREETING="Hi — I'm "+NAME+". Whatever's going on, you can tell me here, even the awkward stuff. What's on your mind before your visit?";
 bubble(GREETING,'u');msgs.push({role:'assistant',content:GREETING});
 showTab(false);}
// Enter sends; Shift+Enter makes a new line (the composer is a textarea so a long spoken or
// typed thought can be reviewed before it becomes part of the record).
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,144)+'px';});
reset();
</script></body></html>"""
# Name/tagline/mode are injected once at startup; a rename never touches the markup.
# Labelled by CONSEQUENCE, not by model. The model name is for us; what happens to the person's
# words is for them -- and this dropdown is the product's central promise made operable, so it is
# the one control that stays on a patient's first screen. The model id is kept in parentheses so
# the demo (and a judge) can still see exactly which Gemma is answering.
_OPTS_LOCAL = ('<option value=e2b>Stay on this device (Gemma 4 E2B)</option>'
               '<option value=e4b>Stay on this device, slower and stronger (Gemma 4 E4B)</option>'
               '<option value=nebius>Send my words to the cloud — faster (Nebius)</option>')
_hb = BACKENDS.get(HOSTED_BACKEND, BACKENDS["nebius"])
_hosted_is_gemma = _hb["kind"] == "local"
_OPTS_HOSTED = ('<option value=' + HOSTED_BACKEND + '>'
                + (_hb["label"].split(" — ")[0] + " — hosted demo" if _hosted_is_gemma
                   else 'Cloud demo (Nebius) — hosted preview') + '</option>')
# HONESTY FIX: this used to be scoped only inside _BANNER_HOSTED, which renders only when
# HOSTED=1. The always-visible page footer (below, __FOOTER_CLAIM__) carried its own separate,
# unconditioned "Nothing leaves this device unless you choose it" string that never mentioned
# FEEDBACK at all -- so on the plain local FEEDBACK build (UNDERSTUDY_FEEDBACK=1, no HOSTED,
# exactly the "hackathon tinkering phase" case the FEEDBACK flag exists for), the one privacy
# claim visible on every single page load never disclosed that pressing Share writes the
# conversation and sheet to disk (see the /feedback handler). The fb-consent text right next to
# the Share button does disclose it, but a claim that is exact only where a user happens to
# expand a collapsed panel, and unconditionally broader everywhere else, is the same "quietly
# grown to cover text it was never about" failure gate/pipeline.py's render_text warns about. Now
# both the hosted banner AND the footer use this one FEEDBACK-aware string.
_STORED = ('Conversations aren&#8217;t stored unless you choose to share them with the builder.'
           if FEEDBACK else 'Conversations aren&#8217;t stored.')
_BANNER_HOSTED = ('<div class=hosted-note><b>Cloud preview.</b> '
                  + ('This link runs the same Gemma model on sponsor infrastructure (Daytona) so you '
                     'can try it from any phone — the shipped product runs it on your own device. '
                     if _hosted_is_gemma else
                     'This link runs in the cloud so you can try it from any phone — the shipped '
                     'product runs entirely on your own device. ')
                  + '<b>Demo only:</b> please don&#8217;t enter real personal health information. '
                  + _STORED + '</div>')
_FEEDBACK_UI = ('<div class=fb><button class=fb-open onclick=fbToggle()>'
                'Tinkering with this? Share your thoughts with the builder</button>'
                '<div class=fb-panel id=fbpanel style=display:none>'
                '<textarea id=fbnote rows=3 placeholder="What did you think? What broke? What would you change?"></textarea>'
                '<div class=fb-row><span class=fb-consent>Pressing Share sends this conversation, the current '
                'sheet, and your note to the builder. Nothing is sent otherwise.</span>'
                '<button class=fb-send onclick=sendFeedback()>Share</button></div>'
                '<div class=fb-status id=fbstatus></div></div></div>')
_MIC_UI = ('<canvas id=micwave width=180 height=30 aria-hidden=true></canvas>'
           '<button class=btn-mic id=mic type=button aria-pressed=false '
           'title="Record voice — click to start, click again to stop" onclick=toggleMic()>&#127908;</button>'
           # "Clear and re-say" (a real but small gap, cut for time in the voice_mode spike --
           # folded back in): a hard sentence often takes several tries, and today re-recording
           # APPENDS onto whatever draft is already there, so a retry produces garbage the person
           # then has to hand-edit. This clears the draft and immediately starts listening again --
           # one click, not two -- but the guarantee is unchanged: the fresh transcript still lands
           # in the composer as an editable draft, Send is still the only confirm gate.
           '<button class=btn-resay id=micresay type=button '
           'title="Clear the draft and record again" onclick=clearAndResay()>Clear &amp; re-say</button>')
PAGE = (PAGE.replace("__APP_NAME__", APP_NAME).replace("__TAGLINE__", TAGLINE)
            .replace("__BACKEND_OPTIONS__", _OPTS_HOSTED if HOSTED else _OPTS_LOCAL)
            .replace("__HOSTED_BANNER__", _BANNER_HOSTED if HOSTED else "")
            .replace("__FEEDBACK_UI__", _FEEDBACK_UI if FEEDBACK else "")
            .replace("__MIC_UI__", _MIC_UI if STT_AVAILABLE else "")
            .replace("__FOOTER_CLAIM__", _STORED)
            .replace("__TTS_AVAILABLE__", "true" if TTS_AVAILABLE else "false"))



# DETERMINISTIC SELF-HARM BACKSTOP for the live chat (review find, 2026-08-01). The sheet path
# has always run safety_net over the transcript; the chat turn relied on one clause in CHAT_SYS,
# which the five-variant harness measured failing 3/3 on the s3 passive-ideation script ("sometimes
# i think theyd be better without me around"). This turn must never depend on the model noticing.
# The one-line off switch below exists because the trade is real: a false-positive gets a slightly
# canned reply instead of the model's phrasing. That trade is accepted -- over-triggering costs
# tone; the miss costs the one moment this product must never miss.
CHAT_SAFETY_GUARD = True

CHAT_GUARD_REPLY = (
    "Thank you for trusting me with that — it matters, and I'm glad you said it here. "
    f"{safety_net.CRISIS} If you are in immediate danger, please call 911 now. "
    "I'm still here, and whenever you want we can keep getting things ready for your doctor — "
    "what you just shared is exactly the kind of thing they need to know about."
)


def _chat_guard(msgs):
    """The hand-authored crisis reply when the NEWEST patient message trips the self-harm net,
    else None. Checks only the newest turn (not history) so one disclosure does not pin every
    later reply to the canned text; the model resumes on the next non-triggering turn with the
    disclosure still in its context. Shared by both /chat paths so the two cannot drift."""
    if not CHAT_SAFETY_GUARD:
        return None
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if user_msgs and safety_net.mentions_self_harm(str(user_msgs[-1].get("content", ""))):
        return CHAT_GUARD_REPLY
    return None


def _chat_canned(msgs):
    """The canned starter-chip reply for this conversation, or None.

    Only as the FIRST user turn -- if the patient later happens to type one of these sentences
    themselves mid-conversation, len(user_msgs) != 1 and it falls through to the model like
    anything else. Shared by both /chat paths (blocking and streamed) so the two cannot drift.
    """
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_msgs) == 1:
        return CHIP_REPLIES.get(str(user_msgs[0].get("content", "")).strip())
    return None


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        try:
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The client is already gone (e.g. it gave up after a huge/odd payload) -- there is
            # no socket left to write the error onto. Swallow rather than let it surface as an
            # unhandled exception in the request thread.
            pass
    def _drain(self, n, cap=8_000_000):
        """Consume up to `n` bytes of the request body without keeping them.

        Used when we are about to reject a request (too large) without reading its body first.
        BaseHTTPRequestHandler defaults to HTTP/1.0 (connection closes after the response), so
        responding immediately while the client is still mid-write raced their send against our
        close: the fuzz run's huge_content (a 2MB message) and many_messages (50k messages) cases
        both hit BrokenPipeError on the CLIENT side for exactly this reason. Draining first lets
        their write complete normally before we close. `cap` bounds this: a Content-Length that
        lies about size (deliberately huge) must not park the thread forever reading it.
        """
        remaining = min(n, cap)
        while remaining > 0:
            try:
                chunk = self.rfile.read(min(remaining, 65536))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
            if not chunk:
                return
            remaining -= len(chunk)
    def do_GET(self):
        b = PAGE.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(b)
    def _handle_stt(self, raw):
        """Local speech-to-text. `raw` is whatever bytes the browser's MediaRecorder produced
        (webm/opus, ogg/opus, or wav) -- faster-whisper decodes it from memory, no temp file.
        Neither the audio nor the transcript is ever written to disk; the returned text is a
        DRAFT for the composer only -- this endpoint never touches msgs[] and has no path into
        the sheet."""
        if not STT_AVAILABLE:
            return self._json({"error": "voice input is not available on this install"}, 501)
        if not raw:
            return self._json({"error": "no audio received"}, 400)
        if len(raw) > 25_000_000:
            return self._json({"error": "that recording is too long — try a shorter clip"}, 413)
        try:
            model = get_stt_model()
        except Exception as e:
            return self._json({"error": f"speech-to-text unavailable: {e}"}, 500)
        try:
            t0 = time.time()
            segments, _info = model.transcribe(
                io.BytesIO(raw), language="en", beam_size=5,
                vad_filter=True, condition_on_previous_text=False)
            text = "".join(seg.text for seg in segments).strip()
            return self._json({"text": text, "ms": int((time.time() - t0) * 1000)})
        except Exception as e:
            return self._json({"error": f"transcription failed: {e}"}, 500)

    def _write_ndjson(self, obj):
        """One streaming event, as one line of JSON. Flushed immediately -- buffering here would
        defeat the entire point of streaming (the browser wouldn't see bytes until Python decided
        to hand them over)."""
        self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _handle_chat(self, req, backend):
        """Streaming /chat. Response is `application/x-ndjson`: one JSON object per line, in
        order -- {"content": piece} events as tokens arrive, then exactly one terminal event,
        either {"done": true, "reply": <full text>, "ms": ..., "where": ...} or {"error": ...}.

        This method owns its own send_response/end_headers and its own try/except: once headers
        for a 200 have gone out, the generic do_POST except-clause below (which calls self._json,
        i.e. ANOTHER send_response) can no longer run for this request -- so every failure mode
        here, including one raised by our own loop rather than by call_model_stream, is caught and
        turned into one last ndjson line instead of leaking a raw traceback into a half-open body
        or crashing the request thread.

        msgs[] is untouched by this method -- exactly as under the old blocking /chat, the browser
        is the one place a completed reply gets appended to msgs[], only after a full reply comes
        back intact (see send() in PAGE's <script>). A reply cut short by an error is never partial
        in msgs[]; it simply never arrives there, same as before.
        """
        msgs = req.get("messages", [])
        if not isinstance(msgs, list):
            msgs = []
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        pieces, where, ms, sent_terminal = [], "device", 0, False

        # A starter-chip reply is canned (zero model calls, lands instantly) but still goes out
        # over this same ndjson channel, as one {"content": ...} plus the normal terminal
        # {"done": ...} -- so the client's line reader never has to branch on which kind of reply
        # it is receiving. See _chat_canned and the /chat dispatch in do_POST.
        t0c = time.time()
        canned = _chat_canned(msgs)
        if canned:
            try:
                self._write_ndjson({"content": canned})
                self._write_ndjson({"done": True, "ms": int((time.time() - t0c) * 1000),
                                    "where": "device", "reply": canned, "canned": True})
            except Exception:
                pass  # client already gone; nothing left to report to
            return
        # Same deterministic self-harm backstop as the blocking path, same wire shape as a
        # canned reply -- the client's line reader never has to know which kind arrived.
        guard = _chat_guard(msgs)
        if guard:
            try:
                self._write_ndjson({"content": guard})
                self._write_ndjson({"done": True, "ms": int((time.time() - t0c) * 1000),
                                    "where": "device", "reply": guard, "guard": True})
            except Exception:
                pass  # client already gone; nothing left to report to
            return
        try:
            for ev in call_model_stream([{"role": "system", "content": CHAT_SYS}] + msgs,
                                        backend, think=False):
                if "error" in ev:
                    self._write_ndjson({"error": ev["error"]})
                    sent_terminal = True
                    break
                piece = ev.get("content")
                if piece:
                    pieces.append(piece)
                    self._write_ndjson({"content": piece})
                if ev.get("done"):
                    where, ms = ev.get("where", "device"), ev.get("ms", 0)
        except Exception as e:
            # A break in OUR loop (e.g. the client vanished mid-write) rather than something
            # call_model_stream already turned into an {"error": ...} event. Best-effort one more
            # line; if the socket is already gone this itself raises and is swallowed below.
            try:
                self._write_ndjson({"error": str(e)})
                sent_terminal = True
            except Exception:
                return
        if not sent_terminal:
            try:
                self._write_ndjson({"done": True, "ms": ms, "where": where,
                                    "reply": "".join(pieces).strip()})
            except Exception:
                pass  # client already gone; nothing left to report to

    def do_POST(self):
        # EVERYTHING below used to run partly OUTSIDE this try (Content-Length read + json.loads
        # happened before the try even opened), so a malformed body raised before the JSON error
        # path could catch it -- the exception propagated up through BaseHTTPRequestHandler,
        # which closes the socket without a response. The client sees a bare socket-level
        # RemoteDisconnected instead of a clean JSON error. An overnight fuzz run
        # (daytona/loops/harvest/fuzz_findings.jsonl) measured ~2,480 hits of exactly this across
        # /chat, /sheet, /polish, /export, /handoff, /feedback -- one root cause, six endpoints.
        # Fix: the ENTIRE body (read, parse, dispatch, write) is now inside one try/except, so
        # every failure mode -- bad JSON, a body that parses but isn't an object, a handler
        # exception, even the client vanishing mid-response -- ends in a clean JSON reply (or, if
        # the socket itself is gone, a quiet no-op) instead of an unhandled exception.
        try:
            n = int(self.headers.get("Content-Length", 0))
            if self.path == "/stt":
                # Raw audio bytes, not JSON — its own path before the JSON parse, with its own
                # larger size cap (a spoken utterance is bigger than a chat turn).
                raw = self.rfile.read(n) if n > 0 else b""
                return self._handle_stt(raw)
            if n > 300_000:
                # Drain the body before responding -- see _drain's docstring. Without this the
                # fuzz run's huge_content (a 2MB message) and many_messages (50k messages) cases
                # raced our early close against the client's still-in-flight send and surfaced as
                # BrokenPipeError instead of a clean rejection.
                self._drain(n)
                return self._json({"error": "request too large"}, 413)
            raw_body = self.rfile.read(n)
            try:
                req = json.loads(raw_body or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return self._json({"error": f"malformed JSON body: {e}"}, 400)
            if not isinstance(req, dict):
                return self._json({"error": "request body must be a JSON object"}, 400)
            backend = req.get("backend", DEFAULT_BACKEND); msgs = req.get("messages", [])
            if not isinstance(msgs, list):
                msgs = []
            if HOSTED:
                backend = HOSTED_BACKEND   # hosted surface serves ONE configured backend; spoofed values must not 500-loop
            if self.path == "/chat":
                # MERGE RECONCILIATION (2026-08-01) -- streaming vs. every other contract on /chat.
                #
                # Streaming rewrites what a /chat RESPONSE IS: not one JSON document but n ndjson
                # lines. That collides with two things that both predate it and both still have to
                # hold. (1) The hardening pass's guarantee that every POST answers with one
                # parseable JSON document -- the property the overnight fuzz run exists to defend.
                # (2) The gate (attempts/verify_app.py) which asserts exactly that shape on /chat.
                # Letting streaming simply replace the blocking path would have silently broken
                # both for every non-browser caller.
                #
                # So /chat NEGOTIATES instead of switching: the blocking single-JSON reply stays
                # the default, and the streamed ndjson reply is opt-in, requested by the one
                # client that can actually read it (send() in PAGE posts stream:true). Same
                # replies, same canned-chip shortcut, same backend -- only the framing differs.
                #
                # An empty conversation is refused here, before either path starts: there is
                # nothing to reply to (/sheet already answers this same case the same way), and a
                # rejection is only expressible as clean JSON while the headers are still ours to
                # choose. Real traffic never reaches it -- send() pushes the user turn into msgs[]
                # before it fetches.
                if not any(isinstance(m, dict) and m.get("role") == "user"
                           and str(m.get("content", "")).strip() for m in msgs):
                    return self._json({"error": "nothing said yet"}, 400)
                if req.get("stream"):
                    # _handle_chat owns its own send_response/end_headers and its own try/except
                    # (see its docstring): once a 200's headers are out, the generic
                    # `except Exception -> self._json(...)` below would attempt a SECOND
                    # send_response and corrupt the response. It is internally total -- every path
                    # ends in a written ndjson line or a deliberate silent give-up on a dead
                    # socket -- so it never relies on that outer handler.
                    return self._handle_chat(req, backend)
                # Blocking path: the default, and the one the gate grades.
                t0c = time.time()
                canned = _chat_canned(msgs)
                if canned:
                    return self._json({"reply": canned,
                                       "ms": int((time.time() - t0c) * 1000),
                                       "where": "device", "canned": True})
                guard = _chat_guard(msgs)
                if guard:
                    return self._json({"reply": guard,
                                       "ms": int((time.time() - t0c) * 1000),
                                       "where": "device", "guard": True})
                # chat: thinking OFF -- warmth doesn't need chain-of-thought and CPU can't spare it
                out = call_model([{"role": "system", "content": CHAT_SYS}] + msgs, backend, think=False)
                if "error" in out: return self._json(out)
                return self._json({"reply": out["content"].strip(), "ms": out["ms"], "where": out["where"]})
            if self.path == "/polish":
                # "Say it better for me" — the delegation-seeker path (Alex ruling 07-30 23:37:
                # pluralism over the single control-seeker use case). The model DRAFTS a clearer
                # wording per line; deterministic gates run over each draft (vocab introduction =
                # suppressed outright, other gate misses = caution strings); the patient approves
                # or rejects PER LINE client-side. Nothing changes silently: an accepted draft is
                # labeled on the sheet and its receipt still points at the verbatim original.
                # Evidence base: 80-cell rewrite test, intake/anvil/REWRITE_RESULTS.md.
                lines_in = req.get("lines", [])[:12]
                think = bool(req.get("think", False))
                drafts, last = [], None
                for ln in lines_in:
                    text = str(ln.get("text", ""))[:600]
                    if not text.strip():
                        continue
                    out = call_model([{"role": "user", "content": polish.build_prompt(text)}],
                                     backend, force_json=True, think=think)
                    if "error" in out:
                        drafts.append({"id": ln.get("id"), "error": out["error"]})
                        continue
                    last = out
                    draft = polish.parse_rewrite(out["content"])
                    if not draft:
                        drafts.append({"id": ln.get("id"), "error": "no usable draft"})
                        continue
                    g = polish.run_gates(draft, text)
                    drafts.append({"id": ln.get("id"),
                                   "draft": None if g["suppressed"] else draft.strip(),
                                   "suppressed": g["suppressed"],
                                   "cautions": g["cautions"]})
                return self._json({"drafts": drafts,
                                   "where": (last or {}).get("where", "device"),
                                   "ms": (last or {}).get("ms", 0)})

            if self.path == "/probe_question":
                # "Ask me more about this" -- step 1: which clinical element to ask about next.
                # Deterministic, no model call (see gate/probe.py for why). `asked` is the list
                # of element keys already asked FOR THIS LINE, tracked client-side.
                q = probe.next_question(str(req.get("text", ""))[:600], req.get("asked", []))
                return self._json({"question": q})

            if self.path == "/probe_answer":
                # "Ask me more about this" -- step 2: the patient's typed answer lands here. No
                # model call: the answer is appended to the END of the transcript and re-segmented,
                # which is what keeps every existing line's id stable (see pipeline.append_answer's
                # docstring for why append-only preserves ids). Also no model call for selection --
                # a patient's direct answer to a question about a line they chose is being ADDED
                # by them, not selected by the model, same provenance class as addLine()'s
                # "patient put this back" case.
                out = pipeline.append_answer(
                    str(req.get("transcript", "")), str(req.get("answer", ""))[:1000],
                    parent_id=req.get("parent_id"), probe_element=req.get("element"))
                return self._json(out)

            if self.path == "/feedback":
                # FEEDBACK builds only. Explicit opt-in share from the tester — the ONLY write
                # path in the whole app. No IP, no user agent, no identity: just what they chose
                # to send, timestamped. Ambient logging stays off (log_message is disabled).
                if not FEEDBACK:
                    return self._json({"error": "feedback is not enabled on this build"}, 404)
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "note": str(req.get("note", ""))[:5000],
                       "messages": req.get("messages"),
                       "sheet": req.get("sheet")}
                fb_dir = Path(__file__).parent / "feedback"
                fb_dir.mkdir(exist_ok=True)
                with open(fb_dir / "feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return self._json({"ok": True})

            if self.path == "/parse_paste":
                # PASTE-IN. Deterministic only -- NO model call, by design (see paste_parser.py):
                # role attribution here is pattern-matching on literal speaker labels a chat
                # product already stamped into the text, never a content-based guess. Nothing
                # returned from this endpoint is IN the sheet's candidate pool yet; the client
                # asks the patient once ("these look like your messages, these look like the
                # assistant's -- right?") before anything is added to msgs, and an advocate note
                # never becomes sheet content at all, only candidate questions.
                text = str(req.get("text", ""))[:20_000]
                kind = req.get("kind", "conversation")
                if not text.strip():
                    return self._json({"error": "nothing pasted"})
                if kind == "advocate":
                    topics = paste_parser.parse_advocate_topics(text)
                    return self._json({"kind": "advocate", "topics": topics})
                result = paste_parser.parse_conversation(text)
                return self._json({"kind": "conversation", **result})
            if self.path == "/speak":
                # "Read my sheet aloud." Re-derives the sheet from the transcript + the ids the
                # patient currently kept -- same discipline as /export, and for the same reason:
                # what gets read has to be what's on screen after their deletions, not a stale
                # server-side copy. No model call (assemble_from_ids is pure/deterministic); the
                # only new work is turning that structured sheet into speech via gate/speech.py
                # (Windows SAPI/System.Speech through PowerShell -- see that module's docstring
                # for why server-side beat browser speechSynthesis here).
                sheet = pipeline.assemble_from_ids(
                    req.get("transcript", ""), req.get("ids", []),
                    dropped_ids=req.get("dropped", []),
                    keep_cues=req.get("keep_cues"),
                    added_ids=req.get("added", []),
                    app_name=APP_NAME,
                    chip_spans=req.get("chip_spans"),
                    accommodations=req.get("accommodations", []),
                    diary=req.get("diary", []))
                segments = pipeline.speech_segments(sheet)
                try:
                    wav = speech.synthesize(segments)
                except Exception as e:
                    return self._json({"error": str(e)})
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.end_headers()
                self.wfile.write(wav)
                return

            if self.path == "/export":
                # Re-derives the sheet from the transcript plus the ids the patient KEPT. No model
                # call, and deliberately no trust in a client-side rendering: what gets emailed has
                # to be what they approved after their deletions, and it has to be assembled by the
                # same code path that made the on-screen one.
                #
                # accommodations travels the same way: the client holds the current "how I need
                # this visit to go" list (curated statements the patient chose, plus anything they
                # typed) and resends it whole on every export, exactly like ids/dropped/added for
                # the transcript-derived lines. No model ever sees or touches this list.
                sheet = pipeline.assemble_from_ids(
                    req.get("transcript", ""), req.get("ids", []),
                    dropped_ids=req.get("dropped", []),
                    keep_cues=req.get("keep_cues"),
                    added_ids=req.get("added", []),
                    app_name=APP_NAME,
                    chip_spans=req.get("chip_spans"),
                    accommodations=req.get("accommodations", []),
                    diary=req.get("diary", []))
                text = pipeline.render_text(sheet)
                # End-round additions (questions for the doctor / dropped threads) are APP-AUTHORED,
                # never patient words, and only reach the export if the patient approved them
                # client-side (see betterWords/POLISH{} for the same approve-per-item discipline).
                # Re-verified here rather than trusted verbatim from the client -- same distrust the
                # rest of this handler already applies to every other client-supplied thing: a
                # question must still pass the vocabulary gate, and a dropped-thread quote must still
                # verify against the LIVE transcript, so a stale or tampered client cannot get
                # something ungrounded into the printed/emailed letter.
                er = req.get("endround") or {}
                patient_only = req.get("transcript", "")
                q_clean = [c for c in (endround.verify_export_item("question", q, patient_only)
                                       for q in er.get("questions", []) if isinstance(q, dict)) if c]
                d_clean = [c for c in (endround.verify_export_item("dropped", d, patient_only)
                                       for d in er.get("dropped", []) if isinstance(d, dict)) if c]
                extra = endround.render_additions(q_clean, d_clean)
                if extra:
                    text = text + "\n\n" + extra
                return self._json({"text": text})

            if self.path == "/endround":
                # THE END-OF-CONVERSATION HIGHER-REASONING ROUND (Alex's own idea, deferred --
                # three_pane/BRAINSTORM_ALEX_2026-07-31.md section 2). One deliberate pass over the
                # WHOLE conversation, thinking ON, producing (a) questions the patient might want to
                # ask their doctor and (b) threads they raised and never came back to. Always deep
                # reasoning -- that IS the feature, not an option the checkbox happens to control --
                # so `think` is hardcoded True here regardless of the "deep reasoning" toggle's state,
                # using the exact same call_model(think=...) plumbing /sheet already wires through.
                def chat_fn(model, messages, temperature=0, max_tokens=900, force_json=True):
                    out = call_model(messages, backend, force_json=force_json, think=True)
                    if "error" in out:
                        raise RuntimeError(out["error"])
                    return out["content"]

                t0 = time.time()
                cfg = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])
                result = endround.build_review(msgs, chat_fn, cfg["model"])
                result["ms"] = int((time.time() - t0) * 1000)
                result["where"] = "cloud" if cfg["kind"] == "cloud" else "device"
                return self._json(result)

            if self.path in ("/sheet", "/handoff"):
                # PATIENT TURNS ONLY. The old /handoff joined every message as
                # 'role: content', so the assistant's words went into the transcript too. Under the
                # old design that only muddied the summary. Under the gate it is far worse: the
                # transcript is the source of truth the verifier checks against, so anything the AI
                # said would VERIFY as the patient's own words and could be assembled into the
                # sheet, wearing a valid receipt. The guarantee is only as good as this line.
                # Build the transcript AND, in the same pass, the char-offset ranges of any user
                # turn that originated from a starter chip (msgs[].origin==='chip', set by
                # send()/startWith() in PAGE). Those ranges are computed here, once, against the
                # exact string gate.pipeline will segment -- see the CHIP PROVENANCE note below for
                # why this is the only point in the pipeline that can know it.
                parts, chip_spans, off = [], [], 0
                for m in msgs:
                    if m.get("role") != "user":
                        continue
                    c = m.get("content", "").strip()
                    if not c:
                        continue
                    start = off
                    if m.get("origin") == "chip":
                        chip_spans.append([start, start + len(c)])
                    parts.append(c)
                    off = start + len(c) + 1  # +1 for the "\n" join() inserts below
                transcript = "\n".join(parts)
                if not transcript:
                    return self._json({"error": "nothing said yet"})

                t0 = time.time()
                cfg = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])
                think = bool(req.get("think", False))

                def chat_fn(model, messages, temperature=0, max_tokens=700, force_json=True):
                    """Adapt call_model to the signature gate.pipeline expects, so the runtime
                    switcher keeps working: whichever backend the dropdown names is the one the
                    selection step runs on."""
                    out = call_model(messages, backend, force_json=force_json, think=think)
                    if "error" in out:
                        raise RuntimeError(out["error"])
                    return out["content"]

                sheet = pipeline.build_sheet(transcript, chat_fn, cfg["model"],
                                             app_name=APP_NAME, chip_spans=chip_spans,
                                             accommodations=req.get("accommodations", []))
                sheet["ms"] = int((time.time() - t0) * 1000)
                sheet["where"] = "cloud" if cfg["kind"] == "cloud" else "device"
                sheet["text"] = pipeline.render_text(sheet)
                return self._json(sheet)
            self._json({"error": "unknown path"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Client vanished mid-request; nothing to write a response onto.
            pass
        except Exception as e:
            return self._json({"error": str(e)})


if __name__ == "__main__":
    print(f"[understudy] default runtime={DEFAULT_BACKEND}  ->  open  http://localhost:{PORT}")
    if HOST != "127.0.0.1":
        print(f"[understudy] bound to {HOST} — reachable from other devices on this network")
    threading.Thread(target=prewarm, args=(DEFAULT_BACKEND,), daemon=True).start()
    class Srv(socketserver.ThreadingTCPServer):
        allow_reuse_address = True   # restarts must not lose a race with TIME_WAIT sockets
        daemon_threads = True        # a hung request thread must not block shutdown
    with Srv((HOST, PORT), H) as s:
        s.serve_forever()
