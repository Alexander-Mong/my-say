"""
speech.py -- server-side text-to-speech for "Read my sheet aloud", via Windows SAPI
(System.Speech) invoked through powershell.exe. Zero pip installs.

WHY POWERSHELL/SYSTEM.SPEECH RATHER THAN pywin32
-------------------------------------------------
Alex already has a SAPI wrapper -- C:\\Projects\\Verticals\\PZ-Companion\\tools\\pzcompanion\\tts.py
-- but it goes through `win32com.client` (pywin32), which is NOT installed in this environment
(checked: `import win32com` fails). That wrapper is also built for a different job: a persistent,
interruptible voice for a live game companion (COM object held open on a worker thread, barge-in
support). This feature is a one-shot "turn this text into a WAV for one HTTP response" job with no
running state, so `Add-Type -AssemblyName System.Speech` from PowerShell is the simpler, dependency-
free tool for it -- it ships with every reachable Windows via .NET, and powershell.exe is already on
PATH. No pip install, no new import for understudy_app.py to fail open on.

WHY SERVER-SIDE AT ALL, NOT BROWSER speechSynthesis
----------------------------------------------------
Evaluated both (see BUILD_REPORT.md for the full writeup). Browser speechSynthesis needs zero
server code and would normally be the simpler pick, but the app's one promise is "nothing leaves
this device" (it's in the footer), and that is NOT reliably true of the browser API: Chrome's
`SpeechSynthesisVoice` objects carry a `localService` boolean specifically because some of
Chrome's own bundled voices ("Google US English" etc.) synthesize by calling a Google endpoint --
verified in this session (see BUILD_REPORT.md) by enumerating `speechSynthesis.getVoices()` and
reading `.localService` rather than assuming. Shipping a feature that could silently phone home,
in the one app whose entire pitch is that it doesn't, is a worse failure than the extra server code
costs. SAPI via System.Speech never leaves the machine -- there is no network client in this file.

VOICE-AS-PUNCTUATION
---------------------
The printed sheet marks "this is Understudy's text, not yours" with [brackets] (gate/pipeline.py,
render_text). Read aloud there is no typography, so the same distinction becomes WHICH VOICE is
speaking: this machine has two installed SAPI voices (Microsoft David Desktop, Microsoft Zira
Desktop -- verified via `$synth.GetInstalledVoices()`), so app-authored text (section headings,
safety information) speaks in one and the patient's own words in the other. That is a real voice
change, not a tone or pitch wobble, so it survives even half-attention. `_speak.ps1` degrades to a
single voice with a rate difference if a machine has fewer than two (or different-named) voices
installed -- see that file's own comment.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
_SCRIPT = HERE / "_speak.ps1"

# Overridable so a machine with different voice names installed doesn't need a code change --
# _speak.ps1 checks these against GetInstalledVoices() and falls back gracefully if either is
# missing (see that file).
APP_VOICE = os.environ.get("UNDERSTUDY_TTS_APP_VOICE", "Microsoft David Desktop")
PATIENT_VOICE = os.environ.get("UNDERSTUDY_TTS_PATIENT_VOICE", "Microsoft Zira Desktop")


def available() -> bool:
    """Cheap proxy check, mirroring STT_AVAILABLE's import-time probe in understudy_app.py: is
    this a Windows box with PowerShell on PATH? This does NOT guarantee the System.Speech
    assembly actually loads (that is only knowable by trying) -- a live probe here would cost
    every server startup the ~1s PowerShell takes to spin up, to protect a feature nobody may
    ever click. A real failure at synth time still surfaces as a clean error to the caller
    (see synthesize()) rather than a crash, so the gate degrades honestly either way; this
    function only controls whether the "Read my sheet aloud" button renders at all, exactly as
    STT_AVAILABLE controls the mic button.
    """
    return os.name == "nt" and shutil.which("powershell") is not None


def synthesize(segments: list[tuple[str, str]]) -> bytes:
    """[(speaker, text), ...] -> WAV bytes. Runs PowerShell/System.Speech once, synchronously.

    Nothing is left on disk afterward, mirroring the STT doctrine elsewhere in this app (audio in,
    audio/transcript out, never persisted): the temp JSON script-input and the temp WAV are both
    deleted in `finally`, whether synthesis succeeded or raised.
    """
    if not available():
        raise RuntimeError("read-aloud is not available on this install (needs Windows + PowerShell)")
    segments = [(v, t) for v, t in segments if t and t.strip()]
    if not segments:
        raise RuntimeError("nothing to read")

    tmp_dir = Path(tempfile.mkdtemp(prefix="understudy_tts_"))
    in_path = tmp_dir / "segments.json"
    out_path = tmp_dir / "out.wav"
    try:
        in_path.write_text(json.dumps({
            "segments": [{"voice": v, "text": t} for v, t in segments],
            "appVoice": APP_VOICE,
            "patientVoice": PATIENT_VOICE,
        }, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(_SCRIPT), "-InJson", str(in_path), "-OutWav", str(out_path)],
            capture_output=True, text=True, timeout=90)
        if proc.returncode != 0 or not out_path.exists():
            detail = (proc.stderr or proc.stdout or "unknown PowerShell error").strip()
            raise RuntimeError(f"speech synthesis failed: {detail[-500:]}")
        data = out_path.read_bytes()
        if not data:
            raise RuntimeError("speech synthesis produced no audio")
        return data
    finally:
        for p in (in_path, out_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass
