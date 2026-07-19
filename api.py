#!/usr/bin/env python3
"""
Say It Right - API Server

Endpoints:
  POST /assess     - Upload audio + text (or target), get pronunciation score
  GET  /phonemes   - Get target phonemes for a text (phrasebook or G2P)
  POST /phonemes   - Save a corrected target to the phrasebook
  GET  /health     - Health check
  GET  /modes      - List available assessment modes and backends
  GET  /           - Simple HTML test page (browser UI)

Usage:
  python api.py                                # default: everyday mode, wav2vec2
  python api.py --mode irab                    # boot in irab mode, Tarteel Whisper
  python api.py --mode irab --backend tarteel  # explicit backend
  python api.py --port 9000                    # custom port
  python api.py --host 127.0.0.1               # localhost only

Example (classic — text only, engine runs G2P):
  curl -X POST http://localhost:8000/assess \\
    -F "text=كتاب" \\
    -F "audio=@test_audio/clip.wav"

Example (manual target — skip G2P entirely):
  curl -X POST http://localhost:8000/assess \\
    -F "target=k,i,t,aa,b" \\
    -F "audio=@test_audio/clip.wav"

Example (per-request mode override — lazily builds second engine):
  curl -X POST http://localhost:8000/assess \\
    -F "text=كِتَابٌ" \\
    -F "audio=@clip.wav" \\
    -F "mode=irab"

Example (two-step: generate, correct, then assess):
  curl "http://localhost:8000/phonemes?text=كتاب"
  # → {"text":"كتاب","phonemes":["k","i","t","aa","b","i"],"source":"g2p"}
  # ... user edits ...
  curl -X POST http://localhost:8000/assess \\
    -F "target=k,i,t,aa,b" \\
    -F "audio=@test_audio/clip.wav"
"""

import argparse
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from pronunciation_engine import PronunciationEngine
from backend_registry import list_modes, list_backends, resolve_backend

# Engine cache: key is (mode, backend) tuple, value is a PronunciationEngine.
# The default engine is built at startup; others are built lazily on first use.
_engines: dict[tuple, PronunciationEngine] = {}
DEFAULT_MODE = "everyday"
DEFAULT_BACKEND = None  # None → mode's default backend


app = FastAPI(title="Say It Right API", version="1.2")


def get_engine(mode: str = None, backend: str = None) -> PronunciationEngine:
    """Return a cached engine for (mode, backend), building it if needed.

    The first call (at startup) builds the default engine. Subsequent calls
    with different mode/backend combos lazily build and cache additional
    engines. Loading is expensive (multiple GB), so we never rebuild.
    """
    # Fall back to defaults if not specified
    mode = mode or DEFAULT_MODE
    backend = backend or DEFAULT_BACKEND

    # Validate the combo against the registry before building
    try:
        resolved = resolve_backend(mode, backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    key = (mode, resolved)
    if key not in _engines:
        print(f"[api] Building new engine for mode='{mode}' backend='{resolved}'...")
        _engines[key] = PronunciationEngine(
            mode=mode, backend=resolved, verbose=True
        )
    return _engines[key]


@app.on_event("startup")
def load_default_engine():
    """Pre-load the default engine at startup so the first request is fast."""
    print(f"[api] Loading default engine (mode={DEFAULT_MODE})...")
    get_engine(DEFAULT_MODE, DEFAULT_BACKEND)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engines_loaded": len(_engines),
        "default_mode": DEFAULT_MODE,
        "default_backend": resolve_backend(DEFAULT_MODE, DEFAULT_BACKEND),
        "cached": [
            {"mode": m, "backend": b} for (m, b) in _engines.keys()
        ],
    }


@app.get("/modes")
def get_modes():
    """List all available modes and backends."""
    modes = list_modes()
    backends = list_backends()
    return {
        "modes": modes,
        "backends": {
            name: {
                "modes": info["modes"],
                "default_for": info.get("default_for"),
                "description": info["description"],
                "dependencies": info["dependencies"],
            }
            for name, info in backends.items()
        },
    }


def _parse_target_string(s):
    """Parse 'k,i,t,aa,b' or 'k i t aa b' into a list."""
    if not s:
        return []
    parts = s.replace(",", " ").split()
    return [p.strip() for p in parts if p.strip()]


@app.get("/phonemes")
def get_phonemes(
    text: str = Query(..., description="Arabic text"),
    raw: bool = Query(False, description="If true, return only the bare phoneme array"),
    mode: str = Query(None, description="Assessment mode override (e.g. 'irab')"),
    backend: str = Query(None, description="Explicit ASR backend override"),
):
    """Get target phonemes for a text.

    Checks phrasebook first, falls back to CAMeL G2P.
    Useful for the two-step workflow: generate → user edits → assess.

    With raw=true, returns just the array: ["k","i","t","aa","b"]
    """
    eng = get_engine(mode, backend)
    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    phonemes, source = eng.get_target_phonemes(text)
    if raw:
        return JSONResponse(content=phonemes)
    return {
        "text": text,
        "phonemes": phonemes,
        "source": source,
        "mode": eng.mode,
        "backend": eng.backend_name,
    }


@app.post("/phonemes")
async def save_phonemes(
    text: str = Form(..., description="Arabic text"),
    phonemes: str = Form(..., description="Phoneme list, e.g. 'k,i,t,aa,b'"),
):
    """Save a corrected target phoneme list to the phrasebook.

    Phrasebook is shared across all modes — the phoneme list is what matters,
    not which engine produced it. So no mode/backend params here.
    """
    eng = get_engine()  # any engine is fine — phrasebook is shared
    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    parsed = _parse_target_string(phonemes)
    if not parsed:
        raise HTTPException(status_code=400, detail="phonemes cannot be empty")

    eng.save_target(text, parsed)
    return {"status": "saved", "text": text, "phonemes": parsed}


@app.post("/assess")
async def assess(
    audio: UploadFile = File(..., description="WAV audio file"),
    text: str = Form(None, description="Arabic target text (required if target not given)"),
    target: str = Form(None, description="Manual target phonemes, e.g. 'k,i,t,aa,b' (skips G2P)"),
    mode: str = Form(None, description="Assessment mode override (e.g. 'irab')"),
    backend: str = Form(None, description="Explicit ASR backend override"),
):
    """Assess pronunciation from uploaded audio.

    Two target sources:
      1. text=كتاب            → engine runs G2P (or phrasebook lookup)
      2. target=k,i,t,aa,b    → skips G2P, uses the provided phoneme list

    Either text or target must be provided. If both are given, target wins.

    Mode/backend: omit to use the server's default engine. Pass mode=irab
    (or mode=irab&backend=tarteel) to assess against a different engine.
    First use of a new mode/backend is slow (model loading); subsequent
    requests are fast (cached).
    """
    eng = get_engine(mode, backend)

    parsed_target = _parse_target_string(target) if target else None

    if not parsed_target and not (text and text.strip()):
        raise HTTPException(
            status_code=400,
            detail="either 'text' or 'target' must be provided",
        )

    # Save uploaded audio to temp file
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if parsed_target:
            # Manual target path
            result = eng.assess_with_target(
                parsed_target, tmp_path, text=text or "", verbose=False
            )
        else:
            # Standard path — G2P or phrasebook lookup
            result = eng.assess(text, tmp_path, verbose=False)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/", response_class=HTMLResponse)
def index():
    """Simple browser UI for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Say It Right</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
            input, button, select { font-size: 16px; margin: 4px 0; }
            #text { width: 100%; padding: 8px; font-size: 20px; }
            #target { width: 100%; padding: 8px; font-family: monospace; }
            #audio { width: 100%; }
            button { padding: 10px 20px; cursor: pointer; margin-right: 8px; }
            .hint { color: #666; font-size: 13px; margin-bottom: 8px; }
            #result { margin-top: 20px; white-space: pre-wrap;
                      background: #f0f0f0; padding: 16px; border-radius: 8px; }
            .mode-row { display: flex; gap: 12px; align-items: center;
                        margin-top: 12px; margin-bottom: 8px; }
            .mode-row label { font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Say It Right</h2>

        <div class="mode-row">
          <label>Mode:</label>
          <select id="mode" onchange="onModeChange()">
            <option value="">(server default)</option>
          </select>
          <label>Backend:</label>
          <select id="backend" onchange="onBackendChange()">
            <option value="">(mode default)</option>
          </select>
        </div>
        <div class="hint" id="mode-hint"></div>

        <label>Arabic text:</label>
        <input type="text" id="text" placeholder="كتاب" dir="rtl">
        <div class="hint">Enter the word/phrase, then click "Get Phonemes" to auto-generate the target.</div>

        <button onclick="getPhonemes()">Get Phonemes</button>

        <label style="margin-top:12px; display:block;">Target phonemes (editable):</label>
        <input type="text" id="target" placeholder="k i t aa b">
        <div class="hint">Auto-generated by CAMeL G2P. Edit to remove case endings (e.g. trailing 'i').
        Leave empty to use text directly.</div>

        <button onclick="savePhonemes()">Save to Phrasebook</button>

        <label style="margin-top:12px; display:block;">Audio file (WAV):</label>
        <input type="file" id="audio" accept=".wav">

        <div style="margin-top:12px;">
          <button onclick="assess()">Assess</button>
        </div>
        <div id="result"></div>

        <script>
        // Fetch modes/backends on page load and populate dropdowns
        async function loadModes() {
            try {
                const res = await fetch('/modes');
                const data = await res.json();
                const modeSel = document.getElementById('mode');
                const backendSel = document.getElementById('backend');
                data.modes.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    modeSel.appendChild(opt);
                });
                Object.keys(data.backends).forEach(b => {
                    const opt = document.createElement('option');
                    opt.value = b;
                    opt.textContent = b + ' (' + data.backends[b].modes.join(', ') + ')';
                    backendSel.appendChild(opt);
                });
            } catch (e) {
                console.error('Failed to load modes:', e);
            }
        }
        loadModes();

        function onModeChange() {
            const mode = document.getElementById('mode').value;
            const hint = document.getElementById('mode-hint');
            if (mode === 'irab') {
                hint.textContent = 'irab: harakat preserved (case endings matter). Backend: Tarteel Whisper.';
            } else if (mode === 'everyday') {
                hint.textContent = 'everyday: harakat stripped (case endings optional). Backend: wav2vec2.';
            } else {
                hint.textContent = '';
            }
        }
        function onBackendChange() { /* no-op, used for future hints */ }

        function getSelectedMode() { return document.getElementById('mode').value; }
        function getSelectedBackend() { return document.getElementById('backend').value; }

        async function getPhonemes() {
            const text = document.getElementById('text').value.trim();
            if (!text) { alert('Enter Arabic text first'); return; }
            document.getElementById('result').textContent = 'Generating phonemes...';
            try {
                let url = '/phonemes?text=' + encodeURIComponent(text);
                const mode = getSelectedMode();
                const backend = getSelectedBackend();
                if (mode) url += '&mode=' + encodeURIComponent(mode);
                if (backend) url += '&backend=' + encodeURIComponent(backend);
                const res = await fetch(url);
                const data = await res.json();
                document.getElementById('target').value = data.phonemes.join(' ');
                document.getElementById('result').textContent =
                    'Source: ' + data.source +
                    (data.mode ? '  |  Mode: ' + data.mode : '') +
                    (data.backend ? '  |  Backend: ' + data.backend : '') +
                    '\\nPhonemes: ' + JSON.stringify(data.phonemes);
            } catch (e) {
                document.getElementById('result').textContent = 'Error: ' + e;
            }
        }

        async function savePhonemes() {
            const text = document.getElementById('text').value.trim();
            const target = document.getElementById('target').value.trim();
            if (!text || !target) { alert('Need both text and target phonemes'); return; }
            const fd = new FormData();
            fd.append('text', text);
            fd.append('phonemes', target);
            try {
                const res = await fetch('/phonemes', { method: 'POST', body: fd });
                const data = await res.json();
                document.getElementById('result').textContent = 'Saved: ' + JSON.stringify(data);
            } catch (e) {
                document.getElementById('result').textContent = 'Error: ' + e;
            }
        }

        async function assess() {
            const text = document.getElementById('text').value.trim();
            const target = document.getElementById('target').value.trim();
            const audio = document.getElementById('audio').files[0];
            if (!audio) { alert('Please provide an audio file'); return; }
            if (!text && !target) { alert('Provide text or target phonemes'); return; }

            const fd = new FormData();
            fd.append('audio', audio);
            if (target) {
                fd.append('target', target);
            } else {
                fd.append('text', text);
            }
            const mode = getSelectedMode();
            const backend = getSelectedBackend();
            if (mode) fd.append('mode', mode);
            if (backend) fd.append('backend', backend);

            document.getElementById('result').textContent = 'Assessing' +
                (mode ? ' (mode=' + mode + ')' : '') + '...';
            try {
                const res = await fetch('/assess', { method: 'POST', body: fd });
                const data = await res.json();
                document.getElementById('result').textContent =
                    JSON.stringify(data, null, 2);
            } catch (e) {
                document.getElementById('result').textContent = 'Error: ' + e;
            }
        }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say It Right API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--mode", default=None,
        choices=list_modes(),
        help="Default assessment mode (default: everyday).",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Default ASR backend (default: mode's default).",
    )
    args = parser.parse_args()

    # Apply the CLI-selected defaults (override the module-level constants)
    if args.mode:
        DEFAULT_MODE = args.mode
    if args.backend:
        DEFAULT_BACKEND = args.backend

    print(f"Starting API server on http://{args.host}:{args.port}")
    print(f"  Default mode:    {DEFAULT_MODE}")
    print(f"  Default backend: {resolve_backend(DEFAULT_MODE, DEFAULT_BACKEND)}")
    print(f"  Browser UI:      http://localhost:{args.port}/")
    print(f"  Modes list:      http://localhost:{args.port}/modes")
    print(f"  Health:          http://localhost:{args.port}/health")
    print(f"  Phonemes:        GET  http://localhost:{args.port}/phonemes?text=كتاب")
    print(f"  Assess:          POST http://localhost:{args.port}/assess")
    uvicorn.run(app, host=args.host, port=args.port)
