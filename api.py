#!/usr/bin/env python3
"""
Say It Right - API Server

Endpoints:
  POST /assess     - Upload audio + text (or target), get pronunciation score
  GET  /phonemes   - Get target phonemes for a text (phrasebook or G2P)
  POST /phonemes   - Save a corrected target to the phrasebook
  GET  /health     - Health check
  GET  /           - Simple HTML test page (browser UI)

Usage:
  python api.py                   # starts on http://0.0.0.0:8000
  python api.py --port 9000       # custom port
  python api.py --host 127.0.0.1  # localhost only

Example (classic — text only, engine runs G2P):
  curl -X POST http://localhost:8000/assess \
    -F "text=كتاب" \
    -F "audio=@test_audio/clip.wav"

Example (manual target — skip G2P entirely):
  curl -X POST http://localhost:8000/assess \
    -F "target=k,i,t,aa,b" \
    -F "audio=@test_audio/clip.wav"

Example (two-step: generate, correct, then assess):
  curl "http://localhost:8000/phonemes?text=كتاب"
  # → {"text":"كتاب","phonemes":["k","i","t","aa","b","i"],"source":"g2p"}
  # ... user edits ...
  curl -X POST http://localhost:8000/assess \
    -F "target=k,i,t,aa,b" \
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

# Global engine - loaded once at startup
engine: PronunciationEngine = None

app = FastAPI(title="Say It Right API", version="1.1")


@app.on_event("startup")
def load_engine():
    global engine
    engine = PronunciationEngine(verbose=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


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
):
    """Get target phonemes for a text.

    Checks phrasebook first, falls back to CAMeL G2P.
    Useful for the two-step workflow: generate → user edits → assess.

    With raw=true, returns just the array: ["k","i","t","aa","b"]
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    phonemes, source = engine.get_target_phonemes(text)
    if raw:
        return JSONResponse(content=phonemes)
    return {"text": text, "phonemes": phonemes, "source": source}


@app.post("/phonemes")
async def save_phonemes(
    text: str = Form(..., description="Arabic text"),
    phonemes: str = Form(..., description="Phoneme list, e.g. 'k,i,t,aa,b'"),
):
    """Save a corrected target phoneme list to the phrasebook."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    parsed = _parse_target_string(phonemes)
    if not parsed:
        raise HTTPException(status_code=400, detail="phonemes cannot be empty")

    engine.save_target(text, parsed)
    return {"status": "saved", "text": text, "phonemes": parsed}


@app.post("/assess")
async def assess(
    audio: UploadFile = File(..., description="WAV audio file"),
    text: str = Form(None, description="Arabic target text (required if target not given)"),
    target: str = Form(None, description="Manual target phonemes, e.g. 'k,i,t,aa,b' (skips G2P)"),
):
    """Assess pronunciation from uploaded audio.

    Two modes:
      1. text=كتاب            → engine runs G2P (or phrasebook lookup)
      2. target=k,i,t,aa,b    → skips G2P, uses the provided phoneme list

    Either text or target must be provided. If both are given, target wins.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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
            result = engine.assess_with_target(
                parsed_target, tmp_path, text=text or "", verbose=False
            )
        else:
            # Standard path — G2P or phrasebook lookup
            result = engine.assess(text, tmp_path, verbose=False)
        return JSONResponse(content=result)
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
            input, button { font-size: 16px; margin: 4px 0; }
            #text { width: 100%; padding: 8px; font-size: 20px; }
            #target { width: 100%; padding: 8px; font-family: monospace; }
            #audio { width: 100%; }
            button { padding: 10px 20px; cursor: pointer; margin-right: 8px; }
            .hint { color: #666; font-size: 13px; margin-bottom: 8px; }
            #result { margin-top: 20px; white-space: pre-wrap;
                      background: #f0f0f0; padding: 16px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h2>Say It Right</h2>

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
        async function getPhonemes() {
            const text = document.getElementById('text').value.trim();
            if (!text) { alert('Enter Arabic text first'); return; }
            document.getElementById('result').textContent = 'Generating phonemes...';
            try {
                const url = '/phonemes?text=' + encodeURIComponent(text);
                const res = await fetch(url);
                const data = await res.json();
                document.getElementById('target').value = data.phonemes.join(' ');
                document.getElementById('result').textContent =
                    'Source: ' + data.source + '\\nPhonemes: ' + JSON.stringify(data.phonemes);
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
            document.getElementById('result').textContent = 'Assessing...';
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
    args = parser.parse_args()

    print(f"Starting API server on http://{args.host}:{args.port}")
    print(f"  Browser UI:  http://localhost:{args.port}/")
    print(f"  Health:      http://localhost:{args.port}/health")
    print(f"  Phonemes:    GET  http://localhost:{args.port}/phonemes?text=كتاب")
    print(f"  Assess:      POST http://localhost:{args.port}/assess")
    uvicorn.run(app, host=args.host, port=args.port)
