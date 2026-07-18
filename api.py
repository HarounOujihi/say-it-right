#!/usr/bin/env python3
"""
Say It Right - API Server

Endpoints:
  POST /assess    - Upload audio + text, get pronunciation score
  GET  /health    - Health check
  GET  /          - Simple HTML test page (browser UI)

Usage:
  python api.py                   # starts on http://0.0.0.0:8000
  python api.py --port 9000       # custom port
  python api.py --host 127.0.0.1  # localhost only

Example:
  curl -X POST http://localhost:8000/assess \
    -F "text=كتاب" \
    -F "audio=@test_audio/clip.wav"
"""

import argparse
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from pronunciation_engine import PronunciationEngine

# Global engine - loaded once at startup
engine: PronunciationEngine = None

app = FastAPI(title="Say It Right API", version="1.0")


@app.on_event("startup")
def load_engine():
    global engine
    engine = PronunciationEngine(verbose=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/assess")
async def assess(
    text: str = Form(..., description="Arabic target text"),
    audio: UploadFile = File(..., description="WAV audio file"),
):
    """Assess pronunciation from uploaded audio + target text."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    # Save uploaded audio to temp file
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
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
            body { font-family: sans-serif; max-width: 600px; margin: 40px auto; }
            input, button { font-size: 16px; margin: 8px 0; }
            #text { width: 100%; padding: 8px; font-size: 20px; }
            #audio { width: 100%; }
            button { padding: 10px 20px; cursor: pointer; }
            #result { margin-top: 20px; white-space: pre-wrap;
                      background: #f0f0f0; padding: 16px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h2>Say It Right</h2>
        <input type="text" id="text" placeholder="Arabic text (e.g. كتاب)" dir="rtl">
        <br>
        <input type="file" id="audio" accept=".wav">
        <br>
        <button onclick="assess()">Assess</button>
        <div id="result"></div>
        <script>
        async function assess() {
            const text = document.getElementById('text').value;
            const audio = document.getElementById('audio').files[0];
            if (!text || !audio) {
                alert('Please provide both text and audio file');
                return;
            }
            const fd = new FormData();
            fd.append('text', text);
            fd.append('audio', audio);
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
    print(f"  Assess:      POST http://localhost:{args.port}/assess")
    uvicorn.run(app, host=args.host, port=args.port)
