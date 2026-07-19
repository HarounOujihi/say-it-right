# Say It Right

Arabic pronunciation assessment with a **pluggable backend architecture**. Swap ASR models in and out without touching the assessment logic, CLI, or API.

Compare expected phonemes (from text) against recognized phonemes (from audio) and get an accuracy score plus a categorized list of errors (substitutions / insertions / deletions).

---

## The Idea: Modes and Backends

Two orthogonal concepts control how an assessment runs:

- **Mode** — *what kind of pronunciation we're checking.* Defines how text is preprocessed (e.g. should harakat be preserved or stripped?).
- **Backend** — *which ASR model turns audio into phonemes.* Multiple backends can serve the same mode; you pick one per request.

| Mode       | What it checks                                          | Harakat   | Default backend |
|------------|---------------------------------------------------------|-----------|-----------------|
| `everyday` | Casual speech — case endings (i`rāb) optional           | stripped  | `wav2vec2`      |
| `irab`     | Diacritized / formal speech — case endings matter       | preserved | `tarteel`       |

You can override the backend per-request. Example: even though `tarteel` is the default for `irab`, you could register a second backend (say `nemo`) for the same mode and pick it explicitly:

```bash
python cli.py --text "كِتَابٌ" --mode irab --backend nemo
```

The CLI, API, and test runner all honor `--mode` / `--backend`. New backends plug in through a single registry file — see [Adding a New Backend](#adding-a-new-backend) below.

---

## Quick Start

```bash
# 1. Clone
git clone git@github.com:HarounOujihi/say-it-right.git
cd say-it-right

# 2. Create + activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install CAMeL Tools data (~36 MB)
camel_data -i light

# 5. Run the interactive CLI (everyday mode by default)
python cli.py

# Or pick a mode up front:
python cli.py --text "كِتَابٌ" --mode irab --audio clip.wav
```

First run downloads the active backend's model from HuggingFace Hub into `~/.cache/huggingface/`. To avoid that, see [Installation](#installation-fresh-setup).

---

## How It Works

```
                         ┌── phrasebook.json (if entry exists) ──┐
Arabic Text ──┼──────────┤                                        ├──► Target Phonemes ──┐
              │          └── CAMeL Tools MLE (G2P fallback) ───────┘                      │
              │                                                                          ├──► Edit Distance ──► Accuracy %
              │                                                       Target Phonemes ──┘
              │                                                       (user-corrected)
Audio WAV ────┼──► [pluggable backend] ──► Recognized Phonemes ─────────────────────────┘
                   wav2vec2 / tarteel /
                   <your model here>
```

1. **Text → phonemes (G2P).** Arabic text is morphologically analyzed via CAMeL Tools MLE Disambiguator (`calima-msa-r13`), producing phonemes from the `caphi` field. Example: `كتاب` → `k i t aa b i`.
2. **Audio → phonemes (ASR).** The active **backend** recognizes phonemes from recorded audio. Each backend is free to do this however it likes — direct CTC decoding, ASR + G2P, or anything else — as long as it returns a list of phoneme strings.
3. **Comparison.** The two phoneme lists are aligned using Levenshtein edit distance, classifying errors as substitutions, insertions, or deletions.

**Target resolution order:** phrasebook (user-corrected) → CAMeL G2P (auto-generated) → manual override (caller-provided). See [Manual Target & Phrasebook](#manual-target--phrasebook).

---

## Usage

### Interactive CLI

```bash
python cli.py                      # prompts for text + audio source
python cli.py --mode irab          # start in irab mode
```

### One-liner CLI

```bash
# Assess existing audio
python cli.py --text "كتاب" --audio test_audio/clip.wav

# Record 3 seconds, then assess
python cli.py --text "كتاب" --record 3

# Explicit mode + backend
python cli.py --text "كِتَابٌ" --mode irab --backend tarteel --audio clip.wav

# JSON output for scripting
python cli.py --text "كتاب" --audio clip.wav --json
```

### API Server

```bash
./start.sh api                        # http://localhost:8000
./start.sh api --port 9000            # custom port
./start.sh api --mode irab            # default server mode = irab
```

Open `http://localhost:8000/` for a small browser UI, or call the REST endpoint directly:

```bash
curl -X POST http://localhost:8000/assess \
  -F "text=كتاب" \
  -F "audio=@test_audio/clip.wav" \
  -F "mode=irab"           # optional per-request override
```

Response:
```json
{
  "text": "كتاب",
  "mode": "everyday",
  "backend": "wav2vec2",
  "accuracy": 83.33,
  "target_phonemes": ["k", "i", "t", "aa", "b", "i"],
  "recognized_phonemes": ["k", "i", "t", "aa", "b"],
  "errors": {
    "substitutions": [],
    "insertions": [],
    "deletions": ["i"]
  },
  "total_errors": 1,
  "target_source": "g2p"
}
```

Inspect available modes and backends:

```bash
curl http://localhost:8000/modes
# {
#   "modes": ["everyday", "irab"],
#   "backends": {
#     "wav2vec2": {"modes": ["everyday"], "default_for": "everyday", ...},
#     "tarteel":  {"modes": ["irab"],     "default_for": "irab",     ...}
#   }
# }
```

### Batch Tests

```bash
./start.sh test
```

---

## Adding a New Backend

This is the core extensibility point. The goal: try a new ASR model without forking the assessment logic, CLI, or API.

### The contract

Every backend is just **two methods on `PronunciationEngine`**:

| Method                       | Job                                                          | Required? |
|------------------------------|--------------------------------------------------------------|-----------|
| `_load_<name>(self, ...)`    | Load model weights, processor, etc. Store them on `self`.    | yes       |
| `_recognize_<name>(self, audio_path)` | Take a WAV path, return a list of phoneme strings.  | yes       |

That's it. The loader runs once during engine init; the recognizer runs once per assessment. Everything else (text normalization, G2P, edit distance, accuracy scoring, error breakdown) is shared infrastructure.

Shared helpers available inside any backend:

- `self._load_audio(path)` → `(float32 numpy array, sample_rate)` — handles mono conversion + 16 kHz resampling
- `self.device` — `torch.device("cuda" if available else "cpu")`
- `self.text_to_phonemes(text, preserve_harakat=...)` — run CAMeL G2P on Arabic text (handy for backends that output text rather than phonemes)
- `self._log(msg)` — print only if `verbose=True`

### Step-by-step: add a backend

**1. Register it** in `backend_registry.py`:

```python
BACKENDS = {
    # ...existing entries...

    "nemo": {
        "modes": ["irab"],            # which modes can use this
        "default_for": None,          # set to "irab" to make it the default
        "loader": "_load_nemo",       # method name on PronunciationEngine
        "recognizer": "_recognize_nemo",
        "dependencies": ["nemo_toolkit"],
        "description": "NVIDIA NeMo FastConformer diacritized ASR",
    },
}
```

**2. Implement the two methods** in `pronunciation_engine.py`:

```python
def _load_nemo(self, **kwargs):
    """Load NeMo model once at engine init."""
    # Import heavy deps lazily so unrelated backends don't pay the cost
    import nemo_toolkit.asr as nemo_asr

    self._log("Loading NeMo FastConformer...")
    self.nemo_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/ar_fastconformer_transducer_large"
    )
    self.nemo_model = self.nemo_model.to(self.device)
    self.nemo_model.eval()

def _recognize_nemo(self, audio_path):
    """Audio → diacritized text → CAMeL G2P → phonemes."""
    try:
        audio_data, sample_rate = self._load_audio(audio_path)

        # NeMo wants a file path; write the resampled audio to a temp wav
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            tmp_path = tmp.name
        try:
            text = self.nemo_model.transcribe([tmp_path])[0].text.strip()
        finally:
            os.unlink(tmp_path)

        if not text:
            return []

        # Same trick as the tarteel backend: route through CAMeL G2P so
        # the recognized phonemes use the SAME notation as the target.
        return self.text_to_phonemes(text, preserve_harakat=True)
    except Exception as e:
        print(f"Audio recognition error (nemo): {e}")
        return []
```

**3. Use it.** No other files need to change — CLI, API, and tests pick up the new backend automatically:

```bash
# CLI
python cli.py --text "كِتَابٌ" --mode irab --backend nemo --audio clip.wav

# API
curl -X POST http://localhost:8000/assess \
  -F "text=كِتَابٌ" \
  -F "audio=@clip.wav" \
  -F "mode=irab" \
  -F "backend=nemo"

# It also shows up in the introspection endpoint:
curl http://localhost:8000/modes
python cli.py --list-backends
```

### Design notes for backend authors

- **Output notation matters.** Both target and recognized phonemes go through the same edit-distance comparison, so they must use the same phoneme inventory. Two safe options:
  - Output raw phonemes in **CAMeL `caphi` notation** (matching what CAMeL G2P produces for the target).
  - Output Arabic text and call `self.text_to_phonemes(text)` — the engine runs it through CAMeL, guaranteeing a matching notation. The `tarteel` backend does this.
- **Fail soft, fail loud.** Wrap the recognition body in `try/except` and return `[]` on failure (so one bad audio file doesn't crash a batch run), but `print()` the underlying error so it's visible.
- **Lazy imports.** Put `import` statements for backend-specific libraries inside the loader method. This keeps `pip install -r requirements.txt` light and means adding a backend doesn't force every user to install its deps.
- **No model reloads.** Load weights once in `_load_<name>`; the engine caches the instance across requests. The API server also caches one `PronunciationEngine` per `(mode, backend)` tuple.

### Roadmap: candidates we haven't wired in yet

The registry has commented-out slots for several candidates we've been evaluating. Each is one PR away:

| Candidate   | Approach                                            | Notes |
|-------------|-----------------------------------------------------|-------|
| `aras2p`    | Purpose-built phoneme mispronunciation detector     | Direct phoneme output, no G2P detour |
| `nemo`      | NVIDIA NeMo FastConformer diacritized ASR           | Fast GPU inference, high quality |
| `whisper-t` | OpenAI Whisper large-v3 + Arabic linguistics        | General ASR, would need a diacritization step |

---

## Built-in Backends

### `wav2vec2` — phoneme-level ASR (default for `everyday`)

- **Model:** `MostafaMaroof/wav2vec2-arabic-phoneme-asr` (~1.2 GB, safetensors)
- **Pipeline:** raw audio → wav2vec2 CTC → phoneme tokens (direct, no G2P detour)
- **Output:** phonemes in the model's own vocabulary, which happens to align with CAMeL `caphi`
- **Latency:** ~2 s per 3 s clip on CPU
- **Local cache:** drop the weights into `model_cache/` and the engine auto-detects them

### `tarteel` — diacritized ASR (default for `irab`)

- **Model:** `tarteel-ai/whisper-base-ar-quran` (Whisper fine-tuned on Quranic Arabic)
- **Pipeline:** raw audio → Whisper → diacritized Arabic text → CAMeL G2P → phonemes
- **Output:** phonemes in CAMeL `caphi` notation (guaranteed match with the target)
- **Why the G2P detour?** Whisper outputs Arabic *text*, not phonemes. Running that text through CAMeL G2P produces phonemes in the same vocabulary the target uses, eliminating the apples-to-oranges comparison problem.
- **Compatibility note:** the model ships with an outdated `generation_config`. We call `generate()` without `language=`/`task=` kwargs (the model is already fine-tuned for Arabic) and manually strip the `<|ar|><|transcribe|><|notimestamps|>` prefix tokens. See `_recognize_tarteel` in `pronunciation_engine.py`.

---

## Manual Target & Phrasebook

CAMeL's G2P sometimes produces phonemes that don't match natural speech — most commonly **case endings** (i`rāb). For example, `كتاب` (kitāb) is spoken as `k i t aa b`, but CAMeL outputs `k i t aa b i` (with a trailing genitive `i`).

Say It Right lets you override the target phonemes in three ways:

### Option A — Pass a manual target (one-off)

```bash
# CLI
python cli.py --audio clip.wav --target "k,i,t,aa,b"

# API
curl -X POST http://localhost:8000/assess \
  -F "target=k,i,t,aa,b" \
  -F "audio=@test_audio/clip.wav"
```

### Option B — Edit interactively, save to phrasebook

```bash
python cli.py --text "كتاب" --audio clip.wav --edit-target --save-target
```

The corrected list is persisted to `phrasebook.json`:

```json
{
  "كتاب": ["k", "i", "t", "aa", "b"]
}
```

### Option C — Two-step via API (for web/mobile clients)

```bash
# Step 1: generate
curl "http://localhost:8000/phonemes?text=كتاب"
# → {"phonemes":["k","i","t","aa","b","i"],"source":"g2p"}

# Step 2: user edits on the client side, then saves
curl -X POST http://localhost:8000/phonemes \
  -F "text=كتاب" -F "phonemes=k,i,t,aa,b"
# → {"status":"saved"}

# Subsequent /assess calls with text=كتاب now auto-use the phrasebook entry
```

### How phrasebook lookup works

Every `assess(text=...)` call checks `phrasebook.json` **first**. If the text is found, the saved phoneme list is used and G2P is skipped entirely. The response includes `"target_source"` so you know which path was taken:

| `target_source` | Meaning |
|-----------------|---------|
| `"g2p"` | CAMeL MLE generated the target (no phrasebook entry) |
| `"phrasebook"` | Used a previously saved correction from `phrasebook.json` |
| `"manual"` | Caller passed an explicit `target` list this time |

The browser UI at `http://localhost:8000/` has a **"Get Phonemes"** button to auto-generate, an editable field to correct, and a **"Save to Phrasebook"** button to persist.

---

## Recording Audio

```bash
./start.sh record test_audio/clip.wav 3    # record 3 seconds
```

Or use any external tool. Audio requirements:
- Format: WAV
- Sample rate: any (auto-resampled to 16 kHz)
- Channels: mono or stereo (auto-converted)

To enable in-app recording:

```bash
pip install sounddevice
# Fedora:
sudo dnf install portaudio-devel
```

If your recordings are silent, check the OS microphone state:

```bash
wpctl status          # look for "MUTED" next to your microphone source
wpctl set-mute <ID> 0 # unmute by ID
```

---

## Technical Details

### CAMeL Tools — MLE Disambiguator (`calima-msa-r13`)

A Maximum Likelihood Estimation morphological disambiguator for Modern Standard Arabic, part of the [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) library.

Earlier versions of CAMeL Tools (pre-1.5) included a dedicated `G2P` class. In 1.6.0 it no longer exists; instead, G2P is done through morphological disambiguation. The disambiguator returns a rich analysis per word including a `caphi` (CAMeL Phonetic) field:

```python
mle = MLEDisambiguator.pretrained('calima-msa-r13')
results = mle.disambiguate(['كتاب'])  # → "book"

# analysis dict contains:
#   'caphi': 'k_i_t_aa_b_i'     ← phonemes separated by underscores
#   'gloss': 'book+[def.gen.]'
#   'pos':   'noun'
#   'root':  'ktb'

phonemes = results[0].analyses[0].analysis['caphi'].split('_')
# → ['k', 'i', 't', 'aa', 'b', 'i']
```

The `caphi` inventory:
- Short vowels: `a`, `i`, `u`
- Long vowels: `aa`, `ii`, `uu`
- Emphatic consonants distinct from non-emphatic (e.g. `t` vs `T`)
- Pharyngeal consonants: `2` (hamza), `3` (ayn), `7` (ha), `kh`, `gh`, `q`

Requires `camel_data -i light` (~36 MB database).

### Audio Loading — Why not torchaudio?

We use `soundfile` (libsndfile backend) instead of `torchaudio.load()`. Torchaudio 2.11+ switched its default backend to `torchcodec`, a separate package requiring extra system deps. `soundfile` was already a dependency (recording helper) and handles WAV reading perfectly:

```python
audio_data, sample_rate = sf.read(audio_path)
```

Resampling to 16 kHz (when needed) uses `numpy.interp`, sufficient for speech-quality audio.

### Comparison — Levenshtein Edit Distance

Dynamic programming (Wagner-Fischer) computes the minimum edit distance between target and recognized phoneme sequences, then backtracks through the DP table to classify each edit:

| Error type       | Meaning                    | Example                              |
|------------------|----------------------------|--------------------------------------|
| **Substitution** | Wrong phoneme spoken       | target `aa` → recognized `a`         |
| **Insertion**    | Extra phoneme added        | recognized `s` not in target         |
| **Deletion**     | Expected phoneme missing   | target `i` not found in audio        |

Accuracy = `(1 - total_errors / len(target_phonemes)) * 100`

---

## Architecture

```
pronunciation_engine.py   ← Core logic (PronunciationEngine class)
                              ├─ _load_<name>()         — one per backend (lazy init)
                              ├─ _recognize_<name>()    — one per backend (audio → phonemes)
                              ├─ audio_to_phonemes()    — dispatcher, routes to active backend
                              ├─ text_to_phonemes()     — CAMeL MLE → caphi
                              ├─ get_target_phonemes()  — phrasebook → G2P fallback
                              ├─ assess_with_target()   — manual target (skip G2P)
                              ├─ compare_phonemes()     — edit distance
                              └─ assess()               — full pipeline
│
├── backend_registry.py   ← Backend routing (add new backends here)
│                             ├─ BACKENDS dict           — registry of all backends
│                             ├─ resolve_backend()       — mode + override → backend name
│                             └─ list_modes/backends()   — introspection
│
├── phrasebook.py         ← JSON-backed user corrections (load/save/lookup)
├── cli.py                ← Interactive + one-liner CLI (--mode, --backend, --target, ...)
├── api.py                ← FastAPI server (/assess, /phonemes, /modes, browser UI)
├── test_pronunciation.py ← Batch test runner
└── record_audio.py       ← Microphone recording helper
```

All three interfaces (CLI, API, batch test) import the same `PronunciationEngine`, so adding a backend in `backend_registry.py` + `pronunciation_engine.py` instantly exposes it everywhere.

### Engine caching in the API server

The API server keeps one `PronunciationEngine` per `(mode, backend)` tuple. The default engine loads at startup; other combos build lazily on first request and stay cached. Loading is expensive (multiple GB into RAM), so engines are never rebuilt.

---

## Project Structure

```
say-it-right/
├── pronunciation_engine.py   # Core assessment logic + backend implementations
├── backend_registry.py       # Backend routing (the "plugin" registry)
├── phrasebook.py             # User-corrected phoneme lookup/save
├── cli.py                    # Interactive + one-liner CLI
├── api.py                    # FastAPI server with browser UI
├── test_pronunciation.py     # Batch test runner
├── record_audio.py           # Audio recording helper
├── start.sh                  # Unified entry point
├── stop.sh                   # Stop any running process
├── requirements.txt          # Python dependencies
├── model_cache/              # wav2vec2 model weights (~1.2 GB)  [gitignored]
├── test_audio/               # Place WAV files here              [gitignored]
├── results/                  # JSON output from tests            [gitignored]
├── phrasebook.json           # User-corrected phoneme targets    [gitignored]
└── venv/                     # Python virtual environment        [gitignored]
```

---

## Installation (Fresh Setup)

```bash
# 1. Clone
git clone git@github.com:HarounOujihi/say-it-right.git
cd say-it-right

# 2. Make scripts executable (Git doesn't always preserve the +x bit)
chmod +x start.sh stop.sh

# 3. Create + activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install CAMeL Tools data (~36 MB, stored in ~/.camel_tools)
camel_data -i light

# 6. Create runtime directories
mkdir -p test_audio results

# 7. Download the wav2vec2 model (~1.2 GB)
#    Option A — let the engine auto-download on first run (→ ~/.cache/huggingface/)
#    Option B — pre-download into model_cache/ (auto-detected, skips HF Hub):
mkdir -p model_cache && cd model_cache
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/model.safetensors
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/processor_config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/tokenizer_config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/vocab.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/added_tokens.json
cd ..
```

If `curl` stalls, add `--retry 30 --retry-delay 5 --retry-all-errors -C -` to resume.

### System Requirements (Fedora)

```bash
sudo dnf install -y python3-devel      # for compiling C extensions (editdistance, etc.)
sudo dnf install -y portaudio-devel    # for audio recording (optional)
```

---

## Dependencies

| Package           | Purpose                                              |
|-------------------|------------------------------------------------------|
| `camel-tools`     | Arabic morphological analysis / G2P via `caphi`      |
| `transformers`    | wav2vec2 / Whisper model loading and inference       |
| `torch`           | Neural network backend (CPU inference)               |
| `soundfile`       | WAV file reading (replaces torchaudio)               |
| `fastapi`         | API server                                           |
| `uvicorn`         | ASGI server for FastAPI                              |
| `python-multipart`| File upload support for API                          |
| `sounddevice`     | Microphone recording (optional)                      |
| `numpy`           | Audio resampling and numeric operations              |

---

## Example Output

```
Assessing pronunciation for: 'كتاب'
Audio file: test_audio/cli_recording.wav
Mode: everyday  |  Backend: wav2vec2

==================================================
ASSESSMENT RESULTS
==================================================
Accuracy: 83.33%
Correct phonemes: 5/6
Errors: 1
  - Substitutions: 0
  - Insertions: 0
  - Deletions: 1
      ['i']
==================================================
```
