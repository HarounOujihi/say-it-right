# Say It Right

Arabic pronunciation assessment tool. Compare expected phonemes (from text) against recognized phonemes (from audio) using CAMeL Tools for grapheme-to-phoneme conversion and wav2vec2 for speech recognition.

## Quick Start

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Run the interactive CLI
python cli.py
```

Or use the wrapper script:

```bash
./start.sh            # interactive CLI (default)
./start.sh api        # API server on port 8000
./start.sh test       # batch tests
```

## How It Works

```
Arabic Text ──► CAMeL Tools MLE ──► Target Phonemes ─┐
                   (G2P)              k i t aa b i    │
                                                      ├─► Edit Distance ──► Accuracy %
Audio WAV ────► wav2vec2 CTC ────► Recognized Phonemes┘
                                  k i t aa b
```

1. **Text to phonemes (G2P)** — Arabic text is morphologically analyzed via CAMeL Tools MLE Disambiguator (`calima-msa-r13`), producing phonemes from the `caphi` field (e.g. `كتاب` → `k i t aa b i`).
2. **Audio to phonemes (ASR)** — A wav2vec2 model (`MostafaMaroof/wav2vec2-arabic-phoneme-asr`) recognizes phonemes from the recorded audio using CTC decoding.
3. **Comparison** — The two phoneme lists are aligned using edit distance (Levenshtein), classifying errors as substitutions, insertions, or deletions.

## Usage Modes

### 1. Interactive CLI

Prompts for Arabic text and audio source (file or record):

```bash
./start.sh cli
# or
python cli.py
```

### 2. One-liner CLI

```bash
# Assess existing audio file
python cli.py --text "كتاب" --audio test_audio/clip.wav

# Record 3 seconds then assess
python cli.py --text "كتاب" --record 3

# JSON output (for scripts)
python cli.py --text "كتاب" --audio clip.wav --json
```

### 3. API Server

```bash
./start.sh api                 # http://localhost:8000
./start.sh api --port 9000     # custom port
```

**Browser UI:** Open `http://localhost:8000/` for a simple upload interface.

**REST endpoint:**

```bash
curl -X POST http://localhost:8000/assess \
  -F "text=كتاب" \
  -F "audio=@test_audio/clip.wav"
```

Response:
```json
{
  "text": "كتاب",
  "accuracy": 83.33,
  "target_phonemes": ["k", "i", "t", "aa", "b", "i"],
  "recognized_phonemes": ["k", "i", "t", "aa", "b"],
  "errors": {
    "substitutions": [],
    "insertions": [],
    "deletions": ["i"]
  },
  "total_errors": 1
}
```

### 4. Batch Test

Run multiple test cases defined in `test_pronunciation.py`:

```bash
./start.sh test
```

## Recording Audio

Use the built-in recorder:

```bash
./start.sh record test_audio/clip.wav 3    # record 3 seconds
```

Or any external tool. Audio requirements:
- Format: WAV
- Sample rate: any (auto-resampled to 16 kHz)
- Channels: mono or stereo (auto-converted)

**To enable recording**, install sounddevice:

```bash
pip install sounddevice
# Fedora:
sudo dnf install portaudio-devel
```

If your recordings are silent, check that the microphone is not muted at the OS level:

```bash
wpctl status          # look for "MUTED" next to your microphone source
wpctl set-mute <ID> 0 # unmute by ID
```

## Technical Details

### Models Used

#### 1. CAMeL Tools — MLE Disambiguator (`calima-msa-r13`)

**What it is:** A Maximum Likelihood Estimation morphological disambiguator for Modern Standard Arabic, part of the [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) library. It analyzes Arabic text morphologically and selects the most likely analysis for each word.

**Why this approach:** Earlier versions of CAMeL Tools (pre-1.5) included a dedicated `G2P` class that directly converted text to phonemes. In version 1.6.0 (the current release), this class no longer exists. Instead, G2P is achieved through morphological disambiguation: the disambiguator returns a rich analysis for each word, including a `caphi` (CAMeL Phonetic) field that contains the phonemic transcription.

**How phonemes are extracted:**

```python
mle = MLEDisambiguator.pretrained('calima-msa-r13')
results = mle.disambiguate(['كتاب'])  # → "book"

# The analysis dict contains:
#   'caphi': 'k_i_t_aa_b_i'     ← phonemes separated by underscores
#   'gloss': 'book+[def.gen.]'
#   'pos':   'noun'
#   'root':  'ktb'

# Split on '_' to get individual phonemes
phonemes = results[0].analyses[0].analysis['caphi'].split('_')
# → ['k', 'i', 't', 'aa', 'b', 'i']
```

The `caphi` representation uses a custom phoneme inventory where:
- Short vowels: `a`, `i`, `u`
- Long vowels: `aa`, `ii`, `uu`
- Emphatic consonants are distinct from non-emphatic (e.g. `t` vs `T`)
- Pharyngeal consonants: `2` (hamza), `3` (ayn), `7` (ha), `kh`, `gh`, `q`

**Data download:** Requires `camel_data -i light` which downloads the `disambig-mle-calima-msa-r13` database (~36 MB).

#### 2. Wav2Vec2 — Arabic Phoneme ASR (`MostafaMaroof/wav2vec2-arabic-phoneme-asr`)

**What it is:** A fine-tuned Wav2Vec2 model (originally from Facebook AI) trained specifically to recognize Arabic phonemes from raw audio waveforms. It uses CTC (Connectionist Temporal Classification) decoding, meaning it doesn't need pre-aligned audio-text pairs during training.

**Why this model:**
- Directly outputs phonemes (not Arabic script), making it ideal for pronunciation assessment
- Fine-tuned on Arabic speech data
- Lightweight (~1.2 GB) compared to larger ASR models
- Runs efficiently on CPU (~2 seconds per 3-second clip)

**Architecture:** Wav2Vec2 uses a multi-layer transformer encoder that processes raw audio at 16 kHz. The model was pre-trained on self-supervised speech representations (learning from unlabeled audio) then fine-tuned with CTC loss on labeled phoneme data. The output is a probability distribution over ~40 phoneme tokens for each time step, which is then collapsed (removing repeats and blanks) into the final phoneme sequence.

**Format:** The model is distributed in `safetensors` format (not `pytorch_model.bin`), which is faster to load and more secure than pickle-based formats.

### Audio Loading — Why not torchaudio?

This project uses `soundfile` (libsndfile backend) for audio I/O instead of `torchaudio.load()`. The reason:

Torchaudio 2.11+ changed its default backend to `torchcodec`, a separate package that requires additional system dependencies. Without `torchcodec` installed, `torchaudio.load()` raises `ImportError: TorchCodec is required`. Since `soundfile` was already a dependency (used by the recording helper) and handles WAV reading perfectly, we use it directly:

```python
# Instead of torchaudio.load():
audio_data, sample_rate = sf.read(audio_path)  # soundfile
```

Resampling to 16 kHz (when needed) is done with `numpy.interp`, which is sufficient for speech-quality audio and avoids pulling in `torchaudio.transforms.Resample` or `scipy.signal.resample`.

### Comparison Algorithm — Levenshtein Edit Distance

The phoneme comparison uses dynamic programming (Wagner-Fischer algorithm) to compute the minimum edit distance between the target and recognized phoneme sequences, then backtracks through the DP table to classify each edit as:

| Error type | Meaning | Example |
|-----------|---------|---------|
| **Substitution** | Wrong phoneme spoken | target `aa` → recognized `a` |
| **Insertion** | Extra phoneme added | recognized `s` not in target |
| **Deletion** | Expected phoneme missing | target `i` not found in audio |

Accuracy = `(1 - total_errors / len(target_phonemes)) * 100`

### Architecture

```
pronunciation_engine.py   ← Core logic (PronunciationEngine class)
                              ├─ text_to_phonemes()   — CAMeL MLE → caphi
                              ├─ audio_to_phonemes()  — soundfile + wav2vec2
                              ├─ compare_phonemes()   — edit distance
                              └─ assess()             — full pipeline
│
├── cli.py                ← Interactive + one-liner CLI (argparse)
├── api.py                ← FastAPI server + browser UI
├── test_pronunciation.py ← Batch test runner
└── record_audio.py       ← Microphone recording helper
```

All three interfaces (CLI, API, batch test) import and share the same `PronunciationEngine` class, ensuring identical assessment logic across modes.

## Project Structure

```
say-it-right/
├── pronunciation_engine.py   # Core assessment logic (shared)
├── cli.py                    # Interactive + one-liner CLI
├── api.py                    # FastAPI server with browser UI
├── test_pronunciation.py     # Batch test runner
├── record_audio.py           # Audio recording helper
├── start.sh                  # Unified entry point
├── stop.sh                   # Stop any running process
├── requirements.txt          # Python dependencies
├── model_cache/              # wav2vec2 model weights (local, ~1.2 GB)
├── test_audio/               # Place WAV files here
├── results/                  # JSON output from tests
└── venv/                     # Python virtual environment
```

## Installation (Fresh Setup)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install CAMeL Tools data
camel_data -i light

# 4. Download model (if not using HF Hub auto-download)
mkdir -p model_cache
cd model_cache
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/model.safetensors
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/processor_config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/tokenizer_config.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/vocab.json
curl -L -O https://huggingface.co/MostafaMaroof/wav2vec2-arabic-phoneme-asr/resolve/main/added_tokens.json
cd ..
```

### System Requirements (Fedora)

```bash
# For compiling C extensions (editdistance, etc.)
sudo dnf install -y python3-devel

# For audio recording (optional)
sudo dnf install -y portaudio-devel
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `camel-tools` | Arabic morphological analysis / G2P via `caphi` field |
| `transformers` | wav2vec2 model loading and inference |
| `torch` | Neural network backend (CPU inference) |
| `soundfile` | WAV file reading (replaces torchaudio) |
| `fastapi` | API server |
| `uvicorn` | ASGI server for FastAPI |
| `python-multipart` | File upload support for API |
| `sounddevice` | Microphone recording (optional) |
| `numpy` | Audio resampling and numeric operations |

## Example Output

```
Assessing pronunciation for: 'كتاب'
Audio file: test_audio/cli_recording.wav

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
