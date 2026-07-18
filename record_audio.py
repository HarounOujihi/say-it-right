#!/usr/bin/env python3
"""Record a short audio clip and save as 16 kHz mono WAV."""
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

# pip install sounddevice
# Fedora: sudo dnf install portaudio-devel  (needed to build sounddevice)

def record(output_path: str, duration: float = 3.0, sr: int = 16000):
    print(f"Recording {duration}s... (speak now)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    print(f"Saved: {path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_audio/clip.wav"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    record(out, duration=dur)
