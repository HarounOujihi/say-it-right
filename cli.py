#!/usr/bin/env python3
"""
Say It Right - CLI

Usage:
  # Interactive mode (prompts for text + audio):
  python cli.py

  # One-liner mode:
  python cli.py --text "كتاب" --audio test_audio/clip.wav

  # Record then assess:
  python cli.py --text "كتاب" --record 3

  # JSON output (for scripts/piping):
  python cli.py --text "كتاب" --audio clip.wav --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pronunciation_engine import PronunciationEngine


def record_clip(output_path, duration=3.0):
    """Record audio using record_audio helper."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Error: sounddevice not installed.")
        print("Install with:  pip install sounddevice")
        print("Fedora also needs:  sudo dnf install portaudio-devel")
        sys.exit(1)

    sr = 16000
    print(f"Recording {duration}s... (speak now)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Say It Right - Arabic pronunciation assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                                          # interactive
  python cli.py --text "كتاب" --audio clip.wav          # direct
  python cli.py --text "كتاب" --record 3                # record 3s then assess
  python cli.py --text "كتاب" --audio clip.wav --json   # JSON output
        """,
    )
    parser.add_argument("--text", "-t", type=str, help="Arabic target text")
    parser.add_argument("--audio", "-a", type=str, help="Path to WAV audio file")
    parser.add_argument(
        "--record", "-r", type=float, nargs="?", const=3.0, metavar="SECONDS",
        help="Record audio for N seconds (default 3s) then assess",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output results as JSON (for scripts/piping)",
    )
    args = parser.parse_args()

    # Determine text and audio source
    text = args.text
    audio_path = args.audio

    # If no args at all, go interactive
    if not text and not audio_path and args.record is None:
        print("=" * 50)
        print("  Say It Right")
        print("=" * 50)

        text = input("\nEnter Arabic text: ").strip()
        if not text:
            print("Error: text cannot be empty.")
            sys.exit(1)

        choice = input(
            "Audio source:\n"
            "  1) Use existing file\n"
            "  2) Record now\n"
            "Choose [1/2]: "
        ).strip()

        if choice == "2":
            duration = input("Duration in seconds [3]: ").strip()
            duration = float(duration) if duration else 3.0
            audio_path = "test_audio/cli_recording.wav"
            record_clip(audio_path, duration)
        else:
            audio_path = input("Enter audio file path: ").strip()

    # Handle --record flag
    elif args.record is not None:
        if not text:
            print("Error: --record requires --text")
            sys.exit(1)
        audio_path = "test_audio/cli_recording.wav"
        record_clip(audio_path, args.record)

    # Validate we have both text and audio
    if not text:
        print("Error: --text is required")
        sys.exit(1)
    if not audio_path:
        print("Error: --audio is required (or use --record)")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    # Run assessment
    engine = PronunciationEngine(verbose=not args.json)
    result = engine.assess(text, audio_path, verbose=not args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
