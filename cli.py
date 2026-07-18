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

  # Manual target (skip G2P, use your own phoneme list):
  python cli.py --audio clip.wav --target "k,i,t,aa,b"

  # Edit target interactively before assessing:
  python cli.py --text "كتاب" --audio clip.wav --edit-target

  # Save corrected target to phrasebook for future runs:
  python cli.py --text "كتاب" --audio clip.wav --edit-target --save-target
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


def parse_target_string(s):
    """Parse a target string like 'k,i,t,aa,b' or 'k i t aa b' into a list."""
    if not s:
        return []
    # Accept comma, space, or mixed separators
    parts = s.replace(",", " ").split()
    return [p.strip() for p in parts if p.strip()]


def edit_target_interactively(engine, text):
    """Show auto-generated phonemes, let user edit, return corrected list.

    Opens $EDITOR if set, otherwise prompts on stdin.
    """
    auto_target, source = engine.get_target_phonemes(text)
    print(f"\nAuto-generated target ({source}): {auto_target}")
    print(f"  (CAMeL may include case endings like final 'i' or 'a' — remove them)")
    editor = os.environ.get("EDITOR")
    if editor:
        # Write to temp file, open editor, read back
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("# Edit the phoneme list below (one per line or space/comma separated)\n")
            tmp.write("# Lines starting with # are ignored\n")
            tmp.write(" ".join(auto_target) + "\n")
            tmp_path = tmp.name
        try:
            os.system(f"{editor} '{tmp_path}'")
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if not ln.strip().startswith("#")]
            edited = parse_target_string(" ".join(lines))
            if not edited:
                print("Warning: empty edit, using original target")
                return auto_target
            return edited
        finally:
            os.unlink(tmp_path)
    else:
        # Fallback: inline prompt
        default_str = " ".join(auto_target)
        user_input = input(f"Edit target [default: {default_str}]: ").strip()
        if not user_input:
            return auto_target
        return parse_target_string(user_input)


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
  python cli.py --audio clip.wav --target "k,i,t,aa,b"  # manual target
  python cli.py --text "كتاب" --audio clip.wav --edit-target
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
    parser.add_argument(
        "--target", type=str, default=None,
        help="Manual target phonemes (skip G2P). "
             "Format: 'k,i,t,aa,b' or 'k i t aa b'",
    )
    parser.add_argument(
        "--edit-target", action="store_true",
        help="Show auto-generated target and edit before assessing",
    )
    parser.add_argument(
        "--save-target", action="store_true",
        help="Save the (possibly edited) target to phrasebook.json for future use",
    )
    args = parser.parse_args()

    # Determine text and audio source
    text = args.text
    audio_path = args.audio
    manual_target = parse_target_string(args.target) if args.target else None

    # If no args at all, go interactive
    if (
        not text
        and not audio_path
        and args.record is None
        and manual_target is None
    ):
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

    # Validate we have audio
    if not audio_path:
        print("Error: --audio is required (or use --record)")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    # Need either text or manual target
    if not text and not manual_target:
        print("Error: --text or --target is required")
        sys.exit(1)

    # Initialize engine
    engine = PronunciationEngine(verbose=not args.json)

    # Determine which assessment path to use
    if manual_target:
        # Manual target path — skip G2P
        if args.save_target and text:
            engine.save_target(text, manual_target)
            if not args.json:
                print(f"Saved target to phrasebook: {text} → {manual_target}")
        result = engine.assess_with_target(
            manual_target, audio_path, text=text, verbose=not args.json
        )
    elif args.edit_target:
        # Edit-target path — show G2P output, let user correct
        if not text:
            print("Error: --edit-target requires --text")
            sys.exit(1)
        corrected = edit_target_interactively(engine, text)
        if args.save_target:
            engine.save_target(text, corrected)
            if not args.json:
                print(f"Saved target to phrasebook: {text} → {corrected}")
        result = engine.assess_with_target(
            corrected, audio_path, text=text, verbose=not args.json
        )
    else:
        # Standard path — G2P or phrasebook lookup
        result = engine.assess(text, audio_path, verbose=not args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
