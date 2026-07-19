#!/usr/bin/env python3
"""
Say It Right - CLI

Usage:
  # Interactive mode (prompts for text + audio):
  python cli.py

  # One-liner mode (existing audio file):
  python cli.py --text "كتاب" --audio test_audio/clip.wav

  # No --audio? Target is shown first, then you're prompted to record:
  python cli.py --text "كتاب"
  python cli.py --text "كتاب" --target "k i t aa b"

  # Explicit record duration:
  python cli.py --text "كتاب" --record 3

  # JSON output (for scripts/piping):
  python cli.py --text "كتاب" --audio clip.wav --json

  # Manual target (skip G2P, use your own phoneme list):
  python cli.py --target "k,i,t,aa,b" --audio clip.wav

  # Edit target interactively before assessing:
  python cli.py --text "كتاب" --audio clip.wav --edit-target

  # Save corrected target to phrasebook for future runs:
  python cli.py --text "كتاب" --audio clip.wav --edit-target --save-target

  # Get only the target phoneme array (no audio needed):
  python cli.py --text "كتاب" --phonemes

Modes:
  # Everyday speech (default — wav2vec2, harakat stripped):
  python cli.py --text "كتاب" --mode everyday

  # I'rāb / diacritized (harakat preserved, Tarteel Whisper backend):
  python cli.py --text "كِتَابٌ" --mode irab

  # Explicit backend override (if multiple backends exist for a mode):
  python cli.py --text "كِتَابٌ" --mode irab --backend tarteel

  # List available modes and backends:
  python cli.py --list-modes
  python cli.py --list-backends

Flow:
  1. Resolve target: --target (manual) > phrasebook > CAMeL G2P
  2. Resolve audio:  --audio (file) > --record N > prompt + countdown
  3. Assess and print results
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pronunciation_engine import PronunciationEngine
from backend_registry import list_modes, list_backends, resolve_backend


def record_clip(output_path, duration=3.0, countdown=True):
    """Record audio using record_audio helper.

    If countdown=True, waits for Enter and counts down before recording.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Error: sounddevice not installed.")
        print("Install with:  pip install sounddevice")
        print("Fedora also needs:  sudo dnf install portaudio-devel")
        sys.exit(1)

    if countdown:
        input("Press Enter when ready to record... ")
        import time
        for i in range(3, 0, -1):
            print(f"\rRecording in {i}... ", end="", flush=True)
            time.sleep(1)
        print("\rRecording now!         ")

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
  python cli.py --text "كتاب"                           # auto-gen target, then record
  python cli.py --text "كتاب" --target "k i t aa b"     # manual target, then record
  python cli.py --text "كتاب" --audio clip.wav          # use existing audio
  python cli.py --text "كتاب" --record 3                # record 3s then assess
  python cli.py --text "كتاب" --audio clip.wav --json   # JSON output
  python cli.py --audio clip.wav --target "k,i,t,aa,b"  # manual target, skip G2P
  python cli.py --text "كتاب" --audio clip.wav --edit-target
  python cli.py --text "كتاب" --phonemes                # print target array only

Modes:
  python cli.py --text "كتاب" --mode everyday           # default: wav2vec2, harakat stripped
  python cli.py --text "كِتَابٌ" --mode irab                       # harakat preserved, Tarteel Whisper
  python cli.py --text "كِتَابٌ" --mode irab --backend tarteel     # explicit backend
  python cli.py --list-modes                             # show available modes
  python cli.py --list-backends                          # show available backends
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
    parser.add_argument(
        "--phonemes", action="store_true",
        help="Print only the target phoneme array for --text and exit "
             "(no audio needed). Checks phrasebook first, falls back to G2P.",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="everyday",
        choices=list_modes(),
        help="Assessment mode (default: everyday). "
             "'everyday': wav2vec2, harakat stripped. "
             "'irab': harakat preserved, diacritized ASR backend.",
    )
    parser.add_argument(
        "--backend", "-b", type=str, default=None,
        help="Explicit ASR backend override (default: mode's default backend). "
             "Use --list-backends to see options.",
    )
    parser.add_argument(
        "--list-modes", action="store_true",
        help="List available assessment modes and exit.",
    )
    parser.add_argument(
        "--list-backends", action="store_true",
        help="List available ASR backends and exit.",
    )
    args = parser.parse_args()

    # --list-modes: print modes and exit
    if args.list_modes:
        print("Available modes:")
        for mode in list_modes():
            default_backend = resolve_backend(mode)
            print(f"  {mode}  (default backend: {default_backend})")
        sys.exit(0)

    # --list-backends: print backends and exit
    if args.list_backends:
        print("Available backends:")
        for name, info in list_backends().items():
            modes_str = ", ".join(info["modes"])
            default_marker = " (default)" if info.get("default_for") else ""
            print(f"  {name}{default_marker}")
            print(f"    modes:         {modes_str}")
            print(f"    description:   {info['description']}")
            print(f"    dependencies:  {', '.join(info['dependencies'])}")
            print()
        sys.exit(0)

    # Determine text and audio source
    text = args.text
    audio_path = args.audio
    manual_target = parse_target_string(args.target) if args.target else None

    # Validate mode/backend combo early (before recording) so users get a
    # clean error message instead of a traceback mid-session.
    try:
        resolved_backend = resolve_backend(args.mode, args.backend)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --phonemes: print target array and exit (no audio needed)
    if args.phonemes:
        if not text:
            print("Error: --phonemes requires --text")
            sys.exit(1)
        engine = PronunciationEngine(
            mode=args.mode, backend=args.backend, verbose=False
        )
        phonemes, source = engine.get_target_phonemes(text)
        if args.json:
            print(json.dumps(phonemes, ensure_ascii=False))
        else:
            print(" ".join(phonemes))
            print(f"[source: {source}]", file=sys.stderr)
        sys.exit(0)

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

    # Handle --record flag (explicit record duration requested)
    elif args.record is not None:
        if not text and not manual_target:
            print("Error: --record requires --text or --target")
            sys.exit(1)

    # Need either text or manual target
    if not text and not manual_target:
        print("Error: --text or --target is required")
        sys.exit(1)

    # Initialize engine
    engine = PronunciationEngine(
        mode=args.mode, backend=args.backend, verbose=not args.json
    )

    # Resolve target first (so we can show it before asking user to record)
    if manual_target:
        # Manual target path — skip G2P
        final_target = manual_target
        target_source = "manual"
        if args.save_target and text:
            engine.save_target(text, manual_target)
            if not args.json:
                print(f"Saved target to phrasebook: {text} → {manual_target}")
    elif args.edit_target:
        # Edit-target path — show G2P output, let user correct
        if not text:
            print("Error: --edit-target requires --text")
            sys.exit(1)
        final_target = edit_target_interactively(engine, text)
        target_source = "manual"
        if args.save_target:
            engine.save_target(text, final_target)
            if not args.json:
                print(f"Saved target to phrasebook: {text} → {final_target}")
    else:
        # Standard path — phrasebook lookup or G2P
        final_target, target_source = engine.get_target_phonemes(text)

    # Show the resolved target
    if not args.json:
        print(f"\nTarget phonemes ({target_source}): {' '.join(final_target)}")

    # Now resolve audio source
    if args.record is not None:
        # Explicit --record flag
        audio_path = "test_audio/cli_recording.wav"
        record_clip(audio_path, args.record)
    elif not audio_path:
        # No --audio and no --record → prompt to record with countdown
        if args.json:
            print("Error: --audio or --record is required when using --json")
            sys.exit(1)
        audio_path = "test_audio/cli_recording.wav"
        record_clip(audio_path, duration=3.0, countdown=True)

    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    # Run the assessment with the resolved target
    result = engine.assess_with_target(
        final_target, audio_path, text=text or "", verbose=not args.json
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
