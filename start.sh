#!/usr/bin/env bash
# start.sh - Say It Right launcher (activate venv and run)
#
# Usage:
#   ./start.sh                  # interactive CLI (default)
#   ./start.sh cli              # same as above
#   ./start.sh cli --text "كتاب" --audio clip.wav
#   ./start.sh api              # start API server on port 8000
#   ./start.sh api --port 9000
#   ./start.sh test             # run old test_pronunciation.py (batch tests)
#   ./start.sh record clip.wav 3  # record audio helper
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

MODE="${1:-cli}"
shift || true

case "$MODE" in
  cli)
    python cli.py "$@"
    ;;
  api)
    python api.py "$@"
    ;;
  test)
    python test_pronunciation.py "$@"
    ;;
  record)
    python record_audio.py "$@"
    ;;
  *)
    echo "Usage: ./start.sh [cli|api|test|record] [args...]"
    echo ""
    echo "Modes:"
    echo "  cli     Interactive CLI - prompts for text + audio (default)"
    echo "  api     Start API server (http://localhost:8000)"
    echo "  test    Run batch test_pronunciation.py"
    echo "  record  Record audio: ./start.sh record <output.wav> <seconds>"
    echo ""
    echo "Examples:"
    echo "  ./start.sh"
    echo "  ./start.sh cli --text \"كتاب\" --audio clip.wav"
    echo "  ./start.sh cli --text \"كتاب\" --record 3"
    echo "  ./start.sh api --port 9000"
    exit 1
    ;;
esac
