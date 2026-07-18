#!/usr/bin/env bash
# stop.sh - Say It Right: kill any running process (CLI, API, or test)
KILLED=0

for pattern in "python cli.py" "python api.py" "python test_pronunciation.py" "python record_audio.py"; do
  if pkill -f "$pattern" 2>/dev/null; then
    echo "Stopped: $pattern"
    KILLED=1
  fi
done

if [ "$KILLED" -eq 0 ]; then
  echo "No running pronunciation process found."
fi
