#!/usr/bin/env python3
"""
Say It Right - Batch Test Script
Runs multiple test cases and saves results to JSON.

Run:  python test_pronunciation.py
"""

from pronunciation_engine import PronunciationEngine
import json
import os
from pathlib import Path


def main():
    """Run the test cases and save results to JSON."""
    tester = PronunciationEngine()

    # Define test cases
    # Replace audio paths with your actual files
    test_cases = [
        {
            "text": "كتاب",
            "audio": "test_audio/correct.wav",
            "label": "Correct pronunciation",
        },
        {
            "text": "كتاب",
            "audio": "test_audio/incorrect.wav",
            "label": "Incorrect pronunciation",
        },
    ]

    results = []
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for idx, case in enumerate(test_cases, 1):
        audio_path = case["audio"]
        if not os.path.exists(audio_path):
            print(
                f"\n[SKIP] Test case {idx} ({case['label']}): "
                f"audio file not found: {audio_path}"
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"TEST CASE {idx}: {case['label']}")
        print(f"{'=' * 60}")

        result = tester.assess(case["text"], audio_path)
        result["label"] = case["label"]
        results.append(result)

    # Save all results to JSON
    output_file = results_dir / "test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
