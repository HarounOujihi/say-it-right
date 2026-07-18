#!/usr/bin/env python3
"""
Say It Right - Pronunciation Assessment Engine
Core logic - no I/O, no CLI, no API.
Used by cli.py, api.py, and test_pronunciation.py.
"""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC
from camel_tools.disambig.mle import MLEDisambiguator
import re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class PronunciationEngine:
    """Core pronunciation assessment engine.

    Loads the wav2vec2 model and CAMeL MLE disambiguator once,
    then can assess many (text, audio) pairs without reloading.
    """

    def __init__(self, model_id=None, verbose=True):
        """Initialize the engine.

        Args:
            model_id: Path to local model dir, or HF Hub model name.
                      If None, auto-detects local model_cache/.
            verbose:  Print progress messages during init.
        """
        self._log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        self._log("Initializing Say It Right engine...")

        # Use local model cache if available (avoids HF Hub download issues)
        if model_id is None:
            local_cache = Path(__file__).parent / "model_cache"
            if (local_cache / "model.safetensors").exists():
                model_id = str(local_cache)
                self._log(f"Using local model cache: {model_id}")
            else:
                model_id = "MostafaMaroof/wav2vec2-arabic-phoneme-asr"
                self._log(f"Using HF Hub model: {model_id}")

        # Initialize MLE Disambiguator for G2P via morphological analysis
        self._log("Loading CAMeL Tools MLE Disambiguator (calima-msa-r13)...")
        self.mle = MLEDisambiguator.pretrained("calima-msa-r13")

        # Load wav2vec2 model
        self._log(f"Loading wav2vec2 model: {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id)

        # Set device (CPU on this machine - AMD GPU has no CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self._log("Initialization complete!")
        self._log(f"Device: {self.device}")

    # ------------------------------------------------------------------ #
    #  Text normalization & G2P
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_arabic_text(text):
        """Basic Arabic text normalization."""
        # Remove Arabic diacritics (tashkeel)
        text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
        # Normalize Alef variants
        text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
        text = text.replace("\u0622", "\u0627")
        # Remove tatweel
        text = text.replace("\u0640", "")
        return text.strip()

    def text_to_phonemes(self, text):
        """Convert Arabic text to phonemes using CAMeL Tools MLE Disambiguator.

        Uses the caphi (camel phonetic) feature from morphological analysis.
        """
        try:
            text = self._normalize_arabic_text(text)
            words = text.split()

            if not words:
                return []

            disambiguated = self.mle.disambiguate(words)

            all_phonemes = []
            for result in disambiguated:
                if result.analyses:
                    analysis = result.analyses[0].analysis
                    caphi = analysis.get("caphi", "")
                    word_phonemes = [
                        p for p in caphi.split("_") if p and p.strip()
                    ]
                    all_phonemes.extend(word_phonemes)

            return all_phonemes
        except Exception as e:
            return []

    # ------------------------------------------------------------------ #
    #  Audio recognition
    # ------------------------------------------------------------------ #

    def audio_to_phonemes(self, audio_path):
        """Recognize phonemes from audio file using wav2vec2.

        Args:
            audio_path: Path to WAV file (any sample rate, mono or stereo).

        Returns:
            List of phoneme strings.
        """
        try:
            # Load audio with soundfile (avoids torchcodec dependency)
            audio_data, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo (shape: [samples, channels])
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), num_samples),
                    np.arange(len(audio_data)),
                    audio_data,
                )
                sample_rate = 16000

            input_values = self.processor(
                audio_data.astype(np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_values

            input_values = input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            phonemes = [
                p for p in transcription.strip().split() if p and p != "[PAD]"
            ]
            return phonemes
        except Exception as e:
            print(f"Audio recognition error: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare_phonemes(target, recognized):
        """Compare two phoneme lists using edit distance.

        Returns:
            (errors_dict, total_errors, accuracy_pct)
        """
        m, n = len(target), len(recognized)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if target[i - 1] == recognized[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    )

        substitutions, insertions, deletions = [], [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and target[i - 1] == recognized[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions.append((target[i - 1], recognized[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions.append(target[i - 1])
                i -= 1
            else:
                insertions.append(recognized[j - 1])
                j -= 1

        errors = {
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
        }
        total_errors = len(substitutions) + len(insertions) + len(deletions)
        accuracy = (1 - total_errors / max(m, 1)) * 100 if m > 0 else 0.0
        return errors, total_errors, accuracy

    # ------------------------------------------------------------------ #
    #  Full assessment
    # ------------------------------------------------------------------ #

    def assess(self, text, audio_path, verbose=True):
        """Run the full assessment pipeline.

        Args:
            text:       Arabic text (target pronunciation).
            audio_path: Path to WAV file of the user's pronunciation.
            verbose:    Print step-by-step results.

        Returns:
            dict with keys:
                text, audio_path, target_phonemes, recognized_phonemes,
                accuracy, errors (substitutions/insertions/deletions),
                total_errors
        """
        log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        log(f"\nAssessing pronunciation for: '{text}'")
        log(f"Audio file: {audio_path}")

        log("Step 1: Converting text to phonemes...")
        target_phonemes = self.text_to_phonemes(text)
        log(f"  Target phonemes: {target_phonemes}")

        log("Step 2: Recognizing phonemes from audio...")
        recognized_phonemes = self.audio_to_phonemes(audio_path)
        log(f"  Recognized phonemes: {recognized_phonemes}")

        log("Step 3: Comparing phonemes...")
        errors, total_errors, accuracy = self.compare_phonemes(
            target_phonemes, recognized_phonemes
        )

        if verbose:
            correct = len(target_phonemes) - len(errors["substitutions"]) - len(errors["deletions"])
            print("\n" + "=" * 50)
            print("ASSESSMENT RESULTS")
            print("=" * 50)
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Correct phonemes: {correct}/{len(target_phonemes)}")
            print(f"Errors: {total_errors}")
            print(f"  - Substitutions: {len(errors['substitutions'])}")
            if errors["substitutions"]:
                for tgt, rec in errors["substitutions"]:
                    print(f"      {tgt} -> {rec}")
            print(f"  - Insertions: {len(errors['insertions'])}")
            if errors["insertions"]:
                print(f"      {errors['insertions']}")
            print(f"  - Deletions: {len(errors['deletions'])}")
            if errors["deletions"]:
                print(f"      {errors['deletions']}")
            print("=" * 50)

        return {
            "text": text,
            "audio_path": audio_path,
            "target_phonemes": target_phonemes,
            "recognized_phonemes": recognized_phonemes,
            "accuracy": round(accuracy, 2),
            "errors": {
                "substitutions": errors["substitutions"],
                "insertions": errors["insertions"],
                "deletions": errors["deletions"],
            },
            "total_errors": total_errors,
        }
