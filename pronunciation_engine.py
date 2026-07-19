#!/usr/bin/env python3
"""
Say It Right - Pronunciation Assessment Engine
Core logic - no I/O, no CLI, no API.
Used by cli.py, api.py, and test_pronunciation.py.

Supports multiple assessment modes via a pluggable backend system:
  - everyday (default): wav2vec2 phoneme ASR, harakat stripped
  - irab:               Tarteel Whisper → diacritized text → CAMeL G2P
                        (same notation on both sides, harakat preserved)

See backend_registry.py for the routing layer and how to add new backends.
"""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC
from camel_tools.disambig.mle import MLEDisambiguator
import re
from pathlib import Path
import warnings

from phrasebook import lookup as phrasebook_lookup, save_entry as phrasebook_save
from backend_registry import resolve_backend, get_backend_info

warnings.filterwarnings("ignore")


class PronunciationEngine:
    """Core pronunciation assessment engine.

    Loads the ASR model and CAMeL MLE disambiguator once,
    then can assess many (text, audio) pairs without reloading.

    The assessment mode controls how harakat are handled and which
    ASR backend is used. The backend is resolved via backend_registry.
    """

    def __init__(self, mode="everyday", backend=None, model_id=None, verbose=True):
        """Initialize the engine.

        Args:
            mode:     Assessment mode ("everyday" or "irab").
            backend:  ASR backend override (e.g. "wav2vec2", "tarteel").
                      If None, uses the default backend for the mode.
            model_id: Path to local model dir, or HF Hub model name.
                      If None, auto-detects local model_cache/.
                      (Only used by wav2vec2 backend currently.)
            verbose:  Print progress messages during init.
        """
        self._log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        self._log("Initializing Say It Right engine...")

        # Resolve which backend to use
        self.mode = mode
        self.backend_name = resolve_backend(mode, backend)
        self.backend_info = get_backend_info(self.backend_name)

        # Mode config: should we preserve or strip harakat from text?
        self.preserve_harakat = (mode == "irab")

        self._log(f"Mode: {mode}  |  Backend: {self.backend_name}")
        self._log(f"Preserve harakat: {self.preserve_harakat}")

        # Always load CAMeL (needed by all modes for G2P)
        self._log("Loading CAMeL Tools MLE Disambiguator (calima-msa-r13)...")
        self.mle = MLEDisambiguator.pretrained("calima-msa-r13")

        # Device (shared by all backends that use torch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-load the backend-specific model
        loader_name = self.backend_info["loader"]
        loader = getattr(self, loader_name)
        loader(model_id=model_id) if model_id else loader()

        self._log("Initialization complete!")
        self._log(f"Device: {self.device}")

    # ------------------------------------------------------------------ #
    #  Backend loaders — each backend implements one _load_<name>() method
    # ------------------------------------------------------------------ #

    def _load_wav2vec2(self, model_id=None, **kwargs):
        """Load wav2vec2 phoneme ASR model (everyday mode)."""
        if model_id is None:
            local_cache = Path(__file__).parent / "model_cache"
            if (local_cache / "model.safetensors").exists():
                model_id = str(local_cache)
                self._log(f"Using local model cache: {model_id}")
            else:
                model_id = "MostafaMaroof/wav2vec2-arabic-phoneme-asr"
                self._log(f"Using HF Hub model: {model_id}")

        self._log(f"Loading wav2vec2 model: {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_tarteel(self, **kwargs):
        """Load TarteelAI Whisper model for diacritized Arabic ASR (irab mode).

        This model outputs Arabic text (potentially diacritized), which is
        then run through CAMeL G2P to produce phonemes in the same notation
        as the target — eliminating the phoneme vocabulary mismatch problem.
        """
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        model_name = "tarteel-ai/whisper-base-ar-quran"
        self._log(f"Loading Tarteel Whisper model: {model_name}...")
        self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper_model = self.whisper_model.to(self.device)
        self.whisper_model.eval()

    # ------------------------------------------------------------------ #
    #  Audio loading helper (shared by all backends)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_audio(audio_path):
        """Load audio file, convert to mono, resample to 16kHz.

        Returns:
            (audio_data as float32 numpy array, sample_rate as int)
        """
        audio_data, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
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

        return audio_data.astype(np.float32), sample_rate

    # ------------------------------------------------------------------ #
    #  Backend recognizers — each backend implements one _recognize_<name>()
    # ------------------------------------------------------------------ #

    def _recognize_wav2vec2(self, audio_path):
        """Recognize phonemes from audio via wav2vec2 CTC decoding.

        Returns: list of phoneme strings.
        """
        try:
            audio_data, sample_rate = self._load_audio(audio_path)

            input_values = self.processor(
                audio_data,
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
            print(f"Audio recognition error (wav2vec2): {e}")
            return []

    def _recognize_tarteel(self, audio_path):
        """Recognize phonemes from audio via Tarteel Whisper + CAMeL G2P.

        Pipeline:
          Audio → Whisper → diacritized Arabic text → CAMeL G2P → phonemes

        This produces phonemes in the SAME notation as the target (both go
        through CAMeL), eliminating the vocabulary mismatch problem.

        Returns: list of phoneme strings.
        """
        try:
            audio_data, sample_rate = self._load_audio(audio_path)

            # Whisper expects the raw audio array
            input_features = self.whisper_processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).input_features

            input_features = input_features.to(self.device)

            # Generate transcription. We intentionally do NOT pass
            # language="ar" / task="transcribe" kwargs because this model
            # (tarteel-ai/whisper-base-ar-quran) has an outdated generation
            # config that is incompatible with those kwargs in transformers
            # 5.x (raises ValueError). The model is fine-tuned for Arabic
            # transcription already, so the defaults produce correct output.
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)

            recognized_text = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            # The model emits <|ar|><|transcribe|><|notimestamps|> prefix
            # tokens but its tokenizer doesn't mark them as special, so
            # skip_special_tokens=True fails to strip them. Remove manually.
            recognized_text = re.sub(r"<\|[^|]+\|>", "", recognized_text).strip()

            if not recognized_text:
                return []

            # Run the recognized text through the same CAMeL G2P used for
            # the target. This produces phonemes in caphi notation — the
            # same notation the target uses.
            phonemes = self.text_to_phonemes(
                recognized_text,
                preserve_harakat=True,
            )
            return phonemes
        except Exception as e:
            print(f"Audio recognition error (tarteel): {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Dispatcher — routes to the active backend's recognizer
    # ------------------------------------------------------------------ #

    def audio_to_phonemes(self, audio_path):
        """Recognize phonemes from audio using the active backend.

        Args:
            audio_path: Path to WAV file (any sample rate, mono or stereo).

        Returns:
            List of phoneme strings.
        """
        recognizer_name = self.backend_info["recognizer"]
        recognizer = getattr(self, recognizer_name)
        return recognizer(audio_path)

    # ------------------------------------------------------------------ #
    #  Text normalization & G2P
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_arabic_text(text, preserve_harakat=False):
        """Normalize Arabic text.

        Args:
            text:              Input text.
            preserve_harakat:  If True, keep diacritics (for irab mode).
                               If False, strip them (for everyday mode).
        """
        if preserve_harakat:
            # Keep harakat, only normalize Alef variants
            text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
            text = text.replace("\u0622", "\u0627")
            text = text.replace("\u0640", "")  # remove tatweel
            return text.strip()

        # Original behavior: strip all diacritics
        text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
        text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
        text = text.replace("\u0622", "\u0627")
        text = text.replace("\u0640", "")
        return text.strip()

    def text_to_phonemes(self, text, preserve_harakat=None):
        """Convert Arabic text to phonemes using CAMeL Tools MLE Disambiguator.

        Uses the caphi (camel phonetic) feature from morphological analysis.

        Args:
            text:             Arabic text.
            preserve_harakat: Override for harakat preservation.
                              If None, uses self.preserve_harakat.

        Note: output includes case endings (i'rāb) which are often dropped
        in normal speech. Use get_target_phonemes() for the phrasebook-aware
        path, or pass a corrected list to assess_with_target().
        """
        try:
            if preserve_harakat is None:
                preserve_harakat = self.preserve_harakat

            text = self._normalize_arabic_text(text, preserve_harakat=preserve_harakat)
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
    #  Phrasebook-aware target lookup
    # ------------------------------------------------------------------ #

    def get_target_phonemes(self, text, use_phrasebook=True):
        """Resolve target phonemes for a text.

        Order of resolution:
          1. If use_phrasebook=True and text is in phrasebook.json,
             return the user-corrected list (skips G2P entirely).
          2. Otherwise, fall back to CAMeL MLE G2P.

        Returns:
            (phonemes, source) where source is "phrasebook" or "g2p".
        """
        if use_phrasebook:
            saved = phrasebook_lookup(text)
            if saved is not None:
                return list(saved), "phrasebook"
        return self.text_to_phonemes(text), "g2p"

    def save_target(self, text, phonemes):
        """Persist a corrected target phoneme list to the phrasebook."""
        phrasebook_save(text, phonemes)

    # ------------------------------------------------------------------ #
    #  Full assessment
    # ------------------------------------------------------------------ #

    def assess(self, text, audio_path, verbose=True, use_phrasebook=True):
        """Run the full assessment pipeline (text → G2P → compare).

        Args:
            text:           Arabic text (target pronunciation).
            audio_path:     Path to WAV file of the user's pronunciation.
            verbose:        Print step-by-step results.
            use_phrasebook: If True and text is in phrasebook.json, use the
                            saved phoneme list instead of running G2P.

        Returns:
            dict with keys:
                text, audio_path, target_phonemes, recognized_phonemes,
                accuracy, errors (substitutions/insertions/deletions),
                total_errors, target_source, mode, backend
        """
        log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        log(f"\nAssessing pronunciation for: '{text}'")
        log(f"Audio file: {audio_path}")
        log(f"Mode: {self.mode}  |  Backend: {self.backend_name}")

        log("Step 1: Resolving target phonemes...")
        target_phonemes, target_source = self.get_target_phonemes(
            text, use_phrasebook=use_phrasebook
        )
        log(f"  Target phonemes ({target_source}): {target_phonemes}")

        log("Step 2: Recognizing phonemes from audio...")
        recognized_phonemes = self.audio_to_phonemes(audio_path)
        log(f"  Recognized phonemes: {recognized_phonemes}")

        log("Step 3: Comparing phonemes...")
        errors, total_errors, accuracy = self.compare_phonemes(
            target_phonemes, recognized_phonemes
        )

        if verbose:
            self._print_results(target_phonemes, errors, total_errors, accuracy)

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
            "target_source": target_source,
            "mode": self.mode,
            "backend": self.backend_name,
        }

    def assess_with_target(self, target_phonemes, audio_path, text="", verbose=True):
        """Assess using an explicit target phoneme list (skips G2P entirely).

        Use this when the user has manually corrected the target phonemes
        and wants to compare audio against the corrected list.

        Args:
            target_phonemes: List of phoneme strings (the corrected target).
            audio_path:      Path to WAV file of the user's pronunciation.
            text:            Optional original text (for record-keeping only).
            verbose:         Print step-by-step results.

        Returns:
            Same dict shape as assess(), with target_source="manual".
        """
        log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        log(f"\nAssessing pronunciation (manual target)")
        log(f"Audio file: {audio_path}")
        log(f"Mode: {self.mode}  |  Backend: {self.backend_name}")

        log("Step 1: Using provided target phonemes...")
        log(f"  Target phonemes (manual): {target_phonemes}")

        log("Step 2: Recognizing phonemes from audio...")
        recognized_phonemes = self.audio_to_phonemes(audio_path)
        log(f"  Recognized phonemes: {recognized_phonemes}")

        log("Step 3: Comparing phonemes...")
        errors, total_errors, accuracy = self.compare_phonemes(
            target_phonemes, recognized_phonemes
        )

        if verbose:
            self._print_results(target_phonemes, errors, total_errors, accuracy)

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
            "target_source": "manual",
            "mode": self.mode,
            "backend": self.backend_name,
        }

    # ------------------------------------------------------------------ #
    #  Result printing (shared by assess and assess_with_target)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_results(target_phonemes, errors, total_errors, accuracy):
        """Print the assessment results table to stdout."""
        correct = (
            len(target_phonemes)
            - len(errors["substitutions"])
            - len(errors["deletions"])
        )
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
