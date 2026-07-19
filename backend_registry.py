"""
Backend Registry — maps assessment modes to their ASR backends.

This is the routing layer. Each backend registers:
  - Which modes it supports
  - Which mode it's the default for
  - The loader method name (lazy init)
  - The recognizer method name (audio → phonemes)
  - Required Python dependencies

To add a new backend, add one entry here and implement two methods on
PronunciationEngine: _load_<name>() and _recognize_<name>().
"""

from typing import Optional, Dict, Any


BACKENDS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------ #
    #  wav2vec2 — phoneme-level ASR, used for everyday speech
    # ------------------------------------------------------------------ #
    "wav2vec2": {
        "modes": ["everyday"],
        "default_for": "everyday",
        "loader": "_load_wav2vec2",
        "recognizer": "_recognize_wav2vec2",
        "dependencies": ["transformers", "torch"],
        "description": "Phoneme-level ASR via wav2vec2 CTC decoding",
    },

    # ------------------------------------------------------------------ #
    #  tarteel — Whisper fine-tuned on Quranic Arabic, outputs diacritized
    #  text which is then run through CAMeL G2P for phoneme conversion.
    # ------------------------------------------------------------------ #
    "tarteel": {
        "modes": ["irab"],
        "default_for": "irab",
        "loader": "_load_tarteel",
        "recognizer": "_recognize_tarteel",
        "dependencies": ["transformers", "torch"],
        "description": "TarteelAI Whisper → diacritized text → CAMeL G2P",
    },

    # ------------------------------------------------------------------ #
    #  Future backends — register here when implemented:
    #
    #  "aras2p": {
    #      "modes": ["irab"],
    #      "default_for": None,
    #      "loader": "_load_aras2p",
    #      "recognizer": "_recognize_aras2p",
    #      "dependencies": ["aras2p"],
    #      "description": "Purpose-built phoneme mispronunciation detector",
    #  },
    #  "nemo": {
    #      "modes": ["irab"],
    #      "default_for": None,
    #      "loader": "_load_nemo",
    #      "recognizer": "_recognize_nemo",
    #      "dependencies": ["nemo_toolkit"],
    #      "description": "NVIDIA NeMo FastConformer diacritized ASR",
    #  },
    # ------------------------------------------------------------------ #
}


def list_backends() -> Dict[str, Dict[str, Any]]:
    """Return the full backend registry (for introspection / help text)."""
    return BACKENDS


def list_modes() -> list:
    """Return all distinct mode names across all backends."""
    modes = set()
    for info in BACKENDS.values():
        modes.update(info["modes"])
    return sorted(modes)


def resolve_backend(mode: str, backend: Optional[str] = None) -> str:
    """Resolve which backend to use for a given mode.

    Args:
        mode:    Assessment mode (e.g. "everyday", "irab").
        backend: Explicit backend override, or None for the mode's default.

    Returns:
        Backend name string (e.g. "wav2vec2", "tarteel").

    Raises:
        ValueError: if the mode has no backend, or the explicit backend
                    doesn't support the requested mode.
    """
    if backend:
        if backend not in BACKENDS:
            available = ", ".join(sorted(BACKENDS.keys()))
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {available}"
            )
        if mode not in BACKENDS[backend]["modes"]:
            raise ValueError(
                f"Backend '{backend}' does not support mode '{mode}'. "
                f"Supported modes: {BACKENDS[backend]['modes']}"
            )
        return backend

    # No explicit backend — find the default for this mode
    for name, info in BACKENDS.items():
        if info.get("default_for") == mode:
            return name

    # No default registered
    available_modes = ", ".join(list_modes())
    raise ValueError(
        f"No default backend for mode '{mode}'. "
        f"Available modes: {available_modes}"
    )


def get_backend_info(backend_name: str) -> Dict[str, Any]:
    """Return the full info dict for a backend."""
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKENDS[backend_name]
