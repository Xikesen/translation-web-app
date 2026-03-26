import os

MAX_SPEAKERS = 4
MAX_WINDOWS = 3
SPEAKER_LABELS = ["A", "B", "C", "D"]
SPEAKER_END_AFTER_SECONDS = 1.2
SPEAKER_KEEP_SECONDS = 3.0
SESSION_MAX_SECONDS = 120
SUPPORTED_LANGS = {"en", "fr", "zh", "ja"}
LANG_DISPLAY_NAMES = {
    "en": "English",
    "fr": "French",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
