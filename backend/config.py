import os

MAX_SPEAKERS = 3
MAX_WINDOWS = 1
SPEAKER_LABELS = ["A", "B", "C"]
SPEAKER_END_AFTER_SECONDS = 1.2
SPEAKER_KEEP_SECONDS = 3.0
SESSION_MAX_SECONDS = int(os.getenv("SESSION_MAX_SECONDS", "3600"))
SUPPORTED_LANGS = {"en", "fr", "zh", "es"}
TARGET_LANG_DEFAULT = os.getenv("TARGET_LANG_DEFAULT", "en")
LANG_DISPLAY_NAMES = {
    "en": "English",
    "fr": "French",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
}

MANAGER_BACKEND = os.getenv("MANAGER_BACKEND", "student_editor")
SINGLETRANS_ROOT = os.getenv("SINGLETRANS_ROOT", r"G:\singletrans")
STUDENT_EDITOR_ACTION_MODEL_DIR = os.getenv(
    "STUDENT_EDITOR_ACTION_MODEL_DIR",
    os.path.join(
        SINGLETRANS_ROOT,
        "logs",
        "student_editor_action_run_v9_meeting_round2_targeted_clean_v3",
        "model",
    ),
)
STUDENT_EDITOR_EDIT_MODEL_DIR = os.getenv(
    "STUDENT_EDITOR_EDIT_MODEL_DIR",
    os.path.join(
        SINGLETRANS_ROOT,
        "logs",
        "student_editor_edit_run_v20_meeting_round2_targeted_clean_v3",
        "model",
    ),
)
STUDENT_EDITOR_DRAFT_BACKEND = os.getenv("STUDENT_EDITOR_DRAFT_BACKEND", "hf_local_candidate")
STUDENT_EDITOR_DRAFT_MODEL = os.getenv("STUDENT_EDITOR_DRAFT_MODEL", "opus_mt_en_zh")
STUDENT_EDITOR_DEVICE = os.getenv("STUDENT_EDITOR_DEVICE", "cuda:0")
STUDENT_EDITOR_TORCH_DTYPE = os.getenv("STUDENT_EDITOR_TORCH_DTYPE", "float16")
STUDENT_EDITOR_RUN_ACTION = os.getenv("STUDENT_EDITOR_RUN_ACTION", "1") == "1"
STUDENT_EDITOR_RUN_EDIT = os.getenv("STUDENT_EDITOR_RUN_EDIT", "1") == "1"
STUDENT_EDITOR_ALWAYS_RUN_EDIT = os.getenv("STUDENT_EDITOR_ALWAYS_RUN_EDIT", "0") == "1"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Default is the fast 3B (low latency). Set OLLAMA_MODEL=qwen2.5:7b for higher
# quality at the cost of ~2x tail latency ("quality mode").
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

TRANSLATION_MAX_NEW_TOKENS = int(os.getenv("TRANSLATION_MAX_NEW_TOKENS", "96"))
TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.0"))
TRANSLATION_TOP_P = float(os.getenv("TRANSLATION_TOP_P", "0.9"))

ASR_BACKEND = os.getenv("ASR_BACKEND", "transformers_whisper_local")
ASR_BATCH_MS = int(os.getenv("ASR_BATCH_MS", "500"))
# End-of-sentence silence. Must be long enough to tell an intra-sentence pause
# from a real sentence end (else sentences get chopped into fragments). With the
# fine 100ms VAD cadence this is now the dominant, honest latency floor.
ASR_COMMIT_SILENCE_MS = int(os.getenv("ASR_COMMIT_SILENCE_MS", "300"))
# Minimum committed audio before a final is allowed: guards against tiny
# fragments and Whisper's short-segment hallucinations ("谢谢"/"Thank you").
ASR_MIN_COMMIT_MS = int(os.getenv("ASR_MIN_COMMIT_MS", "300"))
ASR_MAX_SEGMENT_MS = int(os.getenv("ASR_MAX_SEGMENT_MS", "4000"))
def _default_asr_model_name() -> str:
    if ASR_BACKEND == "transformers_whisper_local":
        return "openai/whisper-large-v3-turbo"
    if ASR_BACKEND == "mlx_whisper_local":
        return "mlx-community/whisper-small-mlx"
    return "small.en"


ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", _default_asr_model_name())
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
ASR_LANGUAGE_HINT = os.getenv("ASR_LANGUAGE_HINT", "en")

# Hybrid ASR (mlx backend): route by the session's *target* language, which for
# a bilingual interpreter reverse-maps to the source language. When the target
# is Chinese the speaker is talking English, so a smaller/faster model is used
# (English ASR stays accurate even on base); otherwise the accurate model is
# used so Chinese recognition is not degraded.
# When True, a stable interim result can be promoted to final without a fresh
# decode of the complete buffer. It lowers latency but can finalize a slightly
# truncated interim ("accuracy low / weird segmentation"), so it is OFF by
# default in favor of a proper final decode.
ASR_REUSE_PARTIAL_AS_FINAL = os.getenv("ASR_REUSE_PARTIAL_AS_FINAL", "0") not in ("0", "false", "False")

ASR_HYBRID_BY_TARGET = os.getenv("ASR_HYBRID_BY_TARGET", "1") not in ("0", "false", "False")
# English-only Whisper for the English source path: more accurate on English
# (numbers, terms) than the multilingual base, at similar speed.
ASR_FAST_MODEL_NAME = os.getenv("ASR_FAST_MODEL_NAME", "mlx-community/whisper-small.en-mlx")
# Targets for which the fast model is used (source is a non-Chinese language).
ASR_FAST_TARGET_LANGS = set(
    filter(None, os.getenv("ASR_FAST_TARGET_LANGS", "zh").split(","))
)
AUDIO_VOICED_RATIO_THRESHOLD = float(os.getenv("AUDIO_VOICED_RATIO_THRESHOLD", "0.12"))
AUDIO_ENERGY_THRESHOLD = float(os.getenv("AUDIO_ENERGY_THRESHOLD", "0.0012"))
# How often (ms of audio) the pipeline evaluates VAD / endpointing. Kept equal
# to the ASR decode interval. Empirically, a finer cadence detects end-of-speech
# a bit sooner but treats natural intra-sentence pauses as sentence ends, chopping
# sentences into fragments (worse segmentation AND, via extra late segments, worse
# tail latency). The coarse cadence intentionally smooths those pauses.
AUDIO_VAD_BATCH_MS = int(os.getenv("AUDIO_VAD_BATCH_MS", "500"))

LID_SMOOTH_WINDOW = int(os.getenv("LID_SMOOTH_WINDOW", "5"))
LID_MIN_CONFIDENCE = float(os.getenv("LID_MIN_CONFIDENCE", "0.65"))

MERGE_SHORT_MAX_CHARS = int(os.getenv("MERGE_SHORT_MAX_CHARS", "24"))
MERGE_MAX_WAIT_SECONDS = float(os.getenv("MERGE_MAX_WAIT_SECONDS", "0.15"))
MERGE_FORCE_CHARS = int(os.getenv("MERGE_FORCE_CHARS", "60"))

TRANSLATION_ROUTER_ENABLE_EXACT_CACHE = os.getenv("TRANSLATION_ROUTER_ENABLE_EXACT_CACHE", "1") == "1"
TRANSLATION_ROUTER_ENABLE_GLOSSARY = os.getenv("TRANSLATION_ROUTER_ENABLE_GLOSSARY", "1") == "1"
TRANSLATION_ROUTER_LOG_HITS = os.getenv("TRANSLATION_ROUTER_LOG_HITS", "1") == "1"
TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS = int(
    os.getenv("TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS", "2000")
)
TRANSLATION_ROUTER_ENABLE_FUZZY_TM = os.getenv("TRANSLATION_ROUTER_ENABLE_FUZZY_TM", "1") == "1"
TRANSLATION_ROUTER_TM_MAX_ITEMS = int(os.getenv("TRANSLATION_ROUTER_TM_MAX_ITEMS", "5000"))
TRANSLATION_ROUTER_FUZZY_THRESHOLD = float(os.getenv("TRANSLATION_ROUTER_FUZZY_THRESHOLD", "0.93"))
TRANSLATION_ROUTER_ENABLE_TM_POLISH = os.getenv("TRANSLATION_ROUTER_ENABLE_TM_POLISH", "1") == "1"
TRANSLATION_ROUTER_TM_POLISH_THRESHOLD = float(
    os.getenv("TRANSLATION_ROUTER_TM_POLISH_THRESHOLD", "0.85")
)
TRANSLATION_EVENT_INCLUDE_ROUTE_METRICS = (
    os.getenv("TRANSLATION_EVENT_INCLUDE_ROUTE_METRICS", "1") == "1"
)
