"""Simplified-Chinese normalization helpers.

All Chinese shown in the UI (ASR source text and translations) must be
Simplified Chinese. Whisper (esp. the small models) and some LLM outputs can
occasionally emit Traditional characters, so we run a cheap OpenCC t2s
conversion as a safety net. If OpenCC is unavailable the text is returned
unchanged (graceful degradation).
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_converter = None
_converter_ready = False


def _get_converter():
    global _converter, _converter_ready
    if _converter_ready:
        return _converter
    _converter_ready = True
    try:
        from opencc import OpenCC  # type: ignore

        _converter = OpenCC("t2s")
    except Exception:
        _converter = None
    return _converter


def to_simplified(text: str) -> str:
    """Convert any Traditional Chinese in *text* to Simplified Chinese.

    Non-Chinese text is returned untouched. Safe to call on every segment.
    """
    if not text:
        return text
    if not _CJK_RE.search(text):
        return text
    converter = _get_converter()
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception:
        return text
