from __future__ import annotations

import json
import re
from pathlib import Path

from services.manager_types import ContextPack


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class ContextPackManager:
    def __init__(self, glossary_path: Path) -> None:
        self._glossary_path = glossary_path
        self._loaded_mtime: float | None = None
        self._glossary_by_pair: dict[str, dict[str, str]] = {}

    def get_context(self, *, source_text: str, source_lang: str, target_lang: str) -> ContextPack:
        self._refresh_if_needed()
        lang_pair = f"{source_lang}->{target_lang}"
        mapping = self._glossary_by_pair.get(lang_pair, {})
        if not mapping:
            return ContextPack()

        normalized_source = _normalize_text(source_text)
        matched_glossary: dict[str, str] = {}
        matched_entities: list[str] = []
        for src, dst in mapping.items():
            if src == normalized_source or src in normalized_source:
                matched_glossary[src] = dst
                if src[:1].isalpha() and any(ch.isupper() for ch in dst):
                    matched_entities.append(dst)
                if len(matched_glossary) >= 12:
                    break

        return ContextPack(
            glossary=matched_glossary,
            entities=sorted(set(matched_entities)),
            retrieved_snippets=[],
        )

    def get_exact_glossary_hit(self, *, source_text: str, source_lang: str, target_lang: str) -> str | None:
        self._refresh_if_needed()
        mapping = self._glossary_by_pair.get(f"{source_lang}->{target_lang}", {})
        return mapping.get(_normalize_text(source_text))

    def _refresh_if_needed(self) -> None:
        if not self._glossary_path.exists():
            self._glossary_by_pair = {}
            self._loaded_mtime = None
            return

        stat = self._glossary_path.stat()
        mtime = stat.st_mtime
        if self._loaded_mtime is not None and mtime <= self._loaded_mtime:
            return

        raw = json.loads(self._glossary_path.read_text(encoding="utf-8"))
        parsed: dict[str, dict[str, str]] = {}
        if isinstance(raw, dict):
            for lang_pair, mapping in raw.items():
                if not isinstance(mapping, dict):
                    continue
                parsed[str(lang_pair)] = {
                    _normalize_text(str(src)): str(dst).strip()
                    for src, dst in mapping.items()
                    if str(src).strip() and str(dst).strip()
                }
        self._glossary_by_pair = parsed
        self._loaded_mtime = mtime
