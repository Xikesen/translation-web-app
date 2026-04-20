import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from config import (
    TRANSLATION_ROUTER_ENABLE_EXACT_CACHE,
    TRANSLATION_ROUTER_ENABLE_FUZZY_TM,
    TRANSLATION_ROUTER_ENABLE_GLOSSARY,
    TRANSLATION_ROUTER_ENABLE_TM_POLISH,
    TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS,
    TRANSLATION_ROUTER_FUZZY_THRESHOLD,
    TRANSLATION_ROUTER_LOG_HITS,
    TRANSLATION_ROUTER_TM_POLISH_THRESHOLD,
    TRANSLATION_ROUTER_TM_MAX_ITEMS,
)
from services.benchmark_metrics import translation_benchmark_store
from services.translation_memory import TMEntry, TranslationMemory
from services.translator import polish_translation_candidate, translate_via_llm

logger = logging.getLogger(__name__)
GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.json"


@dataclass
class TranslationRouteResult:
    translated_text: str
    route: str
    latency_ms: float


class _ExactCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._store: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    def get(self, key: tuple[str, str, str]) -> str | None:
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def put(self, key: tuple[str, str, str], value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_glossary(raw: dict) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("glossary root must be a JSON object")
    parsed: dict[str, dict[str, str]] = {}
    for lang_pair, mapping in raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"glossary mapping for '{lang_pair}' must be an object")
        parsed[lang_pair] = {
            _normalize_text(str(k)): str(v).strip()
            for k, v in mapping.items()
            if str(k).strip() and str(v).strip()
        }
    return parsed


def _load_glossary() -> dict[str, dict[str, str]]:
    if not GLOSSARY_PATH.exists():
        return {}
    try:
        raw = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
        return _normalize_glossary(raw)
    except Exception as exc:
        logger.warning("translation_router: failed to parse glossary: %s", exc)
        return {}


_exact_cache = _ExactCache(max_items=TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS)
_glossary = _load_glossary()
_tm = TranslationMemory(max_items=TRANSLATION_ROUTER_TM_MAX_ITEMS)


def replace_glossary(raw: dict) -> dict[str, int]:
    global _glossary
    normalized = _normalize_glossary(raw)
    _glossary = normalized
    # Clear derived caches so newly uploaded terms take effect immediately.
    _exact_cache.clear()
    _tm.clear()
    item_count = sum(len(mapping) for mapping in normalized.values())
    return {"lang_pairs": len(normalized), "entries": item_count}


def _route_log(route: str, source_lang: str, target_lang: str, source_text: str) -> None:
    if not TRANSLATION_ROUTER_LOG_HITS:
        return
    preview = source_text.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    logger.info(
        "translation_router route=%s lang=%s->%s text='%s'",
        route,
        source_lang,
        target_lang,
        preview,
    )


def _jaccard_3gram(a: str, b: str) -> float:
    def grams(text: str) -> set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[i : i + 3] for i in range(len(text) - 2)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _levenshtein_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, replace))
        prev = curr
    dist = prev[-1]
    max_len = max(len(a), len(b))
    return 1.0 - (dist / max_len)


def _best_fuzzy_tm_match(source_lang: str, target_lang: str, normalized_text: str) -> tuple[TMEntry | None, float]:
    best_entry: TMEntry | None = None
    best_score = -1.0
    for entry in _tm.iter_pair_entries(source_lang=source_lang, target_lang=target_lang):
        jaccard = _jaccard_3gram(normalized_text, entry.source_text)
        lev = _levenshtein_similarity(normalized_text, entry.source_text)
        score = max(jaccard, lev)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry, best_score


def _glossary_translate(source_text: str, source_lang: str, target_lang: str) -> str | None:
    lang_pair = f"{source_lang}->{target_lang}"
    mapping = _glossary.get(lang_pair)
    if not mapping:
        return None
    normalized = _normalize_text(source_text)
    exact = mapping.get(normalized)
    if exact:
        return exact
    return None


async def _translate_with_router_internal(
    source_text: str, source_lang: str, target_lang: str
) -> TranslationRouteResult:
    started_at = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - started_at) * 1000

    def _success(route_name: str, translated_text: str) -> TranslationRouteResult:
        latency_ms = _elapsed_ms()
        translation_benchmark_store.record_success(route=route_name, latency_ms=latency_ms)
        return TranslationRouteResult(
            translated_text=translated_text,
            route=route_name,
            latency_ms=round(latency_ms, 2),
        )

    def _route_name_with_score(base_route: str, score: float) -> str:
        return f"{base_route}(score={score:.2f})"

    normalized_text = _normalize_text(source_text)
    cache_key = (source_lang, target_lang, normalized_text)
    fuzzy_direct_threshold = max(0.0, min(1.0, TRANSLATION_ROUTER_FUZZY_THRESHOLD))
    fuzzy_polish_threshold = max(0.0, min(fuzzy_direct_threshold, TRANSLATION_ROUTER_TM_POLISH_THRESHOLD))

    try:
        if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
            cached = _exact_cache.get(cache_key)
            if cached is not None:
                _route_log("exact_cache", source_lang, target_lang, source_text)
                return _success("exact_cache", cached)

        if TRANSLATION_ROUTER_ENABLE_GLOSSARY:
            glossary_hit = _glossary_translate(source_text, source_lang, target_lang)
            if glossary_hit is not None:
                _route_log("glossary", source_lang, target_lang, source_text)
                if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
                    _exact_cache.put(cache_key, glossary_hit)
                _tm.upsert(source_lang, target_lang, normalized_text, glossary_hit)
                return _success("glossary", glossary_hit)

        if TRANSLATION_ROUTER_ENABLE_FUZZY_TM:
            tm_exact = _tm.get_exact(source_lang, target_lang, normalized_text)
            if tm_exact is not None:
                _route_log("tm_exact", source_lang, target_lang, source_text)
                if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
                    _exact_cache.put(cache_key, tm_exact)
                return _success("tm_exact", tm_exact)

            candidate, score = _best_fuzzy_tm_match(
                source_lang=source_lang,
                target_lang=target_lang,
                normalized_text=normalized_text,
            )
            if candidate is not None and score >= fuzzy_direct_threshold:
                candidate.hit_count += 1
                route_name = _route_name_with_score("tm_fuzzy", score)
                _route_log(route_name, source_lang, target_lang, source_text)
                if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
                    _exact_cache.put(cache_key, candidate.target_text)
                _tm.upsert(source_lang, target_lang, normalized_text, candidate.target_text)
                return _success(route_name, candidate.target_text)

            if (
                candidate is not None
                and TRANSLATION_ROUTER_ENABLE_TM_POLISH
                and score >= fuzzy_polish_threshold
            ):
                try:
                    polished = await polish_translation_candidate(
                        source_text=source_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        candidate_translation=candidate.target_text,
                    )
                    route_name = _route_name_with_score("tm_polish", score)
                    _route_log(route_name, source_lang, target_lang, source_text)
                    if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
                        _exact_cache.put(cache_key, polished)
                    _tm.upsert(source_lang, target_lang, normalized_text, polished)
                    return _success(route_name, polished)
                except Exception as exc:
                    logger.warning("translation_router: tm polish failed, fallback to llm: %s", exc)

        llm_output = await translate_via_llm(
            text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        _route_log("llm", source_lang, target_lang, source_text)
        if TRANSLATION_ROUTER_ENABLE_EXACT_CACHE:
            _exact_cache.put(cache_key, llm_output)
        _tm.upsert(source_lang, target_lang, normalized_text, llm_output)
        return _success("llm", llm_output)
    except Exception:
        translation_benchmark_store.record_error()
        raise


async def translate_with_router_detailed(
    source_text: str, source_lang: str, target_lang: str
) -> TranslationRouteResult:
    return await _translate_with_router_internal(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
    )


async def translate_with_router(source_text: str, source_lang: str, target_lang: str) -> str:
    result = await _translate_with_router_internal(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    return result.translated_text
