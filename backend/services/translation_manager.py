from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path

from config import MANAGER_BACKEND, TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS
from services.benchmark_metrics import translation_benchmark_store
from services.context_pack_manager import ContextPackManager
from services.manager_engine import build_manager_engine
from services.manager_types import GateDecision, TranslationResult
from services.risk_writer import RiskAwareWriterV0
from services.source_state_manager import build_source_state
from services.translation_gate import TranslationGateV0
from services.translation_router import translate_with_router_detailed


logger = logging.getLogger(__name__)
GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "glossary.json"
SOURCE_CONTEXT_MAX_CHARS = 240


class _ExactCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._store: OrderedDict[tuple[str, str, str, str], TranslationResult] = OrderedDict()

    def get(self, key: tuple[str, str, str, str]) -> TranslationResult | None:
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def put(self, key: tuple[str, str, str, str], value: TranslationResult) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


_context_manager = ContextPackManager(GLOSSARY_PATH)
_gate = TranslationGateV0()
_writer = RiskAwareWriterV0()
_engine = build_manager_engine()
_cache = _ExactCache(max_items=TRANSLATION_ROUTER_EXACT_CACHE_MAX_ITEMS)


async def translate_with_manager_detailed(
    *,
    source_text: str,
    source_lang: str,
    target_lang: str,
    committed_source_text: str,
    previous_target_text: str = "",
) -> TranslationResult:
    source_state = build_source_state(
        raw_text=source_text,
        committed_source_text=committed_source_text,
        source_lang=source_lang,
        asr_is_final=True,
    )
    gate_decision = _gate.decide(source_state)
    write_plan = _writer.plan(source_state, gate_decision)
    context_pack = _context_manager.get_context(
        source_text=source_state.raw_text,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    cache_key = (
        source_lang,
        target_lang,
        write_plan.mode,
        source_state.raw_text.strip().lower(),
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    glossary_hit = _context_manager.get_exact_glossary_hit(
        source_text=source_state.raw_text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    if glossary_hit is not None:
        result = TranslationResult(
            translated_text=glossary_hit,
            route="manager_glossary",
            latency_ms=0.0,
            gate_decision=gate_decision,
            write_plan=write_plan,
            source_state=source_state,
            context_pack=context_pack,
            risk_notes=[],
            backend_meta={"engine": "manager_glossary", "mode": write_plan.mode},
        )
        _cache.put(cache_key, result)
        translation_benchmark_store.record_success(route="manager_glossary", latency_ms=0.0)
        return result

    use_legacy_router = MANAGER_BACKEND == "ollama" and _should_use_legacy_router(
        gate_decision=gate_decision,
        write_mode=write_plan.mode,
        source_state=source_state,
        committed_source_text=committed_source_text,
        context_glossary=context_pack.glossary,
    )
    if use_legacy_router:
        legacy = await translate_with_router_detailed(
            source_text=source_state.raw_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        result = TranslationResult(
            translated_text=legacy.translated_text,
            route=f"manager_router:{legacy.route}",
            latency_ms=legacy.latency_ms,
            gate_decision=gate_decision,
            write_plan=write_plan,
            source_state=source_state,
            context_pack=context_pack,
            risk_notes=[],
            backend_meta={"engine": "legacy_router", "mode": write_plan.mode},
        )
        _cache.put(cache_key, result)
        return result

    started = time.perf_counter()
    try:
        translated_text, backend_meta, latency_ms = await _engine.generate(
            source_state=source_state,
            write_plan=write_plan,
            context_pack=context_pack,
            source_lang=source_lang,
            target_lang=target_lang,
            previous_target_text=previous_target_text,
        )
        route = f"manager_{getattr(_engine, 'backend_name', 'engine')}"
        risk_notes = []
        if write_plan.blocked_spans:
            risk_notes.append("blocked=" + ",".join(write_plan.blocked_spans))
        fallback_reason = str(backend_meta.get("fallback_reason") or "").strip()
        if fallback_reason:
            risk_notes.append(fallback_reason)
        result = TranslationResult(
            translated_text=translated_text,
            route=route,
            latency_ms=latency_ms,
            gate_decision=gate_decision,
            write_plan=write_plan,
            source_state=source_state,
            context_pack=context_pack,
            risk_notes=risk_notes,
            backend_meta=backend_meta,
        )
        _cache.put(cache_key, result)
        translation_benchmark_store.record_success(route=route, latency_ms=latency_ms)
        return result
    except Exception:
        translation_benchmark_store.record_error()
        logger.exception("translation manager failed; falling back to source text")
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return TranslationResult(
            translated_text=source_state.raw_text,
            route="manager_fallback_error",
            latency_ms=latency_ms,
            gate_decision=gate_decision,
            write_plan=write_plan,
            source_state=source_state,
            context_pack=context_pack,
            risk_notes=["manager_fallback"],
            backend_meta={"engine": "manager_fallback", "mode": write_plan.mode},
        )


def _should_use_legacy_router(
    *,
    gate_decision: GateDecision,
    write_mode: str,
    source_state,
    committed_source_text: str,
    context_glossary: dict[str, str],
) -> bool:
    if gate_decision.action == "READ":
        return True
    if write_mode != "literal":
        return False
    if committed_source_text.strip():
        return False
    if context_glossary:
        return False
    if source_state.entities or source_state.temporal_markers:
        return False
    return True
