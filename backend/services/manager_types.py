from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal["READ", "WRITE_DRAFT", "WRITE_COMMITTABLE"]
WriterMode = Literal["literal", "concise", "hold_high_risk"]


@dataclass
class SemanticSkeleton:
    actor: str | None
    action: str | None
    polarity: Literal["positive", "negative", "unknown"]
    obj: str | None
    condition_present: bool
    number_present: bool


@dataclass
class SourceState:
    raw_text: str
    committed_source_text: str
    live_source_tail: str
    source_lang: str
    asr_is_final: bool = False
    entities: list[str] = field(default_factory=list)
    temporal_markers: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    compare_spans: list[str] = field(default_factory=list)
    semantic_skeleton: SemanticSkeleton | None = None


@dataclass
class GateDecision:
    action: ActionType
    reason_codes: list[str]
    confidence: float


@dataclass
class WritePlan:
    mode: WriterMode
    allowed_source_span: tuple[int, int]
    blocked_spans: list[str]
    compression_budget: float


@dataclass
class ContextPack:
    glossary: dict[str, str] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    retrieved_snippets: list[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    translated_text: str
    route: str
    latency_ms: float
    gate_decision: GateDecision
    write_plan: WritePlan
    source_state: SourceState
    context_pack: ContextPack
    risk_notes: list[str] = field(default_factory=list)
    backend_meta: dict[str, Any] = field(default_factory=dict)
