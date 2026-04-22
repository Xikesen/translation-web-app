from __future__ import annotations

from services.manager_types import GateDecision, SourceState, WritePlan


class RiskAwareWriterV0:
    FILLERS = {"well", "you", "know", "i", "mean"}

    def plan(self, source_state: SourceState, gate: GateDecision) -> WritePlan:
        raw = source_state.raw_text
        token_count = len(raw.split())

        hard_blocked: list[str] = []
        if source_state.numbers:
            hard_blocked.extend(source_state.numbers)
        if source_state.negations:
            hard_blocked.extend(source_state.negations)
        if source_state.conditions:
            hard_blocked.extend(source_state.conditions)
        if source_state.compare_spans:
            hard_blocked.extend(source_state.compare_spans)

        soft_risk = bool(source_state.entities or source_state.temporal_markers)

        mode = "literal"
        compression_budget = 0.0
        if hard_blocked:
            mode = "hold_high_risk"
        elif gate.action == "WRITE_DRAFT" and not soft_risk:
            mode = "concise"
            compression_budget = 0.25

        return WritePlan(
            mode=mode,
            allowed_source_span=(0, token_count),
            blocked_spans=sorted(set(hard_blocked)),
            compression_budget=compression_budget,
        )

    def compress_text(self, text: str) -> str:
        tokens = text.split()
        kept = [token for token in tokens if token.lower().strip(",.?!") not in self.FILLERS]
        return " ".join(kept)
