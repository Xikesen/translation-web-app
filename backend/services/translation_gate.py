from __future__ import annotations

from services.manager_types import GateDecision, SourceState


HANGING_TAIL_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class TranslationGateV0:
    def decide(self, source_state: SourceState) -> GateDecision:
        text = source_state.raw_text.strip()
        if not text:
            return GateDecision(action="READ", reason_codes=["EMPTY_TEXT"], confidence=0.0)

        token_count = len(text.split())
        critical_risk = bool(
            source_state.numbers
            or source_state.negations
            or source_state.conditions
            or source_state.compare_spans
        )
        soft_risk = bool(source_state.entities or source_state.temporal_markers)
        clause_ready = self._clause_ready(source_state)
        terminal = text.endswith((".", "!", "?", ";", ":", "\u3002", "\uff01", "\uff1f", "\uff1b"))
        hanging_tail = self._has_hanging_tail(source_state)

        if not clause_ready:
            return GateDecision(action="WRITE_DRAFT", reason_codes=["CLAUSE_INCOMPLETE"], confidence=0.45)

        if critical_risk and hanging_tail and not source_state.asr_is_final:
            return GateDecision(action="WRITE_DRAFT", reason_codes=["HARD_RISK_WAITING_TAIL"], confidence=0.55)

        if critical_risk:
            if source_state.asr_is_final or terminal or token_count >= 6:
                return GateDecision(action="WRITE_COMMITTABLE", reason_codes=["HARD_RISK_COMMITTABLE"], confidence=0.8)
            return GateDecision(action="WRITE_DRAFT", reason_codes=["HARD_RISK_BUFFERED_DRAFT"], confidence=0.6)

        if soft_risk and not (source_state.asr_is_final or terminal):
            return GateDecision(action="WRITE_DRAFT", reason_codes=["SOFT_RISK_EARLY_DRAFT"], confidence=0.65)

        if source_state.asr_is_final or terminal or token_count >= 8:
            return GateDecision(action="WRITE_COMMITTABLE", reason_codes=["LOW_RISK_COMMITTABLE"], confidence=0.85)

        return GateDecision(action="WRITE_DRAFT", reason_codes=["LOW_RISK_DRAFT"], confidence=0.7)

    def _clause_ready(self, source_state: SourceState) -> bool:
        skeleton = source_state.semantic_skeleton
        if skeleton is None:
            return len(source_state.raw_text.split()) >= 2
        if skeleton.action is not None and skeleton.actor is not None:
            return True
        if skeleton.action is not None and len(source_state.raw_text.split()) >= 4:
            return True
        if source_state.entities and len(source_state.raw_text.split()) >= 4:
            return True
        if source_state.temporal_markers and len(source_state.raw_text.split()) >= 4:
            return True
        return False

    def _has_hanging_tail(self, source_state: SourceState) -> bool:
        tokens = [token.strip(",.?!").lower() for token in source_state.raw_text.split() if token.strip(",.?!")]
        if not tokens:
            return False
        return tokens[-1] in HANGING_TAIL_TOKENS
