from __future__ import annotations

import re

from services.manager_types import SemanticSkeleton, SourceState


NEGATIONS = {"not", "never", "no", "without"}
CONDITIONS = {"if", "unless", "except", "only", "when"}
COMPARE = {"more", "less", "than"}
PRONOUNS = {"i", "we", "you", "they", "he", "she", "it"}
AUXILIARY_VERBS = {"am", "is", "are", "was", "were", "be", "will"}
LEXICAL_VERB_CUES = {
    "agree",
    "agrees",
    "approve",
    "approves",
    "ask",
    "asks",
    "delay",
    "expand",
    "expands",
    "launch",
    "need",
    "needs",
    "present",
    "presents",
    "reject",
    "review",
    "reviews",
    "start",
    "starts",
}
DISCOURSE_OPENERS = {"well", "so", "okay", "ok", "actually", "basically", "right", "now"}
MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
RELATIVE_TIME_MARKERS = {"today", "tomorrow", "tonight", "yesterday"}
TIME_OF_DAY_MARKERS = {"morning", "afternoon", "evening", "night"}
PERIOD_MARKERS = {"week", "month", "quarter", "year"}
PRONOUN_CONTRACTION_PREFIXES = ("i'", "we'", "you'", "they'", "he'", "she'", "it'")
NON_ENTITY_CAPITALIZED_TOKENS = {"going"}


def build_source_state(
    *,
    raw_text: str,
    committed_source_text: str,
    source_lang: str,
    asr_is_final: bool,
) -> SourceState:
    text = raw_text.strip()
    committed = committed_source_text.strip()
    tokens = [token for token in text.split() if token]
    lowered = [token.lower().strip(",.?!") for token in tokens]
    live_tail = text
    if committed and text.startswith(committed):
        live_tail = text[len(committed) :].strip() or text

    english_like = _is_english_like(source_lang=source_lang, text=text)
    entities = _extract_entities(tokens, lowered) if english_like else []
    temporal_markers = _extract_temporal_markers(tokens, lowered) if english_like else []
    numbers = _extract_numbers(tokens, lowered) if english_like else _basic_number_tokens(tokens)
    negations = [token for token in tokens if _is_negation_token(token.lower().strip(",.?!"))] if english_like else []
    conditions = [token for token in tokens if token.lower().strip(",.?!") in CONDITIONS] if english_like else []
    compare_spans = [token for token in tokens if token.lower().strip(",.?!") in COMPARE] if english_like else []
    skeleton = _build_skeleton(lowered) if english_like else None

    return SourceState(
        raw_text=text,
        committed_source_text=committed,
        live_source_tail=live_tail,
        source_lang=source_lang,
        asr_is_final=asr_is_final,
        entities=entities,
        temporal_markers=temporal_markers,
        numbers=numbers,
        negations=negations,
        conditions=conditions,
        compare_spans=compare_spans,
        semantic_skeleton=skeleton,
    )


def merge_source_context(existing: str, new_text: str, *, max_chars: int) -> str:
    left = existing.strip()
    right = new_text.strip()
    if not left:
        return right[-max_chars:]
    if not right:
        return left[-max_chars:]

    merged = f"{left} {right}".strip()
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]


def _is_english_like(*, source_lang: str, text: str) -> bool:
    if source_lang == "en":
        return True
    if source_lang != "auto":
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _build_skeleton(lowered: list[str]) -> SemanticSkeleton:
    actor = next((token for token in lowered if token in PRONOUNS), None)
    action = next((token for token in reversed(lowered) if token in LEXICAL_VERB_CUES), None)
    if action is None:
        action = next((token for token in lowered if token in AUXILIARY_VERBS), None)

    polarity = "negative" if any(_is_negation_token(token) for token in lowered) else "positive"
    if actor is None and action is None:
        polarity = "unknown"

    obj = None
    if action and action in lowered:
        idx = lowered.index(action)
        if idx + 1 < len(lowered):
            obj = lowered[idx + 1]

    return SemanticSkeleton(
        actor=actor,
        action=action,
        polarity=polarity,
        obj=obj,
        condition_present=any(token in CONDITIONS for token in lowered),
        number_present=any(re.search(r"\d", token or "") for token in lowered),
    )


def _extract_entities(tokens: list[str], lowered: list[str]) -> list[str]:
    entities: list[str] = []
    for token, lowered_token in zip(tokens, lowered, strict=False):
        stripped = token.strip(",.?!")
        if len(stripped) <= 1 or not stripped[:1].isupper():
            continue
        if lowered_token in DISCOURSE_OPENERS or lowered_token in MONTH_NAMES:
            continue
        if lowered_token in PRONOUNS or lowered_token in AUXILIARY_VERBS or lowered_token in LEXICAL_VERB_CUES:
            continue
        if lowered_token in NON_ENTITY_CAPITALIZED_TOKENS:
            continue
        if any(lowered_token.startswith(prefix) for prefix in PRONOUN_CONTRACTION_PREFIXES):
            continue
        if re.fullmatch(r"Q[1-4]", stripped, re.IGNORECASE):
            continue
        if re.fullmatch(r"(19|20)\d{2}", stripped):
            continue
        entities.append(stripped)
    return sorted(set(entities))


def _extract_temporal_markers(tokens: list[str], lowered: list[str]) -> list[str]:
    markers: list[str] = []
    for idx, token in enumerate(tokens):
        lowered_token = lowered[idx]
        stripped = token.strip(",.?!")
        if lowered_token in MONTH_NAMES:
            markers.append(stripped)
            if idx + 1 < len(tokens):
                next_token = tokens[idx + 1].strip(",.?!")
                if re.fullmatch(r"\d{1,2}", next_token):
                    markers.append(f"{stripped} {next_token}")
            continue
        if re.fullmatch(r"Q[1-4]", stripped, re.IGNORECASE):
            markers.append(stripped)
            continue
        if re.fullmatch(r"(19|20)\d{2}", stripped):
            markers.append(stripped)
            continue
        if lowered_token in RELATIVE_TIME_MARKERS:
            markers.append(stripped)
            if idx + 1 < len(tokens) and lowered[idx + 1] in TIME_OF_DAY_MARKERS:
                markers.append(f"{stripped} {tokens[idx + 1].strip(',.?!')}")
            continue
        if lowered_token in {"next", "this"} and idx + 1 < len(tokens):
            next_lowered = lowered[idx + 1]
            if next_lowered in PERIOD_MARKERS or next_lowered in TIME_OF_DAY_MARKERS:
                markers.append(f"{stripped} {tokens[idx + 1].strip(',.?!')}")
    return sorted(set(markers))


def _extract_numbers(tokens: list[str], lowered: list[str]) -> list[str]:
    numbers = [token for token in tokens if re.search(r"\d", token)]
    if not numbers:
        return []

    temporal_numbers: set[str] = set()
    for idx, token in enumerate(tokens):
        stripped = token.strip(",.?!")
        if re.fullmatch(r"Q[1-4]", stripped, re.IGNORECASE):
            temporal_numbers.add(stripped)
            continue
        if not stripped.isdigit():
            continue
        prev_lowered = lowered[idx - 1] if idx > 0 else ""
        if prev_lowered in MONTH_NAMES:
            temporal_numbers.add(stripped)
            continue
        if len(stripped) == 4 and re.fullmatch(r"(19|20)\d{2}", stripped):
            temporal_numbers.add(stripped)
    return [token for token in numbers if token.strip(",.?!") not in temporal_numbers]


def _basic_number_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if re.search(r"\d", token)]


def _is_negation_token(lowered_token: str) -> bool:
    return (
        lowered_token in NEGATIONS
        or lowered_token.endswith("n't")
        or lowered_token in {"won't", "cant", "can't"}
    )
