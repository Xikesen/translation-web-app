from __future__ import annotations

import re
from dataclasses import dataclass


MONTH_NAMES = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}
ENTITY_HEADS = {"project", "program", "initiative", "platform", "product"}
GENERIC_ENTITY_TAILS = {
    "budget",
    "decision",
    "delay",
    "issue",
    "launch",
    "manager",
    "meeting",
    "plan",
    "review",
    "risk",
    "schedule",
    "scope",
    "status",
    "team",
    "timeline",
    "update",
}
ENTITY_BREAK_TAILS = {
    "after",
    "and",
    "at",
    "before",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "to",
    "until",
    "with",
}
UPPERCASE_TOKENS = {"ai", "api", "asr", "kpi", "llm", "qa", "roi", "ui", "ux"}
TERMINAL_PUNCTUATION = (".", "!", "?", ";", ":", ",")
NON_FINAL_END_TOKENS = {
    "after",
    "and",
    "at",
    "before",
    "because",
    "but",
    "by",
    "for",
    "if",
    "in",
    "of",
    "on",
    "or",
    "so",
    "than",
    "to",
    "until",
    "when",
    "with",
}
TEMPORAL_MARKER_RE = re.compile(
    r"\b("
    r"q[1-4]|"
    r"quarter\s+(?:one|two|three|four)|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r")\b",
    re.IGNORECASE,
)
PROJECT_ENTITY_RE = re.compile(
    r"\b(project|program|initiative|platform|product)\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2})\b"
)
BUSINESS_RELEASE_REWRITE_RE = re.compile(
    r"\b(ship|release)\b(?=\s+(?:before|by|in|on|after)\s+(?:q[1-4]\b|quarter\s+(?:one|two|three|four)\b|"
    r"january|february|march|april|may|june|july|august|september|october|november|december))",
    re.IGNORECASE,
)
QUARTER_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedTranslationInput:
    raw_text: str
    normalized_text: str
    notes: list[str]

    @property
    def changed(self) -> bool:
        return self.raw_text != self.normalized_text


def normalize_translation_input(
    *,
    source_text: str,
    source_lang: str,
) -> NormalizedTranslationInput:
    raw = re.sub(r"\s+", " ", source_text.strip())
    if not raw:
        return NormalizedTranslationInput(raw_text="", normalized_text="", notes=[])
    if not _is_english_like(source_lang=source_lang, text=raw):
        return NormalizedTranslationInput(raw_text=raw, normalized_text=raw, notes=[])

    notes: list[str] = []
    text = raw

    updated = QUARTER_RE.sub(lambda match: f"Q{match.group(1)}", text)
    if updated != text:
        text = updated
        notes.append("quarter_uppercase")

    updated = re.sub(
        r"\b("
        + "|".join(MONTH_NAMES.keys())
        + r")\b",
        lambda match: MONTH_NAMES[match.group(1).lower()],
        text,
        flags=re.IGNORECASE,
    )
    if updated != text:
        text = updated
        notes.append("month_titlecase")

    updated = PROJECT_ENTITY_RE.sub(_restore_entity_surface_form, text)
    if updated != text:
        text = updated
        notes.append("entity_surface_form")

    updated = BUSINESS_RELEASE_REWRITE_RE.sub(lambda match: _preserve_case(match.group(1), "launch"), text)
    if updated != text:
        text = updated
        notes.append("business_release_rewrite")

    if _should_append_terminal_period(text):
        text = f"{text}."
        notes.append("terminal_period")

    return NormalizedTranslationInput(raw_text=raw, normalized_text=text, notes=notes)


def _is_english_like(*, source_lang: str, text: str) -> bool:
    if source_lang == "en":
        return True
    if source_lang == "auto":
        return bool(re.search(r"[A-Za-z]", text))
    return False


def _restore_entity_surface_form(match: re.Match[str]) -> str:
    head = match.group(1)
    tail = match.group(2)
    tail_tokens = tail.split()
    if not tail_tokens:
        return match.group(0)
    candidate_tokens: list[str] = []
    for token in tail_tokens:
        lowered = token.lower()
        if lowered in ENTITY_BREAK_TAILS:
            break
        candidate_tokens.append(token)
    if not candidate_tokens:
        return match.group(0)
    if all(token.lower() in GENERIC_ENTITY_TAILS for token in candidate_tokens):
        return match.group(0)
    normalized_tail = " ".join(_smart_title_token(token) for token in candidate_tokens)
    remainder_tokens = tail_tokens[len(candidate_tokens) :]
    remainder = f" {' '.join(remainder_tokens)}" if remainder_tokens else ""
    return f"{head.capitalize()} {normalized_tail}{remainder}"


def _smart_title_token(token: str) -> str:
    lowered = token.lower()
    if lowered in UPPERCASE_TOKENS:
        return lowered.upper()
    if re.fullmatch(r"q[1-4]", lowered):
        return lowered.upper()
    return lowered[:1].upper() + lowered[1:]


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _should_append_terminal_period(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.endswith(TERMINAL_PUNCTUATION):
        return False
    tokens = stripped.split()
    if len(tokens) < 4:
        return False
    last_token = re.sub(r"[^\w-]+", "", tokens[-1]).lower()
    if not last_token or last_token in NON_FINAL_END_TOKENS:
        return False
    return True
