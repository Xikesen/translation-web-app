from collections import Counter

from config import (
    LID_MIN_CONFIDENCE,
    LID_SMOOTH_WINDOW,
    MAX_SPEAKERS,
    MAX_WINDOWS,
    MERGE_FORCE_CHARS,
    MERGE_MAX_WAIT_SECONDS,
    MERGE_SHORT_MAX_CHARS,
    SPEAKER_LABELS,
    SUPPORTED_LANGS,
    TARGET_LANG_DEFAULT,
    TRANSLATION_EVENT_INCLUDE_ROUTE_METRICS,
)
from services.source_state_manager import merge_source_context
from services.translation_input_normalizer import normalize_translation_input
from services.translation_manager import SOURCE_CONTEXT_MAX_CHARS, translate_with_manager_detailed
from state import SessionState, SpeakerState
from utils import now_ts


def assign_speaker(session: SessionState, speaker_key: str) -> SpeakerState | None:
    if speaker_key in session.speakers_by_key:
        return session.speakers_by_key[speaker_key]
    if len(session.speakers_by_key) >= MAX_SPEAKERS:
        return None

    label = SPEAKER_LABELS[len(session.speakers_by_key)]
    speaker = SpeakerState(label=label)
    session.speakers_by_key[speaker_key] = speaker
    session.speakers_by_label[label] = speaker
    return speaker


def _visible_labels(session: SessionState) -> list[str]:
    ordered = sorted(
        session.speakers_by_label.values(),
        key=lambda s: s.last_active_ts,
        reverse=True,
    )
    return [s.label for s in ordered[:MAX_WINDOWS]]


def _normalize_source_lang(source_lang: str) -> str:
    lang = source_lang.strip().lower()
    if lang in SUPPORTED_LANGS:
        return lang
    return "auto"


def _update_stable_source_lang(
    speaker: SpeakerState,
    observed_lang: str,
    source_confidence: float | None,
) -> None:
    if observed_lang not in SUPPORTED_LANGS:
        if speaker.stable_source_lang in SUPPORTED_LANGS:
            speaker.source_lang = speaker.stable_source_lang
        else:
            speaker.source_lang = "auto"
        return

    confidence = source_confidence if source_confidence is not None else 0.0
    speaker.lid_history.append((observed_lang, confidence))
    recent = list(speaker.lid_history)[-LID_SMOOTH_WINDOW:]
    qualified = [lang for lang, conf in recent if conf >= LID_MIN_CONFIDENCE]
    if not qualified:
        if speaker.stable_source_lang in SUPPORTED_LANGS:
            speaker.source_lang = speaker.stable_source_lang
        else:
            speaker.source_lang = observed_lang
        return

    counts = Counter(qualified)
    stable_lang, _ = max(counts.items(), key=lambda item: (item[1], item[0]))
    speaker.stable_source_lang = stable_lang
    speaker.source_lang = stable_lang


def _append_pending_text(speaker: SpeakerState, source_text: str, current_ts: float) -> None:
    text = source_text.strip()
    if not text:
        return
    if not speaker.pending_text:
        speaker.pending_text = text
        speaker.pending_started_ts = current_ts
        speaker.pending_updated_ts = current_ts
        return
    needs_space = not speaker.pending_text.endswith((" ", "\n")) and not text.startswith(
        (".", ",", "!", "?", ";", ":", "。", "，", "！", "？", "；", "：")
    )
    if needs_space:
        speaker.pending_text += " "
    speaker.pending_text += text
    speaker.pending_updated_ts = current_ts


def _should_flush_pending(speaker: SpeakerState, latest_text: str, current_ts: float) -> bool:
    pending = speaker.pending_text.strip()
    if not pending:
        return False
    if len(pending) >= MERGE_FORCE_CHARS:
        return True
    if pending.endswith((".", "!", "?", ";", ":", "。", "！", "？", "；", "：")):
        return True
    if len(latest_text.strip()) >= MERGE_SHORT_MAX_CHARS:
        return True
    if speaker.pending_started_ts is not None and (current_ts - speaker.pending_started_ts) >= MERGE_MAX_WAIT_SECONDS:
        return True
    return False


def flush_pending_text(speaker: SpeakerState) -> str:
    text = speaker.pending_text.strip()
    speaker.pending_text = ""
    speaker.pending_started_ts = None
    speaker.pending_updated_ts = None
    return text


def prepare_utterance(
    session: SessionState,
    speaker_key: str,
    source_text: str,
    source_lang: str,
    source_confidence: float | None = None,
    force_commit: bool = False,
) -> tuple[list[dict], SpeakerState | None, str | None]:
    events: list[dict] = []
    speaker = assign_speaker(session, speaker_key)
    if speaker is None:
        return (
            [
                {
                    "type": "error",
                    "message": f"max {MAX_SPEAKERS} speakers reached",
                    "ts": now_ts(),
                }
            ],
            None,
            None,
        )

    normalized_lang = _normalize_source_lang(source_lang)
    if normalized_lang == "auto":
        _update_stable_source_lang(speaker, observed_lang="auto", source_confidence=source_confidence)
    else:
        _update_stable_source_lang(speaker, observed_lang=normalized_lang, source_confidence=source_confidence)

    current_ts = now_ts()
    speaker.last_active_ts = current_ts
    was_active = speaker.is_active
    speaker.is_active = True
    speaker.end_sent = False
    speaker.keep_until_ts = None

    if not was_active:
        events.append(
            {
                "type": "speaker_start",
                "speaker_label": speaker.label,
                "ts": current_ts,
            }
        )

    events.append(
        {
            "type": "transcript",
            "speaker_label": speaker.label,
            "source_lang": speaker.source_lang,
            "source_text": source_text,
            "target_lang": session.target_lang,
            "visible_labels": _visible_labels(session),
            "ts": current_ts,
        }
    )

    _append_pending_text(speaker, source_text, current_ts)
    translation_source_text: str | None = None
    if force_commit or _should_flush_pending(speaker, source_text, current_ts):
        translation_source_text = flush_pending_text(speaker)
    return events, speaker, translation_source_text


async def translate_partial(
    *,
    source_text: str,
    source_lang: str,
    target_lang: str,
    committed_source_text: str = "",
    previous_target_text: str = "",
) -> str | None:
    """Translate an interim (non-final) segment for live display.

    Does not mutate any speaker/session state. Besides producing an early
    translation, it warms the translation cache so the eventual final segment
    (often identical text) is served from cache with ~0ms latency.
    """
    normalized = normalize_translation_input(source_text=source_text, source_lang=source_lang)
    text = normalized.normalized_text
    if not text:
        return None
    src = source_lang if source_lang in SUPPORTED_LANGS else "auto"
    tgt = target_lang if target_lang in SUPPORTED_LANGS else TARGET_LANG_DEFAULT
    try:
        route_result = await translate_with_manager_detailed(
            source_text=text,
            source_lang=src,
            target_lang=tgt,
            committed_source_text=committed_source_text,
            previous_target_text=previous_target_text,
        )
    except Exception:
        return None
    return route_result.translated_text


async def build_translation_events(
    session: SessionState,
    speaker: SpeakerState,
    source_text: str,
) -> list[dict]:
    events: list[dict] = []
    route_name = "manager_fallback_error"
    latency_ms = 0.0
    normalized_input = normalize_translation_input(
        source_text=source_text,
        source_lang=speaker.source_lang,
    )
    translation_source_text = normalized_input.normalized_text
    try:
        source_lang = speaker.source_lang if speaker.source_lang in SUPPORTED_LANGS else "auto"
        target_lang = session.target_lang if session.target_lang in SUPPORTED_LANGS else TARGET_LANG_DEFAULT
        previous_target_text = speaker.last_two_lines[-1] if speaker.last_two_lines else ""
        route_result = await translate_with_manager_detailed(
            source_text=translation_source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            committed_source_text=speaker.committed_source_text,
            previous_target_text=previous_target_text,
        )
        translated_text = route_result.translated_text
        route_name = route_result.route
        latency_ms = route_result.latency_ms
        speaker.committed_source_text = merge_source_context(
            speaker.committed_source_text,
            translation_source_text,
            max_chars=SOURCE_CONTEXT_MAX_CHARS,
        )
    except Exception as exc:
        translated_text = source_text
        events.append(
            {
                "type": "error",
                "message": f"translation failed: {type(exc).__name__}: {exc}",
                "ts": now_ts(),
            }
        )

    speaker.last_two_lines.append(translated_text)

    payload = {
        "type": "translation",
        "speaker_label": speaker.label,
        "source_lang": speaker.source_lang,
        "target_lang": session.target_lang,
        "source_text": source_text,
        "translated_text": translated_text,
        "visible_labels": _visible_labels(session),
        "ts": now_ts(),
    }
    if normalized_input.changed:
        payload["normalized_source_text"] = translation_source_text
        payload["normalization_notes"] = normalized_input.notes
    if TRANSLATION_EVENT_INCLUDE_ROUTE_METRICS:
        payload["route"] = route_name
        payload["latency_ms"] = latency_ms
        if "route_result" in locals():
            payload["gate_action"] = route_result.gate_decision.action
            payload["gate_reasons"] = route_result.gate_decision.reason_codes
            payload["write_mode"] = route_result.write_plan.mode
            payload["risk_notes"] = route_result.risk_notes
    events.append(payload)
    return events


async def process_utterance(
    session: SessionState,
    speaker_key: str,
    source_text: str,
    source_lang: str,
) -> list[dict]:
    events, speaker, translation_source_text = prepare_utterance(
        session=session,
        speaker_key=speaker_key,
        source_text=source_text,
        source_lang=source_lang,
        source_confidence=1.0 if source_lang in SUPPORTED_LANGS else None,
        force_commit=True,
    )
    if speaker is None:
        return events
    if translation_source_text:
        events.extend(
            await build_translation_events(
                session=session,
                speaker=speaker,
                source_text=translation_source_text,
            )
        )
    return events
