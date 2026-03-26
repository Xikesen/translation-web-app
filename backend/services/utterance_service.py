from config import MAX_SPEAKERS, MAX_WINDOWS, SPEAKER_LABELS, SUPPORTED_LANGS
from services.translator import translate_text
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


async def process_utterance(
    session: SessionState,
    speaker_key: str,
    source_text: str,
    source_lang: str,
) -> list[dict]:
    events: list[dict] = []
    speaker = assign_speaker(session, speaker_key)
    if speaker is None:
        return [
            {
                "type": "error",
                "message": "max 4 speakers reached",
                "ts": now_ts(),
            }
        ]

    speaker.source_lang = source_lang if source_lang in SUPPORTED_LANGS else "auto"
    speaker.last_active_ts = now_ts()
    was_active = speaker.is_active
    speaker.is_active = True
    speaker.end_sent = False
    speaker.keep_until_ts = None

    if not was_active:
        events.append(
            {
                "type": "speaker_start",
                "speaker_label": speaker.label,
                "ts": now_ts(),
            }
        )

    try:
        translated_text = await translate_text(
            source_text,
            source_lang=speaker.source_lang,
            target_lang=session.target_lang,
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

    ordered = sorted(
        session.speakers_by_label.values(),
        key=lambda s: s.last_active_ts,
        reverse=True,
    )
    visible_labels = [s.label for s in ordered[:MAX_WINDOWS]]

    events.append(
        {
            "type": "utterance",
            "speaker_label": speaker.label,
            "source_lang": speaker.source_lang,
            "target_lang": session.target_lang,
            "source_text": source_text,
            "translated_text": translated_text,
            "visible_labels": visible_labels,
            "ts": now_ts(),
        }
    )
    return events
