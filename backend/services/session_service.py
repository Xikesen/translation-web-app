import asyncio
import uuid

from config import (
    SESSION_MAX_SECONDS,
    SPEAKER_END_AFTER_SECONDS,
    SPEAKER_KEEP_SECONDS,
)
from services.audio_stream_service import remove_pipeline
from services.utterance_service import build_translation_events, flush_pending_text
from models import StartSessionResponse, StopSessionResponse
from state import SessionState, manager, sessions
from utils import now_ts


async def emit_session_end(session: SessionState, reason: str) -> None:
    if not session.is_running:
        return
    for speaker in session.speakers_by_label.values():
        pending_text = flush_pending_text(speaker)
        if not pending_text:
            continue
        translation_events = await build_translation_events(
            session=session,
            speaker=speaker,
            source_text=pending_text,
        )
        for event in translation_events:
            await manager.broadcast(session.session_id, event)
    session.is_running = False
    await manager.broadcast(
        session.session_id,
        {
            "type": "session_end",
            "reason": reason,
            "session_id": session.session_id,
            "ts": now_ts(),
        },
    )
    remove_pipeline(session.session_id)


async def session_watchdog(session: SessionState) -> None:
    while session.is_running:
        if now_ts() - session.created_at >= SESSION_MAX_SECONDS:
            await emit_session_end(session, reason="max_duration_reached")
            break

        for speaker in session.speakers_by_label.values():
            if speaker.is_active and now_ts() - speaker.last_active_ts >= SPEAKER_END_AFTER_SECONDS:
                speaker.is_active = False
                pending_text = flush_pending_text(speaker)
                if pending_text:
                    translation_events = await build_translation_events(
                        session=session,
                        speaker=speaker,
                        source_text=pending_text,
                    )
                    for event in translation_events:
                        await manager.broadcast(session.session_id, event)
                if not speaker.end_sent:
                    speaker.end_sent = True
                    speaker.keep_until_ts = now_ts() + SPEAKER_KEEP_SECONDS
                    await manager.broadcast(
                        session.session_id,
                        {
                            "type": "speaker_end",
                            "speaker_label": speaker.label,
                            "last_lines": list(speaker.last_two_lines),
                            "keep_until_ts": speaker.keep_until_ts,
                            "ts": now_ts(),
                        },
                    )
            elif (
                not speaker.is_active
                and speaker.keep_until_ts is not None
                and now_ts() >= speaker.keep_until_ts
            ):
                await manager.broadcast(
                    session.session_id,
                    {
                        "type": "speaker_expire",
                        "speaker_label": speaker.label,
                        "ts": now_ts(),
                    },
                )
                speaker.keep_until_ts = None
        await asyncio.sleep(0.25)


def create_session(target_lang: str) -> StartSessionResponse:
    session_id = str(uuid.uuid4())
    session = SessionState(
        session_id=session_id,
        target_lang=target_lang,
        created_at=now_ts(),
    )
    sessions[session_id] = session
    asyncio.create_task(session_watchdog(session))
    return StartSessionResponse(
        session_id=session_id,
        target_lang=target_lang,
        max_duration_seconds=SESSION_MAX_SECONDS,
    )


async def stop_session_by_id(session_id: str) -> StopSessionResponse:
    session = sessions.get(session_id)
    if session:
        await emit_session_end(session, reason="manual_stop")
    return StopSessionResponse(session_id=session_id, status="stopped")
