import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from config import MAX_WINDOWS, SUPPORTED_LANGS
from models import (
    StartSessionRequest,
    StartSessionResponse,
    StopSessionResponse,
    UtteranceRequest,
    UtteranceResponse,
)
from services.session_service import create_session, stop_session_by_id
from services.audio_stream_service import process_audio_chunk
from services.benchmark_metrics import translation_benchmark_store
from services.translation_router import replace_glossary
from services.utterance_service import (
    build_translation_events,
    prepare_utterance,
    process_utterance,
)
from state import manager, sessions
from utils import now_ts


app = FastAPI(title="Realtime Translation MVP")
WEBRTC_HTML_PATH = Path(__file__).parent / "static" / "webrtc_client.html"
GLOSSARY_PATH = Path(__file__).parent / "data" / "glossary.json"


async def _translate_and_broadcast(
    session_id: str,
    speaker,
    source_text: str,
) -> None:
    session = sessions.get(session_id)
    if session is None or not session.is_running:
        return
    events = await build_translation_events(
        session=session,
        speaker=speaker,
        source_text=source_text,
    )
    for event in events:
        await manager.broadcast(session_id, event)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/benchmark/translation")
async def benchmark_translation() -> dict:
    return translation_benchmark_store.snapshot()


@app.post("/benchmark/translation/reset")
async def benchmark_translation_reset() -> dict[str, str]:
    translation_benchmark_store.reset()
    return {"status": "ok"}


@app.post("/glossary/upload")
async def upload_glossary(raw: dict) -> dict:
    try:
        summary = replace_glossary(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOSSARY_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "saved_path": str(GLOSSARY_PATH),
        **summary,
    }


@app.get("/webrtc", response_class=HTMLResponse)
async def webrtc_page() -> HTMLResponse:
    if not WEBRTC_HTML_PATH.exists():
        return HTMLResponse("<h3>webrtc client not found</h3>", status_code=404)
    return HTMLResponse(WEBRTC_HTML_PATH.read_text(encoding="utf-8"))


@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest) -> StartSessionResponse:
    return create_session(req.target_lang)


@app.post("/session/{session_id}/stop", response_model=StopSessionResponse)
async def stop_session(session_id: str) -> StopSessionResponse:
    return await stop_session_by_id(session_id)


@app.post("/session/{session_id}/utterance", response_model=UtteranceResponse)
async def post_utterance(session_id: str, req: UtteranceRequest) -> UtteranceResponse:
    session = sessions.get(session_id)
    if session is None or not session.is_running:
        return UtteranceResponse(
            events=[
                {
                    "type": "error",
                    "message": "session not found or stopped",
                    "ts": now_ts(),
                }
            ]
        )

    source_text = req.text.strip()
    if not source_text:
        return UtteranceResponse(events=[])

    events = await process_utterance(
        session=session,
        speaker_key=req.speaker_key.strip() or "anonymous",
        source_text=source_text,
        source_lang=req.source_lang.strip().lower(),
    )
    return UtteranceResponse(events=events)


@app.websocket("/ws/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    session = sessions.get(session_id)
    if session is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "session not found"})
        await websocket.close()
        return

    await manager.connect(session_id, websocket)
    await websocket.send_json(
        {
            "type": "connected",
            "session_id": session_id,
            "target_lang": session.target_lang,
            "max_windows": MAX_WINDOWS,
            "ts": now_ts(),
        }
    )

    try:
        while True:
            if not session.is_running:
                await websocket.close()
                break

            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            if msg_type == "set_target_lang":
                incoming_lang = str(msg.get("target_lang", "")).strip().lower()
                if incoming_lang in SUPPORTED_LANGS:
                    session.target_lang = incoming_lang
                    await manager.broadcast(
                        session_id,
                        {
                            "type": "target_lang_updated",
                            "target_lang": incoming_lang,
                            "ts": now_ts(),
                        },
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"unsupported target_lang: {incoming_lang}",
                            "ts": now_ts(),
                        }
                    )
                continue
            if msg_type != "utterance":
                continue

            events = await process_utterance(
                session=session,
                speaker_key=str(msg.get("speaker_key", "")).strip() or "anonymous",
                source_text=str(msg.get("text", "")).strip(),
                source_lang=str(msg.get("source_lang", "auto")).strip().lower(),
            )
            for event in events:
                await manager.broadcast(session_id, event)
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)


@app.websocket("/ws_audio/{session_id}")
async def session_audio_ws(websocket: WebSocket, session_id: str) -> None:
    session = sessions.get(session_id)
    if session is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "session not found"})
        await websocket.close()
        return

    await manager.connect(session_id, websocket)
    await websocket.send_json(
        {
            "type": "audio_connected",
            "session_id": session_id,
            "target_lang": session.target_lang,
            "supported_target_langs": sorted(SUPPORTED_LANGS),
            "sample_rate": 16000,
            "sample_format": "s16le",
            "channels": 1,
            "frame_ms": 20,
            "ts": now_ts(),
        }
    )

    try:
        while True:
            if not session.is_running:
                await websocket.close()
                break

            msg = await websocket.receive()
            text_payload = msg.get("text")
            if text_payload:
                try:
                    incoming = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "invalid control message json",
                            "ts": now_ts(),
                        }
                    )
                    continue
                if incoming.get("type") == "set_target_lang":
                    incoming_lang = str(incoming.get("target_lang", "")).strip().lower()
                    if incoming_lang in SUPPORTED_LANGS:
                        session.target_lang = incoming_lang
                        await manager.broadcast(
                            session_id,
                            {
                                "type": "target_lang_updated",
                                "target_lang": incoming_lang,
                                "ts": now_ts(),
                            },
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"unsupported target_lang: {incoming_lang}",
                                "ts": now_ts(),
                            }
                        )
                continue

            chunk = msg.get("bytes")
            if not isinstance(chunk, (bytes, bytearray)):
                continue

            transcribed_chunks = await process_audio_chunk(session, bytes(chunk))
            for item in transcribed_chunks:
                events, speaker, translation_source_text = prepare_utterance(
                    session=session,
                    speaker_key=item.speaker_id,
                    source_text=item.source_text,
                    source_lang=item.source_lang,
                    source_confidence=item.source_confidence,
                    force_commit=item.is_final,
                )
                for event in events:
                    await manager.broadcast(session_id, event)
                if speaker is not None and translation_source_text:
                    asyncio.create_task(
                        _translate_and_broadcast(
                            session_id=session_id,
                            speaker=speaker,
                            source_text=translation_source_text,
                        )
                    )
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
