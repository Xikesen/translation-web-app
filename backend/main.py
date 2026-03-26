from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from config import MAX_WINDOWS
from models import (
    StartSessionRequest,
    StartSessionResponse,
    StopSessionResponse,
    UtteranceRequest,
    UtteranceResponse,
)
from services.session_service import create_session, stop_session_by_id
from services.utterance_service import process_utterance
from state import manager, sessions
from utils import now_ts


app = FastAPI(title="Realtime Translation MVP")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
            if msg.get("type") != "utterance":
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
