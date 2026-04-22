from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket


@dataclass
class SpeakerState:
    label: str
    source_lang: str = "auto"
    stable_source_lang: str = "auto"
    committed_source_text: str = ""
    last_active_ts: float = 0.0
    is_active: bool = False
    end_sent: bool = False
    keep_until_ts: float | None = None
    last_two_lines: deque[str] = field(default_factory=lambda: deque(maxlen=2))
    lid_history: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=8))
    pending_text: str = ""
    pending_started_ts: float | None = None
    pending_updated_ts: float | None = None


@dataclass
class SessionState:
    session_id: str
    target_lang: str
    created_at: float
    speakers_by_key: dict[str, SpeakerState] = field(default_factory=dict)
    speakers_by_label: dict[str, SpeakerState] = field(default_factory=dict)
    is_running: bool = True


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        if session_id in self.connections:
            self.connections[session_id].discard(websocket)
            if not self.connections[session_id]:
                del self.connections[session_id]

    async def broadcast(self, session_id: str, event: dict[str, Any]) -> None:
        dead = []
        for ws in self.connections.get(session_id, set()):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)


manager = ConnectionManager()
sessions: dict[str, SessionState] = {}
