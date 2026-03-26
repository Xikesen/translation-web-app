from typing import Any

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    target_lang: str = Field(pattern="^(en|fr|zh|ja)$")


class StartSessionResponse(BaseModel):
    session_id: str
    target_lang: str
    max_duration_seconds: int


class StopSessionResponse(BaseModel):
    session_id: str
    status: str


class UtteranceRequest(BaseModel):
    speaker_key: str
    text: str
    source_lang: str = "auto"


class UtteranceResponse(BaseModel):
    events: list[dict[str, Any]]
