import time
from typing import Any

import streamlit as st

from config import MAX_WINDOWS
from speaker_types import SpeakerBox


def init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": None,
        "target_lang": "fr",
        "speakers": {},
        "visible_labels": [],
        "events": [],
        "connected": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_event(msg: str) -> None:
    st.session_state.events.append(msg)
    st.session_state.events = st.session_state.events[-20:]


def apply_backend_event(payload: dict[str, Any]) -> None:
    etype = payload.get("type")
    now = time.time()

    if etype == "speaker_start":
        label = payload["speaker_label"]
        speaker = st.session_state.speakers.get(label) or SpeakerBox(label=label)
        speaker.is_active = True
        speaker.keep_until_ts = None
        speaker.last_active_ts = now
        st.session_state.speakers[label] = speaker
        add_event(f"speaker {label} start")
        return

    if etype == "utterance":
        label = payload["speaker_label"]
        speaker = st.session_state.speakers.get(label) or SpeakerBox(label=label)
        speaker.is_active = True
        speaker.keep_until_ts = None
        speaker.source_lang = payload.get("source_lang", "auto")
        speaker.last_active_ts = payload.get("ts", now)
        speaker.lines.append(payload.get("translated_text", ""))
        speaker.lines = speaker.lines[-8:]
        st.session_state.speakers[label] = speaker
        st.session_state.visible_labels = payload.get("visible_labels", [])
        return

    if etype == "speaker_end":
        label = payload["speaker_label"]
        speaker = st.session_state.speakers.get(label) or SpeakerBox(label=label)
        speaker.is_active = False
        speaker.keep_until_ts = payload.get("keep_until_ts")
        speaker.lines = payload.get("last_lines", speaker.lines[-2:])
        st.session_state.speakers[label] = speaker
        add_event(f"speaker {label} end; keep for 3s")
        return

    if etype == "speaker_expire":
        label = payload["speaker_label"]
        if label in st.session_state.speakers:
            del st.session_state.speakers[label]
        add_event(f"speaker {label} expired")
        return

    if etype == "session_end":
        add_event(f"session ended: {payload.get('reason')}")
        st.session_state.connected = False
        return

    if etype == "error":
        add_event(f"error: {payload.get('message')}")


def cleanup_local_expired_boxes() -> None:
    now = time.time()
    to_delete: list[str] = []
    for label, speaker in st.session_state.speakers.items():
        if speaker.keep_until_ts and now >= speaker.keep_until_ts:
            to_delete.append(label)
    for label in to_delete:
        del st.session_state.speakers[label]


def sorted_visible_labels() -> list[str]:
    items = list(st.session_state.speakers.values())
    active = sorted(
        [s for s in items if s.is_active],
        key=lambda s: s.last_active_ts,
        reverse=True,
    )
    cooling = sorted(
        [s for s in items if not s.is_active],
        key=lambda s: s.last_active_ts,
        reverse=True,
    )
    merged = active + cooling
    return [s.label for s in merged[:MAX_WINDOWS]]
