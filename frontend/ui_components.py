import streamlit as st

from config import MAX_WINDOWS, SUPPORTED_LANGS
from speaker_types import SpeakerBox


def render_header() -> None:
    st.title("Realtime Translation MVP")
    st.caption("MVP: text input simulates real-time utterances, preserving speaker lifecycle.")


def render_controls(default_target_lang: str) -> tuple[str, bool, bool]:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        target_lang = st.selectbox(
            "Target language",
            options=SUPPORTED_LANGS,
            index=SUPPORTED_LANGS.index(default_target_lang),
        )
    with c2:
        clicked_start = st.button("Start", use_container_width=True)
    with c3:
        clicked_stop = st.button("Stop", use_container_width=True)
    return target_lang, clicked_start, clicked_stop


def render_utterance_form(has_session: bool) -> tuple[bool, str, str, str]:
    with st.form("utterance_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1, 1, 3])
        speaker_key = col1.selectbox("Speaker Key", ["mic-1", "mic-2", "mic-3", "mic-4"])
        source_lang = col2.selectbox("Source Lang", ["auto", "en", "fr", "zh", "ja"], index=0)
        text = col3.text_input("Speak (simulated text)")
        submitted = st.form_submit_button("Send Utterance", disabled=not has_session)
        return submitted, speaker_key, source_lang, text


def render_speaker_windows(speakers: dict[str, SpeakerBox], labels: list[str]) -> None:
    st.subheader("Speaker Windows (max 3)")
    cols = st.columns(MAX_WINDOWS)
    for idx in range(MAX_WINDOWS):
        with cols[idx]:
            if idx >= len(labels):
                st.info("Empty")
                continue
            label = labels[idx]
            speaker = speakers[label]
            state = "ACTIVE" if speaker.is_active else "COOLDOWN"
            st.markdown(f"### Speaker {label}")
            st.write(f"State: `{state}`")
            st.write(f"Source: `{speaker.source_lang}`")
            for line in speaker.lines[-4:]:
                st.write(f"- {line}")


def render_events(events: list[str]) -> None:
    st.subheader("Events")
    for event in reversed(events):
        st.text(event)
