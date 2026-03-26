import streamlit as st
from api_client import send_utterance_request, start_session_request, stop_session_request
from state_store import (
    add_event,
    apply_backend_event,
    cleanup_local_expired_boxes,
    init_state,
    sorted_visible_labels,
)
from ui_components import (
    render_controls,
    render_events,
    render_header,
    render_speaker_windows,
    render_utterance_form,
)

st.set_page_config(page_title="Realtime Translation MVP", layout="wide")


def main() -> None:
    init_state()
    cleanup_local_expired_boxes()

    render_header()
    target_lang, clicked_start, clicked_stop = render_controls(st.session_state.target_lang)
    if clicked_start:
        try:
            data = start_session_request(target_lang)
            st.session_state.session_id = data["session_id"]
            st.session_state.target_lang = target_lang
            st.session_state.speakers = {}
            st.session_state.visible_labels = []
            st.session_state.connected = True
            add_event(f"session started: {data['session_id']}")
        except Exception as exc:
            add_event(f"start failed: {exc}")

    if clicked_stop:
        session_id = st.session_state.get("session_id")
        if session_id:
            try:
                stop_session_request(session_id)
            except Exception as exc:
                add_event(f"stop request failed: {exc}")
        st.session_state.connected = False

    st.write(f"Session ID: `{st.session_state.session_id}`")
    st.write(f"Channel ready: `{st.session_state.connected}`")

    submitted, speaker_key, source_lang, text = render_utterance_form(bool(st.session_state.session_id))
    if submitted and text.strip():
        session_id = st.session_state.get("session_id")
        if not session_id:
            add_event("no active session, click Start")
        else:
            add_event(f"sending utterance from {speaker_key}...")
            with st.spinner("Translating..."):
                try:
                    events = send_utterance_request(session_id, speaker_key, text.strip(), source_lang)
                    add_event(f"received {len(events)} event(s)")
                    for event in events:
                        apply_backend_event(event)
                except Exception as exc:
                    add_event(f"send failed: {exc}")
    elif submitted and not text.strip():
        add_event("empty text, please type something")

    labels = sorted_visible_labels()
    render_speaker_windows(st.session_state.speakers, labels)
    render_events(st.session_state.events)


if __name__ == "__main__":
    main()
