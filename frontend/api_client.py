import requests

from config import BACKEND_HTTP, UTTERANCE_TIMEOUT_SECONDS


def start_session_request(target_lang: str) -> dict:
    resp = requests.post(
        f"{BACKEND_HTTP}/session/start",
        json={"target_lang": target_lang},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def stop_session_request(session_id: str) -> None:
    requests.post(f"{BACKEND_HTTP}/session/{session_id}/stop", timeout=10)


def send_utterance_request(
    session_id: str,
    speaker_key: str,
    text: str,
    source_lang: str,
) -> list[dict]:
    resp = requests.post(
        f"{BACKEND_HTTP}/session/{session_id}/utterance",
        json={
            "speaker_key": speaker_key,
            "text": text,
            "source_lang": source_lang,
        },
        timeout=UTTERANCE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("events", [])
