from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Any

import websockets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = ROOT / "data" / "synthetic_audio_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "run" / "synthetic_acceptance"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases file must be a JSON array")
    cases: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        target_lang = str(item.get("target_lang") or "zh").strip() or "zh"
        if case_id and text:
            cases.append({"id": case_id, "text": text, "target_lang": target_lang})
    if not cases:
        raise ValueError("no usable synthetic audio cases found")
    return cases


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def _synthesize_tts(*, text: str, wav_path: Path, voice: str | None, rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    voice_select = ""
    if voice:
        voice_select = f"$synth.SelectVoice('{_ps_quote(voice)}');"
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"{voice_select}"
        f"$synth.Rate = {int(rate)}; "
        "$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono"
        "); "
        f"$synth.SetOutputToWaveFile('{_ps_quote(str(wav_path))}', $format); "
        f"$synth.Speak('{_ps_quote(text)}'); "
        "$synth.Dispose();"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )


def _start_session(*, base_url: str, target_lang: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/session/start",
        data=json.dumps({"target_lang": target_lang}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


async def _replay_wav(*, base_url: str, wav_path: Path, target_lang: str, timeout_s: float) -> dict[str, Any]:
    session = _start_session(base_url=base_url, target_lang=target_lang)
    session_id = str(session["session_id"])
    ws_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    uri = f"{ws_url}/ws_audio/{session_id}"
    events: list[dict[str, Any]] = []

    async with websockets.connect(uri, max_size=None, ping_interval=None, close_timeout=30) as ws:
        with wave.open(str(wav_path), "rb") as wav_file:
            chunk_frames = 320
            while True:
                data = wav_file.readframes(chunk_frames)
                if not data:
                    break
                await ws.send(data)
                await asyncio.sleep(0.02)

        silence = struct.pack("<h", 0) * 320
        for _ in range(60):
            await ws.send(silence)
            await asyncio.sleep(0.02)

        try:
            while True:
                payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
                events.append(payload)
                if payload.get("type") == "translation":
                    break
        except asyncio.TimeoutError:
            pass

    transcript = next((event for event in events if event.get("type") == "transcript"), None)
    translation = next((event for event in events if event.get("type") == "translation"), None)
    return {
        "session": session,
        "events": events,
        "transcript": transcript,
        "translation": translation,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "case"


async def _run_cases(
    *,
    cases: list[dict[str, Any]],
    output_dir: Path,
    base_url: str,
    voice: str | None,
    rate: int,
    timeout_s: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        wav_path = output_dir / f"{_slugify(case_id)}.wav"
        _synthesize_tts(text=str(case["text"]), wav_path=wav_path, voice=voice, rate=rate)
        replay = await _replay_wav(
            base_url=base_url,
            wav_path=wav_path,
            target_lang=str(case["target_lang"]),
            timeout_s=timeout_s,
        )
        results.append(
            {
                "id": case_id,
                "text": case["text"],
                "target_lang": case["target_lang"],
                "wav_path": str(wav_path),
                **replay,
            }
        )
    return {
        "base_url": base_url,
        "voice": voice,
        "rate": rate,
        "case_count": len(results),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic TTS audio and replay it through ooni-glass.")
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--voice")
    parser.add_argument("--rate", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    cases = _load_cases(args.cases_file)
    report = await _run_cases(
        cases=cases,
        output_dir=args.output_dir,
        base_url=args.base_url,
        voice=args.voice,
        rate=args.rate,
        timeout_s=args.timeout_s,
    )
    report_path = args.output_dir / "synthetic_audio_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "case_count": report["case_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
