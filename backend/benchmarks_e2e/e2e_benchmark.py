"""End-to-end (audio) benchmark for the realtime-translation pipeline.

Streams synthesized speech through the REAL backend path
(/ws_audio -> VAD -> Whisper ASR -> translation manager + skill) and scores the
full chain:

  - ASR quality : WER (English, word-level) / CER (Chinese, char-level)
                  between the reference transcript and what Whisper produced.
  - MT quality  : chrF between the reference translation and the pipeline's
                  translation output (+ optional LLM judge).
  - Latency     : audio duration, time-to-first-transcript, and tail latency
                  (last audio frame -> final translation), i.e. how long after
                  the speaker stops until the translation lands.

Prerequisites
  1. Generate audio:      python generate_samples.py
  2. Start the backend:   uvicorn main:app --port 8000   (with Ollama running)
  3. Run this:            python benchmarks_e2e/e2e_benchmark.py

Usage
  python e2e_benchmark.py --server http://localhost:8000
  python e2e_benchmark.py --no-realtime          # stream as fast as possible
  python e2e_benchmark.py --judge --judge-model qwen3:8b
  python e2e_benchmark.py --out e2e_results.json --csv e2e_results.csv

This tests the deployed pipeline over WebSocket; it does not import backend code,
so whatever ASR/translation backend the server runs is what gets measured.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
import time
import wave
from collections import Counter
from pathlib import Path

import httpx
import websockets

DATA_DIR = Path(__file__).parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
AUDIO_DIR = DATA_DIR / "audio"

FRAME_MS = 20
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
FRAME_BYTES = (SAMPLE_RATE * FRAME_MS // 1000) * BYTES_PER_SAMPLE  # 640 bytes
TRAILING_SILENCE_MS = 1600  # push silence to trigger ASR endpointing


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _norm_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rate(reference: str, hypothesis: str, lang: str) -> float:
    """WER for English (word units), CER for Chinese (char units). 0=perfect."""
    ref_n = _norm_for_wer(reference)
    hyp_n = _norm_for_wer(hypothesis)
    if lang == "zh":
        ref_units = [c for c in ref_n if not c.isspace()]
        hyp_units = [c for c in hyp_n if not c.isspace()]
    else:
        ref_units = ref_n.split()
        hyp_units = hyp_n.split()
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return round(_edit_distance(ref_units, hyp_units) / len(ref_units), 4)


def _char_ngrams(text: str, n: int) -> Counter:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < n:
        return Counter()
    return Counter("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))


def chrf_score(hypothesis: str, reference: str, *, max_n: int = 6, beta: float = 2.0) -> float:
    hyp, ref = hypothesis.strip().lower(), reference.strip().lower()
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp_ng, ref_ng = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not hyp_ng or not ref_ng:
            continue
        overlap = sum((hyp_ng & ref_ng).values())
        precisions.append(overlap / max(1, sum(hyp_ng.values())))
        recalls.append(overlap / max(1, sum(ref_ng.values())))
    if not precisions or not recalls:
        return 0.0
    avg_p, avg_r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    denom = beta * beta * avg_p + avg_r
    if denom == 0:
        return 0.0
    return round(100.0 * (1 + beta * beta) * avg_p * avg_r / denom, 2)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def judge_adequacy(client: httpx.AsyncClient, ollama_url: str, model: str, source: str, reference: str, hypothesis: str) -> float | None:
    system = (
        "You are a strict bilingual translation evaluator. Score how accurately the "
        "candidate conveys the meaning of the source, using the reference as a guide. "
        "100 = perfect meaning and fidelity, 0 = wrong or missing. Reply with ONLY an integer 0-100."
    )
    user = f"Source:\n{source}\n\nReference:\n{reference}\n\nCandidate:\n{hypothesis}\n\nScore (0-100):"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 16},
    }
    try:
        resp = await client.post(f"{ollama_url}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        content = str(resp.json().get("message", {}).get("content", ""))
        content = re.sub(r"<think>.*?</think>", " ", content, flags=re.DOTALL)
        numbers = re.findall(r"\d{1,3}", content)
        return float(max(0, min(100, int(numbers[-1])))) if numbers else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Audio + WebSocket driving
# --------------------------------------------------------------------------- #
def read_wav_frames(path: Path) -> tuple[list[bytes], float]:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path.name} must be 16kHz mono s16le; got "
                f"{w.getframerate()}Hz {w.getnchannels()}ch {w.getsampwidth()*8}bit"
            )
        pcm = w.readframes(w.getnframes())
    duration_s = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if frames and len(frames[-1]) < FRAME_BYTES:
        frames[-1] = frames[-1] + b"\x00" * (FRAME_BYTES - len(frames[-1]))
    return frames, duration_s


async def run_sample(
    http: httpx.AsyncClient,
    server: str,
    ws_base: str,
    sample: dict,
    realtime: bool,
) -> dict:
    audio_path = AUDIO_DIR / f"{sample['id']}.wav"
    frames, duration_s = read_wav_frames(audio_path)

    start = await http.post(f"{server}/session/start", json={"target_lang": sample["target_lang"]})
    start.raise_for_status()
    session_id = start.json()["session_id"]

    events: list[tuple[float, dict]] = []
    uri = f"{ws_base}/ws_audio/{session_id}"
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            stop = asyncio.Event()

            async def receiver() -> None:
                try:
                    while not stop.is_set():
                        msg = await ws.recv()
                        if isinstance(msg, (bytes, bytearray)):
                            continue
                        events.append((time.monotonic(), json.loads(msg)))
                except (websockets.ConnectionClosed, asyncio.CancelledError):
                    return
                except Exception:
                    return

            recv_task = asyncio.create_task(receiver())

            t_stream_start = time.monotonic()
            for frame in frames:
                await ws.send(frame)
                if realtime:
                    await asyncio.sleep(FRAME_MS / 1000.0)
            t_audio_end = time.monotonic()

            silence = b"\x00" * FRAME_BYTES
            for _ in range(TRAILING_SILENCE_MS // FRAME_MS):
                await ws.send(silence)
                if realtime:
                    await asyncio.sleep(FRAME_MS / 1000.0)

            # Wait until events settle (no new event for `idle`s) or hard cap.
            # CPU Whisper decode (~2-4s) + Ollama translation (first call cold) can
            # be slow, so keep the idle window generous or events get cut off.
            idle, max_wait = 8.0, 60.0
            last_count, last_change = len(events), time.monotonic()
            while True:
                await asyncio.sleep(0.2)
                now = time.monotonic()
                if len(events) != last_count:
                    last_count, last_change = len(events), now
                if now - last_change >= idle:
                    break
                if now - t_audio_end >= max_wait:
                    break

            stop.set()
            recv_task.cancel()
    finally:
        try:
            await http.post(f"{server}/session/{session_id}/stop")
        except Exception:
            pass

    # Reconstruct transcript & translation from the event stream.
    transcripts = [(ts, e) for ts, e in events if e.get("type") == "transcript"]
    translations = [(ts, e) for ts, e in events if e.get("type") == "translation"]
    hyp_transcript = " ".join(str(e.get("source_text", "")).strip() for _, e in transcripts).strip()
    hyp_translation = " ".join(str(e.get("translated_text", "")).strip() for _, e in translations).strip()

    t_first_transcript = (transcripts[0][0] - t_stream_start) * 1000 if transcripts else None
    t_last_translation_tail = (translations[-1][0] - t_audio_end) * 1000 if translations else None

    return {
        "id": sample["id"],
        "lang": sample["lang"],
        "target_lang": sample["target_lang"],
        "reference_text": sample["text"],
        "reference_translation": sample["reference_translation"],
        "hyp_transcript": hyp_transcript,
        "hyp_translation": hyp_translation,
        "audio_s": round(duration_s, 2),
        "error_rate": error_rate(sample["text"], hyp_transcript, sample["lang"]),
        "chrf": chrf_score(hyp_translation, sample["reference_translation"]),
        "first_transcript_ms": round(t_first_transcript, 1) if t_first_transcript is not None else None,
        "tail_latency_ms": round(t_last_translation_tail, 1) if t_last_translation_tail is not None else None,
        "n_transcript_events": len(transcripts),
        "n_translation_events": len(translations),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def summarize(rows: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    er, chrf = col("error_rate"), col("chrf")
    first, tail = col("first_transcript_ms"), col("tail_latency_ms")
    judge = col("judge")
    out = {
        "count": len(rows),
        "error_rate_mean": round(statistics.mean(er), 4) if er else None,
        "chrf_mean": round(statistics.mean(chrf), 2) if chrf else None,
        "first_transcript_ms_mean": round(statistics.mean(first), 1) if first else None,
        "tail_latency_ms_mean": round(statistics.mean(tail), 1) if tail else None,
        "tail_latency_ms_p95": round(percentile(tail, 95), 1) if tail else None,
    }
    if judge:
        out["judge_mean"] = round(statistics.mean(judge), 2)
    return out


async def main_async(args: argparse.Namespace) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]

    missing = [s["id"] for s in samples if not (AUDIO_DIR / f"{s['id']}.wav").exists()]
    if missing:
        print(f"Error: missing audio for {missing}. Run: python generate_samples.py")
        return 1

    server = args.server.rstrip("/")
    ws_base = "ws" + server[len("http"):]  # http->ws, https->wss

    print(f"Server: {server}   samples: {len(samples)}   realtime: {not args.no_realtime}")
    async with httpx.AsyncClient(timeout=30.0) as http:
        try:
            (await http.get(f"{server}/health")).raise_for_status()
        except Exception as exc:
            print(f"\nError: backend not reachable at {server}: {exc}")
            print("Start it with: uvicorn main:app --port 8000")
            return 1

        rows: list[dict] = []
        for sample in samples:
            print(f"  running {sample['id']} ({sample['lang']}->{sample['target_lang']}) ...", flush=True)
            row = await run_sample(http, server, ws_base, sample, realtime=not args.no_realtime)
            if args.judge:
                row["judge"] = await judge_adequacy(
                    http, args.ollama_url, args.judge_model,
                    row["reference_text"], row["reference_translation"], row["hyp_translation"],
                )
            rows.append(row)
            unit = "CER" if sample["lang"] == "zh" else "WER"
            print(
                f"    {unit}={row['error_rate']:.3f}  chrF={row['chrf']:.1f}  "
                f"tail={row['tail_latency_ms']}ms  ASR=\"{row['hyp_transcript'][:42]}\""
            )

    overall = summarize(rows)
    by_lang = {lang: summarize([r for r in rows if r["lang"] == lang]) for lang in sorted({r["lang"] for r in rows})}

    print("\n" + "=" * 72)
    print("END-TO-END SUMMARY (audio -> ASR -> translation)")
    print("=" * 72)
    print(f"  samples              : {overall['count']}")
    print(f"  ASR error rate (mean): {overall['error_rate_mean']}   (WER en / CER zh; lower better)")
    print(f"  translation chrF     : {overall['chrf_mean']}   (higher better)")
    if "judge_mean" in overall:
        print(f"  translation judge    : {overall['judge_mean']}   ({args.judge_model})")
    print(f"  first transcript     : {overall['first_transcript_ms_mean']} ms")
    print(f"  tail latency (mean)  : {overall['tail_latency_ms_mean']} ms   (speaker stop -> final translation)")
    print(f"  tail latency (p95)   : {overall['tail_latency_ms_p95']} ms")
    for lang, s in by_lang.items():
        print(f"  [{lang}] err={s['error_rate_mean']}  chrf={s['chrf_mean']}  tail={s['tail_latency_ms_mean']}ms")
    print("=" * 72)

    if args.out:
        Path(args.out).write_text(
            json.dumps({"overall": overall, "by_lang": by_lang, "rows": rows, "config": vars(args)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.out}")
    if args.csv:
        fields = ["id", "lang", "target_lang", "audio_s", "error_rate", "chrf", "judge",
                  "first_transcript_ms", "tail_latency_ms", "reference_text", "hyp_transcript",
                  "reference_translation", "hyp_translation"]
        with Path(args.csv).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({**r, "judge": r.get("judge", "")})
        print(f"CSV written to {args.csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end audio benchmark (ASR + translation + latency).")
    parser.add_argument("--server", default="http://localhost:8000", help="backend base URL")
    parser.add_argument("--no-realtime", action="store_true", help="stream audio as fast as possible (default: real-time pace)")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge translation score")
    parser.add_argument("--judge-model", default="qwen3:8b", help="model for the LLM judge")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL for the judge")
    parser.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    parser.add_argument("--out", default=None, help="write full results JSON here")
    parser.add_argument("--csv", default=None, help="write per-sample CSV here")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
