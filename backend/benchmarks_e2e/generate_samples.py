"""Synthesize the end-to-end audio sample set with macOS `say` + `afconvert`.

For every entry in data/manifest.json this produces a 16 kHz / mono / s16le WAV
under data/audio/<id>.wav — exactly the format the /ws_audio pipeline expects.

Usage
  python generate_samples.py            # only synthesize missing files
  python generate_samples.py --force    # re-synthesize everything
  python generate_samples.py --rate 175 # words-per-minute for `say`

Requires macOS (`say` and `afconvert` are stock system tools). Chinese needs a
Chinese voice installed (default "Tingting"); English uses "Samantha".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
AUDIO_DIR = DATA_DIR / "audio"


def which(tool: str) -> bool:
    return subprocess.run(["/usr/bin/which", tool], capture_output=True).returncode == 0


def synthesize(text: str, voice: str | None, rate: int, out_wav: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "tts.aiff"
        say_cmd = ["say", "-r", str(rate), "-o", str(aiff)]
        if voice:
            say_cmd += ["-v", voice]
        say_cmd.append(text)
        subprocess.run(say_cmd, check=True)
        # Convert to 16 kHz mono signed 16-bit little-endian WAV.
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(out_wav)],
            check=True,
        )


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize e2e audio samples via macOS say.")
    parser.add_argument("--force", action="store_true", help="re-synthesize even if the wav exists")
    parser.add_argument("--rate", type=int, default=170, help="speech rate (words per minute) for say")
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("Error: this generator relies on macOS `say`/`afconvert`. Provide wavs manually instead.")
        return 1
    for tool in ("say", "afconvert"):
        if not which(tool):
            print(f"Error: required tool '{tool}' not found (expected on macOS).")
            return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    voices: dict[str, str] = manifest.get("voices", {})
    samples = manifest["samples"]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    made, skipped = 0, 0
    for sample in samples:
        out_wav = AUDIO_DIR / f"{sample['id']}.wav"
        if out_wav.exists() and not args.force:
            print(f"  skip  {sample['id']} (exists, {wav_duration_s(out_wav):.2f}s)")
            skipped += 1
            continue
        voice = sample.get("voice") or voices.get(sample["lang"])
        try:
            synthesize(sample["text"], voice, args.rate, out_wav)
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL  {sample['id']}: {exc}. Try a different voice for lang '{sample['lang']}'.")
            return 1
        print(f"  made  {sample['id']} -> {out_wav.name}  ({wav_duration_s(out_wav):.2f}s, voice={voice})")
        made += 1

    print(f"\nDone. {made} synthesized, {skipped} skipped. Audio dir: {AUDIO_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
