import asyncio
import math
from array import array
from dataclasses import dataclass, field
import threading
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None

from config import (
    AUDIO_ENERGY_THRESHOLD,
    AUDIO_VOICED_RATIO_THRESHOLD,
    ASR_BACKEND,
    ASR_BATCH_MS,
    ASR_COMMIT_SILENCE_MS,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_LANGUAGE_HINT,
    ASR_MAX_SEGMENT_MS,
    ASR_MIN_COMMIT_MS,
    ASR_MODEL_NAME,
    MAX_SPEAKERS,
    SPEAKER_LABELS,
)
from state import SessionState


@dataclass
class TranscribedChunk:
    speaker_id: str
    source_text: str
    source_lang: str
    source_confidence: float
    is_final: bool = True
    utterance_id: str | None = None
    stability: float | None = None


@dataclass
class _ASRAudioFrame:
    ts_start_ms: int
    ts_end_ms: int
    pcm16: bytes
    sample_rate_hz: int = 16_000
    channels: int = 1
    vad_prob: float = 0.0
    energy: float = 0.0


@dataclass
class _ASRPartialResult:
    text: str
    token_conf: list[float]
    is_final: bool
    utterance_id: str | None
    stability: float | None
    emitted_ts_ms: int | None
    source_lang: str
    source_confidence: float


@dataclass
class _SpeakerStreamingState:
    asr: Any


class _SpeakerTracker:
    def __init__(self) -> None:
        self.prototypes: dict[str, np.ndarray] = {}
        self.match_threshold = 0.82
        self.momentum = 0.85

    def _embed(self, pcm_bytes: bytes) -> np.ndarray:
        signal = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if signal.size == 0:
            return np.zeros(8, dtype=np.float32)
        spectrum = np.abs(np.fft.rfft(signal))
        if spectrum.size < 16:
            spectrum = np.pad(spectrum, (0, 16 - spectrum.size))
        bands = np.array_split(spectrum, 8)
        feats = np.array([float(np.mean(band)) for band in bands], dtype=np.float32)
        feats = np.log1p(feats)
        norm = np.linalg.norm(feats)
        if norm > 0:
            feats = feats / norm
        return feats

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def assign(self, pcm_bytes: bytes) -> str | None:
        emb = self._embed(pcm_bytes)
        if not self.prototypes:
            label = SPEAKER_LABELS[0]
            self.prototypes[label] = emb
            return label

        best_label = None
        best_score = -1.0
        for label, proto in self.prototypes.items():
            score = self._cosine(emb, proto)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        if best_score < self.match_threshold and len(self.prototypes) < MAX_SPEAKERS:
            label = SPEAKER_LABELS[len(self.prototypes)]
            self.prototypes[label] = emb
            return label

        proto = self.prototypes[best_label]
        updated = self.momentum * proto + (1.0 - self.momentum) * emb
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        self.prototypes[best_label] = updated
        return best_label


_ASR_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}
_TRANSFORMERS_PIPELINE_CACHE: dict[tuple[str, str, str], tuple[Any, threading.Lock]] = {}


def _get_asr_model(*, model_name: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_name, device, compute_type)
    model = _ASR_MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _ASR_MODEL_CACHE[key] = model
    return model


class _FasterWhisperStreamingASR:
    backend_name = "faster_whisper_local"

    def __init__(
        self,
        *,
        model_name: str,
        language_hint: str,
        device: str,
        compute_type: str,
        decode_interval_ms: int,
        endpoint_silence_ms: int,
        min_commit_ms: int,
        max_segment_ms: int,
    ) -> None:
        self.model_name = model_name
        self.language_hint = language_hint
        self.device = device
        self.compute_type = compute_type
        self.decode_interval_ms = decode_interval_ms
        self.endpoint_silence_ms = endpoint_silence_ms
        self.min_commit_ms = min_commit_ms
        self.max_segment_ms = max_segment_ms

        self._utterance_index = 0
        self._active_utterance_id = self._new_utterance_id()
        self._buffer = bytearray()
        self._buffered_audio_ms = 0
        self._buffered_since_decode_ms = 0
        self._trailing_silence_ms = 0
        self._utterance_start_ms: int | None = None
        self._last_frame_end_ms: int | None = None
        self._last_partial_signature: tuple[str, bool] | None = None
        self._previous_tokens: list[str] = []

    def has_pending_audio(self) -> bool:
        return bool(self._buffer)

    def push_frame(self, frame: _ASRAudioFrame) -> list[_ASRPartialResult]:
        if self._utterance_start_ms is None:
            self._utterance_start_ms = frame.ts_start_ms
        self._last_frame_end_ms = frame.ts_end_ms
        frame_duration_ms = max(1, frame.ts_end_ms - frame.ts_start_ms)
        self._buffer.extend(frame.pcm16)
        self._buffered_audio_ms += frame_duration_ms
        self._buffered_since_decode_ms += frame_duration_ms
        self._update_trailing_silence(frame=frame, frame_duration_ms=frame_duration_ms)

        should_finalize = (
            self._buffered_audio_ms >= self.min_commit_ms
            and self._trailing_silence_ms >= self.endpoint_silence_ms
        ) or self._buffered_audio_ms >= self.max_segment_ms
        should_decode = should_finalize or self._buffered_since_decode_ms >= self.decode_interval_ms
        if not should_decode:
            return []

        decode_reason = "endpoint_silence" if should_finalize else "decode_interval"
        partial = self._decode_current_buffer(
            default_emitted_ts_ms=frame.ts_end_ms,
            is_final=should_finalize,
            decode_reason=decode_reason,
        )
        self._buffered_since_decode_ms = 0
        if partial is None:
            return []

        if should_finalize:
            self._reset_active_utterance()
        return [partial]

    def flush(self) -> list[_ASRPartialResult]:
        if not self._buffer:
            return []
        partial = self._decode_current_buffer(
            default_emitted_ts_ms=self._last_frame_end_ms,
            is_final=True,
            decode_reason="flush",
        )
        self._reset_active_utterance()
        return [partial] if partial is not None else []

    def _decode_current_buffer(
        self,
        *,
        default_emitted_ts_ms: int | None,
        is_final: bool,
        decode_reason: str,
    ) -> _ASRPartialResult | None:
        if not self._buffer:
            return None

        model = _get_asr_model(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        audio = _pcm16_to_audio_input(bytes(self._buffer))
        segments, info = model.transcribe(
            audio,
            language=self.language_hint or None,
            beam_size=1,
            best_of=1,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.2,
            compression_ratio_threshold=2.8,
        )
        tokens, _token_times_ms, token_conf = _segments_to_token_payload(
            segments,
            ts_offset_ms=self._utterance_start_ms or 0,
        )
        text = " ".join(tokens).strip()
        if not text:
            return None

        signature = (text, is_final)
        if signature == self._last_partial_signature:
            return None
        self._last_partial_signature = signature

        stability = 1.0 if is_final else _stable_prefix_ratio(self._previous_tokens, tokens)
        self._previous_tokens = tokens.copy()
        source_lang = str(getattr(info, "language", "") or "unknown").lower()
        source_confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        emitted_ts_ms = default_emitted_ts_ms
        return _ASRPartialResult(
            text=text,
            token_conf=token_conf,
            is_final=is_final,
            utterance_id=self._active_utterance_id,
            stability=stability,
            emitted_ts_ms=emitted_ts_ms,
            source_lang=source_lang,
            source_confidence=source_confidence,
        )

    def _update_trailing_silence(self, *, frame: _ASRAudioFrame, frame_duration_ms: int) -> None:
        if frame.vad_prob <= AUDIO_VOICED_RATIO_THRESHOLD or frame.energy <= AUDIO_ENERGY_THRESHOLD:
            self._trailing_silence_ms += frame_duration_ms
            return
        self._trailing_silence_ms = 0

    def _reset_active_utterance(self) -> None:
        self._utterance_index += 1
        self._active_utterance_id = self._new_utterance_id()
        self._buffer = bytearray()
        self._buffered_audio_ms = 0
        self._buffered_since_decode_ms = 0
        self._trailing_silence_ms = 0
        self._utterance_start_ms = None
        self._last_frame_end_ms = None
        self._last_partial_signature = None
        self._previous_tokens = []

    def _new_utterance_id(self) -> str:
        return f"fw-local:utt-{self._utterance_index}"


class _TransformersWhisperStreamingASR:
    backend_name = "transformers_whisper_local"

    def __init__(
        self,
        *,
        model_name: str,
        language_hint: str,
        device: str,
        compute_type: str,
        decode_interval_ms: int,
        endpoint_silence_ms: int,
    ) -> None:
        self.model_name = model_name
        self.language_hint = language_hint
        self.device = device
        self.compute_type = compute_type
        self.decode_interval_ms = decode_interval_ms
        self.endpoint_silence_ms = endpoint_silence_ms

        self._pipeline_runner: Any | None = None
        self._pipeline_lock: threading.Lock | None = None
        self._utterance_index = 0
        self._active_utterance_id = self._new_utterance_id()
        self._sample_rate_hz: int | None = None
        self._buffer = bytearray()
        self._buffered_audio_ms = 0
        self._buffered_since_decode_ms = 0
        self._trailing_silence_ms = 0
        self._utterance_start_ms: int | None = None
        self._last_frame_end_ms: int | None = None
        self._last_partial_signature: tuple[str, bool] | None = None
        self._previous_tokens: list[str] = []
        self._speech_observed_in_utterance = False

    def has_pending_audio(self) -> bool:
        return bool(self._buffer)

    def push_frame(self, frame: _ASRAudioFrame) -> list[_ASRPartialResult]:
        if self._sample_rate_hz is None:
            self._sample_rate_hz = frame.sample_rate_hz
        elif frame.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("TransformersWhisperStreamingASR requires a stable input sample rate within one session")

        if self._utterance_start_ms is None:
            self._utterance_start_ms = frame.ts_start_ms
        self._last_frame_end_ms = frame.ts_end_ms
        frame_duration_ms = max(1, frame.ts_end_ms - frame.ts_start_ms)
        self._buffer.extend(frame.pcm16)
        self._buffered_audio_ms += frame_duration_ms
        self._buffered_since_decode_ms += frame_duration_ms
        if _is_speech_frame(frame):
            self._speech_observed_in_utterance = True
        self._update_trailing_silence(frame=frame, frame_duration_ms=frame_duration_ms)

        finalize_candidate = (
            self._speech_observed_in_utterance
            and self._buffered_audio_ms >= max(1600, self.decode_interval_ms * 2)
            and self._trailing_silence_ms >= self.endpoint_silence_ms
        )
        should_decode = self._speech_observed_in_utterance and (
            finalize_candidate or self._buffered_since_decode_ms >= self.decode_interval_ms
        )
        if not should_decode:
            return []

        partial, did_finalize = self._decode_current_buffer(
            default_emitted_ts_ms=frame.ts_end_ms,
            finalize_candidate=finalize_candidate,
        )
        self._buffered_since_decode_ms = 0
        if partial is None:
            return []

        if did_finalize:
            self._reset_active_utterance()
        return [partial]

    def flush(self) -> list[_ASRPartialResult]:
        if not self._buffer or not self._speech_observed_in_utterance:
            return []
        partial, _did_finalize = self._decode_current_buffer(
            default_emitted_ts_ms=self._last_frame_end_ms,
            finalize_candidate=True,
            force_final=True,
        )
        self._reset_active_utterance()
        return [partial] if partial is not None else []

    def _decode_current_buffer(
        self,
        *,
        default_emitted_ts_ms: int | None,
        finalize_candidate: bool,
        force_final: bool = False,
    ) -> tuple[_ASRPartialResult | None, bool]:
        if not self._buffer or self._sample_rate_hz is None:
            return None, False

        pipeline_runner, pipeline_lock = self._ensure_pipeline()
        audio = _pcm16_bytes_to_float32(bytes(self._buffer))
        audio = _resample_if_needed(audio, source_rate_hz=self._sample_rate_hz, target_rate_hz=16_000)
        with pipeline_lock:
            result = pipeline_runner(
                {"array": audio, "sampling_rate": 16_000},
                return_timestamps="word",
                generate_kwargs={
                    "language": _normalize_whisper_language(self.language_hint),
                    "task": "transcribe",
                },
            )

        tokens, token_times_ms, token_conf = _pipeline_result_to_token_payload(
            result,
            ts_offset_ms=self._utterance_start_ms or 0,
            default_emitted_ts_ms=default_emitted_ts_ms,
            is_final=force_final,
        )
        if not force_final:
            tokens = _strip_terminal_sentence_punctuation(tokens)
            token_conf = [0.85] * len(tokens)
        text = " ".join(tokens).strip()
        if not text:
            return None, False

        is_final = force_final or (
            finalize_candidate
            and _can_finalize_decoded_text(
                tokens=tokens,
                trailing_silence_ms=self._trailing_silence_ms,
                endpoint_silence_ms=self.endpoint_silence_ms,
            )
        )
        signature = (text, is_final)
        if signature == self._last_partial_signature:
            return None, False
        self._last_partial_signature = signature

        emitted_ts_ms = token_times_ms[-1][1] if token_times_ms else default_emitted_ts_ms
        stability = 1.0 if is_final else _stable_prefix_ratio(self._previous_tokens, tokens)
        self._previous_tokens = tokens.copy()
        source_lang = _normalize_source_lang_hint(self.language_hint)
        source_confidence = 1.0 if is_final else 0.85

        return (
            _ASRPartialResult(
                text=text,
                token_conf=token_conf,
                is_final=is_final,
                utterance_id=self._active_utterance_id,
                stability=stability,
                emitted_ts_ms=emitted_ts_ms,
                source_lang=source_lang,
                source_confidence=source_confidence,
            ),
            is_final,
        )

    def _ensure_pipeline(self) -> tuple[Any, threading.Lock]:
        if self._pipeline_runner is None or self._pipeline_lock is None:
            self._pipeline_runner, self._pipeline_lock = _get_transformers_pipeline(
                model_name=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._pipeline_runner, self._pipeline_lock

    def _update_trailing_silence(self, *, frame: _ASRAudioFrame, frame_duration_ms: int) -> None:
        if not self._speech_observed_in_utterance:
            self._trailing_silence_ms = 0
            return
        if _is_silence_frame(frame):
            self._trailing_silence_ms += frame_duration_ms
            return
        self._trailing_silence_ms = 0

    def _reset_active_utterance(self) -> None:
        self._utterance_index += 1
        self._active_utterance_id = self._new_utterance_id()
        self._buffer = bytearray()
        self._buffered_audio_ms = 0
        self._buffered_since_decode_ms = 0
        self._trailing_silence_ms = 0
        self._utterance_start_ms = None
        self._last_frame_end_ms = None
        self._last_partial_signature = None
        self._previous_tokens = []
        self._speech_observed_in_utterance = False

    def _new_utterance_id(self) -> str:
        return f"tf-whisper-local:utt-{self._utterance_index}"


class RealtimeAudioPipeline:
    def __init__(self) -> None:
        self.sample_rate = 16000
        self.frame_ms = 20
        self.batch_ms = ASR_BATCH_MS
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.frames_per_batch = max(1, int(self.batch_ms / self.frame_ms))
        self.vad = webrtcvad.Vad(1) if webrtcvad is not None else None
        self.voiced_ratio_threshold = AUDIO_VOICED_RATIO_THRESHOLD
        self.energy_threshold = AUDIO_ENERGY_THRESHOLD
        self.frame_buffer: list[bytes] = []
        self.tracker = _SpeakerTracker()
        self.stream_cursor_ms = 0
        self.speaker_streams: dict[str, _SpeakerStreamingState] = {}

    def _split_frames(self, chunk: bytes) -> list[bytes]:
        frames: list[bytes] = []
        for idx in range(0, len(chunk), self.frame_bytes):
            frame = chunk[idx : idx + self.frame_bytes]
            if len(frame) == self.frame_bytes:
                frames.append(frame)
        return frames

    def _batch_vad_prob(self, frames: list[bytes]) -> float:
        if not frames:
            return 0.0
        if self.vad is None:
            return 1.0 if self._batch_energy(frames) >= self.energy_threshold else 0.0
        voiced = sum(1 for frame in frames if self.vad.is_speech(frame, self.sample_rate))
        return voiced / len(frames)

    def _batch_energy(self, frames: list[bytes]) -> float:
        merged = b"".join(frames)
        signal = np.frombuffer(merged, dtype=np.int16).astype(np.float32) / 32768.0
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal**2)))

    def _get_stream(self, speaker_id: str) -> _SpeakerStreamingState:
        stream = self.speaker_streams.get(speaker_id)
        if stream is None:
            if ASR_BACKEND == "faster_whisper_local":
                asr = _FasterWhisperStreamingASR(
                    model_name=ASR_MODEL_NAME,
                    language_hint=ASR_LANGUAGE_HINT,
                    device=ASR_DEVICE,
                    compute_type=ASR_COMPUTE_TYPE,
                    decode_interval_ms=ASR_BATCH_MS,
                    endpoint_silence_ms=ASR_COMMIT_SILENCE_MS,
                    min_commit_ms=ASR_MIN_COMMIT_MS,
                    max_segment_ms=ASR_MAX_SEGMENT_MS,
                )
            elif ASR_BACKEND == "transformers_whisper_local":
                asr = _TransformersWhisperStreamingASR(
                    model_name=ASR_MODEL_NAME,
                    language_hint=ASR_LANGUAGE_HINT,
                    device=ASR_DEVICE,
                    compute_type=ASR_COMPUTE_TYPE,
                    decode_interval_ms=ASR_BATCH_MS,
                    endpoint_silence_ms=ASR_COMMIT_SILENCE_MS,
                )
            else:
                raise ValueError(f"unsupported ASR_BACKEND: {ASR_BACKEND}")
            stream = _SpeakerStreamingState(asr=asr)
            self.speaker_streams[speaker_id] = stream
        return stream

    def push_chunk(self, chunk_bytes: bytes) -> list[TranscribedChunk]:
        transcribed: list[TranscribedChunk] = []
        frames = self._split_frames(chunk_bytes)
        for frame in frames:
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) < self.frames_per_batch:
                continue

            batch_frames = self.frame_buffer
            self.frame_buffer = []
            batch_bytes = b"".join(batch_frames)
            batch_duration_ms = len(batch_frames) * self.frame_ms
            frame_start_ms = self.stream_cursor_ms
            frame_end_ms = frame_start_ms + batch_duration_ms
            self.stream_cursor_ms = frame_end_ms

            vad_prob = self._batch_vad_prob(batch_frames)
            energy = self._batch_energy(batch_frames)
            is_speech = vad_prob >= self.voiced_ratio_threshold or energy >= self.energy_threshold
            silence_bytes = b"\x00" * len(batch_bytes)

            if is_speech:
                speaker_id = self.tracker.assign(batch_bytes)
                if speaker_id is not None:
                    transcribed.extend(
                        self._push_frame_to_speaker(
                            speaker_id=speaker_id,
                            pcm_bytes=batch_bytes,
                            ts_start_ms=frame_start_ms,
                            ts_end_ms=frame_end_ms,
                            vad_prob=vad_prob,
                            energy=energy,
                        )
                    )
                    for other_speaker_id, state in self.speaker_streams.items():
                        if other_speaker_id == speaker_id or not state.asr.has_pending_audio():
                            continue
                        transcribed.extend(
                            self._push_frame_to_speaker(
                                speaker_id=other_speaker_id,
                                pcm_bytes=silence_bytes,
                                ts_start_ms=frame_start_ms,
                                ts_end_ms=frame_end_ms,
                                vad_prob=0.0,
                                energy=0.0,
                            )
                        )
                continue

            for speaker_id, state in self.speaker_streams.items():
                if not state.asr.has_pending_audio():
                    continue
                transcribed.extend(
                    self._push_frame_to_speaker(
                        speaker_id=speaker_id,
                        pcm_bytes=silence_bytes,
                        ts_start_ms=frame_start_ms,
                        ts_end_ms=frame_end_ms,
                        vad_prob=0.0,
                        energy=0.0,
                    )
                )

        return transcribed

    def flush(self) -> list[TranscribedChunk]:
        transcribed: list[TranscribedChunk] = []
        for speaker_id, state in self.speaker_streams.items():
            for partial in state.asr.flush():
                chunk = _final_partial_to_chunk(speaker_id, partial)
                if chunk is not None:
                    transcribed.append(chunk)
        return transcribed

    def _push_frame_to_speaker(
        self,
        *,
        speaker_id: str,
        pcm_bytes: bytes,
        ts_start_ms: int,
        ts_end_ms: int,
        vad_prob: float,
        energy: float,
    ) -> list[TranscribedChunk]:
        stream = self._get_stream(speaker_id)
        frame = _ASRAudioFrame(
            ts_start_ms=ts_start_ms,
            ts_end_ms=ts_end_ms,
            pcm16=pcm_bytes,
            vad_prob=vad_prob,
            energy=energy,
        )
        chunks: list[TranscribedChunk] = []
        for partial in stream.asr.push_frame(frame):
            chunk = _final_partial_to_chunk(speaker_id, partial)
            if chunk is not None:
                chunks.append(chunk)
        return chunks


def _final_partial_to_chunk(
    speaker_id: str,
    partial: _ASRPartialResult,
) -> TranscribedChunk | None:
    if not partial.is_final:
        return None
    source_text = partial.text.strip()
    if not source_text:
        return None
    confidence = sum(partial.token_conf) / len(partial.token_conf) if partial.token_conf else partial.source_confidence
    return TranscribedChunk(
        speaker_id=speaker_id,
        source_text=source_text,
        source_lang=partial.source_lang,
        source_confidence=max(0.0, min(1.0, confidence)),
        is_final=partial.is_final,
        utterance_id=partial.utterance_id,
        stability=partial.stability,
    )


def _pcm16_to_audio_input(pcm16: bytes) -> Any:
    samples = array("h")
    samples.frombytes(pcm16)
    if not samples:
        return []

    normalized = [max(-1.0, min(1.0, sample / 32768.0)) for sample in samples]
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return normalized
    return np.asarray(normalized, dtype=np.float32)


def _get_transformers_pipeline(*, model_name: str, device: str, compute_type: str) -> tuple[Any, threading.Lock]:
    key = (model_name, device, compute_type)
    cached = _TRANSFORMERS_PIPELINE_CACHE.get(key)
    if cached is None:
        cached = (_load_transformers_whisper_pipeline(model_name=model_name, device=device, compute_type=compute_type), threading.Lock())
        _TRANSFORMERS_PIPELINE_CACHE[key] = cached
    return cached


def _load_transformers_whisper_pipeline(*, model_name: str, device: str, compute_type: str) -> Any:
    import torch  # type: ignore
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # type: ignore

    torch_dtype = _torch_dtype_for_compute_type(torch, compute_type=compute_type, device=device)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype)
    if device != "cpu":
        model.to(device)
    pipeline_device = 0 if device.startswith("cuda") else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipeline_device,
    )


def _torch_dtype_for_compute_type(torch_module: Any, *, compute_type: str, device: str) -> Any:
    if device.startswith("cuda") and compute_type in {"float16", "int8_float16"}:
        return torch_module.float16
    return torch_module.float32


def _pcm16_bytes_to_float32(pcm16: bytes) -> np.ndarray:
    samples = array("h")
    samples.frombytes(pcm16)
    if not samples:
        return np.asarray([], dtype=np.float32)
    return np.asarray(samples, dtype=np.float32) / 32768.0


def _resample_if_needed(audio: np.ndarray, *, source_rate_hz: int, target_rate_hz: int) -> np.ndarray:
    if source_rate_hz == target_rate_hz:
        return audio
    if len(audio) == 0:
        return audio
    source_points = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_len = max(1, int(round(len(audio) * target_rate_hz / source_rate_hz)))
    target_points = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_points, source_points, audio).astype(np.float32)


def _pipeline_result_to_token_payload(
    result: dict[str, Any],
    *,
    ts_offset_ms: int,
    default_emitted_ts_ms: int | None,
    is_final: bool,
) -> tuple[list[str], list[tuple[int, int]], list[float]]:
    chunks = result.get("chunks") or []
    tokens: list[str] = []
    token_times_ms: list[tuple[int, int]] = []
    token_conf: list[float] = []
    if chunks:
        for chunk in chunks:
            token = str(chunk.get("text", "") or "").strip()
            if not token:
                continue
            raw_ts = chunk.get("timestamp") or (None, None)
            start_s = raw_ts[0] if len(raw_ts) > 0 else None
            end_s = raw_ts[1] if len(raw_ts) > 1 else None
            start_ms = ts_offset_ms + int(round(float(start_s or 0.0) * 1000))
            end_ms = ts_offset_ms + int(round(float(end_s or start_s or 0.0) * 1000))
            if end_ms < start_ms:
                end_ms = start_ms
            tokens.append(token)
            token_times_ms.append((start_ms, end_ms))
            token_conf.append(1.0 if is_final else 0.85)
        if tokens:
            return tokens, token_times_ms, token_conf

    text = str(result.get("text", "") or "").strip()
    if not text:
        return [], [], []
    tokens = text.split()
    if not tokens:
        return [], [], []

    end_ms = default_emitted_ts_ms if default_emitted_ts_ms is not None else ts_offset_ms
    duration_per_token_ms = max(1, int(math.ceil((end_ms - ts_offset_ms) / len(tokens)))) if tokens else 1
    for index, token in enumerate(tokens):
        token_start = ts_offset_ms + index * duration_per_token_ms
        token_end = min(end_ms, token_start + duration_per_token_ms) if end_ms >= token_start else token_start
        token_times_ms.append((token_start, token_end))
        token_conf.append(1.0 if is_final else 0.8)
    return tokens, token_times_ms, token_conf


def _segments_to_token_payload(
    segments: Any,
    *,
    ts_offset_ms: int = 0,
) -> tuple[list[str], list[tuple[int, int]], list[float]]:
    tokens: list[str] = []
    token_times_ms: list[tuple[int, int]] = []
    token_conf: list[float] = []
    for segment in list(segments):
        words = getattr(segment, "words", None) or []
        if words:
            for word in words:
                token = str(getattr(word, "word", "") or "").strip()
                if not token:
                    continue
                start_ms = ts_offset_ms + int(round(float(getattr(word, "start", 0.0)) * 1000))
                end_ms = ts_offset_ms + int(round(float(getattr(word, "end", 0.0)) * 1000))
                probability = float(getattr(word, "probability", 0.0) or 0.0)
                tokens.append(token)
                token_times_ms.append((start_ms, end_ms))
                token_conf.append(max(0.0, min(1.0, probability)))
            continue

        segment_text = str(getattr(segment, "text", "") or "").strip()
        if not segment_text:
            continue
        segment_tokens = segment_text.split()
        if not segment_tokens:
            continue
        start_ms = ts_offset_ms + int(round(float(getattr(segment, "start", 0.0)) * 1000))
        end_ms = ts_offset_ms + int(round(float(getattr(segment, "end", 0.0)) * 1000))
        avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
        confidence = _avg_logprob_to_confidence(avg_logprob)
        duration_per_token_ms = max(1, int(math.ceil((end_ms - start_ms) / max(len(segment_tokens), 1))))
        for idx, token in enumerate(segment_tokens):
            token_start = start_ms + idx * duration_per_token_ms
            token_end = min(end_ms, token_start + duration_per_token_ms)
            tokens.append(token)
            token_times_ms.append((token_start, token_end))
            token_conf.append(confidence)
    return tokens, token_times_ms, token_conf


def _stable_prefix_ratio(previous_tokens: list[str], current_tokens: list[str]) -> float:
    if not current_tokens:
        return 0.0
    prefix_len = 0
    for previous_token, current_token in zip(previous_tokens, current_tokens, strict=False):
        if previous_token != current_token:
            break
        prefix_len += 1
    return min(1.0, prefix_len / len(current_tokens))


def _avg_logprob_to_confidence(avg_logprob: float) -> float:
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def _normalize_whisper_language(language: str | None) -> str:
    if language is None:
        return "english"
    normalized = language.strip().lower()
    aliases = {
        "en": "english",
        "en-us": "english",
        "en_us": "english",
        "english": "english",
    }
    return aliases.get(normalized, normalized)


def _normalize_source_lang_hint(language: str | None) -> str:
    if language is None:
        return "en"
    normalized = language.strip().lower()
    aliases = {
        "en": "en",
        "en-us": "en",
        "en_us": "en",
        "english": "en",
        "zh": "zh",
        "chinese": "zh",
        "french": "fr",
        "fr": "fr",
        "spanish": "es",
        "es": "es",
    }
    return aliases.get(normalized, normalized or "en")


def _is_speech_frame(frame: _ASRAudioFrame) -> bool:
    return frame.vad_prob > AUDIO_VOICED_RATIO_THRESHOLD or frame.energy > AUDIO_ENERGY_THRESHOLD


def _is_silence_frame(frame: _ASRAudioFrame) -> bool:
    return frame.vad_prob <= AUDIO_VOICED_RATIO_THRESHOLD and frame.energy <= AUDIO_ENERGY_THRESHOLD


def _strip_terminal_sentence_punctuation(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    stripped_tokens = tokens.copy()
    stripped_tokens[-1] = stripped_tokens[-1].rstrip(".!?")
    return [token for token in stripped_tokens if token]


def _can_finalize_decoded_text(
    *,
    tokens: list[str],
    trailing_silence_ms: int,
    endpoint_silence_ms: int,
) -> bool:
    if not tokens:
        return False
    cleaned = [token.strip(",.?!").lower() for token in tokens if token.strip(",.?!")]
    if not cleaned:
        return False
    if tokens[-1].endswith((".", "!", "?")):
        return True
    if len(cleaned) < 4:
        return False
    if cleaned[-1] in _HANGING_TAIL_TOKENS:
        return False
    return trailing_silence_ms >= endpoint_silence_ms + 250


_HANGING_TAIL_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


_session_pipelines: dict[str, RealtimeAudioPipeline] = {}


def get_pipeline(session_id: str) -> RealtimeAudioPipeline:
    pipeline = _session_pipelines.get(session_id)
    if pipeline is None:
        pipeline = RealtimeAudioPipeline()
        _session_pipelines[session_id] = pipeline
    return pipeline


def remove_pipeline(session_id: str) -> None:
    _session_pipelines.pop(session_id, None)


async def process_audio_chunk(session: SessionState, chunk_bytes: bytes) -> list[TranscribedChunk]:
    pipeline = get_pipeline(session.session_id)
    return await asyncio.to_thread(pipeline.push_chunk, chunk_bytes)
