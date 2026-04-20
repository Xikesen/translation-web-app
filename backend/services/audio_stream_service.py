import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None

from config import (
    ASR_BATCH_MS,
    ASR_COMMIT_SILENCE_MS,
    ASR_MAX_SEGMENT_MS,
    ASR_MIN_COMMIT_MS,
    MAX_SPEAKERS,
    SPEAKER_LABELS,
)
from state import SessionState


@dataclass
class SegmentCommit:
    speaker_id: str
    segment_bytes: bytes
    duration_ms: int


@dataclass
class TranscribedChunk:
    speaker_id: str
    source_text: str
    source_lang: str
    source_confidence: float


@dataclass
class _SpeakerSegmentState:
    chunks: list[bytes] = field(default_factory=list)
    start_ts: float | None = None
    last_voice_ts: float | None = None


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
        feats = np.array([float(np.mean(b)) for b in bands], dtype=np.float32)
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
        n = np.linalg.norm(updated)
        if n > 0:
            updated = updated / n
        self.prototypes[best_label] = updated
        return best_label


_ASR_MODEL: WhisperModel | None = None


def _get_asr_model() -> WhisperModel:
    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _ASR_MODEL


def _transcribe_sync(segment_bytes: bytes) -> tuple[str, str, float]:
    audio = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return "", "unknown", 0.0
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    model = _get_asr_model()
    segments, info = model.transcribe(
        np.ascontiguousarray(audio, dtype=np.float32),
        language=None,
        beam_size=1,
        best_of=1,
        vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.2,
        compression_ratio_threshold=2.8,
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    lang = getattr(info, "language", "unknown") or "unknown"
    lang_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return text, lang, lang_probability


class RealtimeAudioPipeline:
    def __init__(self) -> None:
        self.sample_rate = 16000
        self.frame_ms = 20
        self.batch_ms = ASR_BATCH_MS
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.frames_per_batch = max(1, int(self.batch_ms / self.frame_ms))
        self.vad = webrtcvad.Vad(1) if webrtcvad is not None else None
        self.voiced_ratio_threshold = 0.2
        self.frame_buffer: list[bytes] = []
        self.tracker = _SpeakerTracker()
        self.commit_silence_ms = ASR_COMMIT_SILENCE_MS
        self.max_segment_ms = ASR_MAX_SEGMENT_MS
        self.min_commit_ms = ASR_MIN_COMMIT_MS
        self.seg_states: dict[str, _SpeakerSegmentState] = {}

    def _split_frames(self, chunk: bytes) -> list[bytes]:
        frames: list[bytes] = []
        for i in range(0, len(chunk), self.frame_bytes):
            f = chunk[i : i + self.frame_bytes]
            if len(f) == self.frame_bytes:
                frames.append(f)
        return frames

    def _is_speech_batch(self, frames: list[bytes]) -> bool:
        if not frames:
            return False
        if self.vad is not None:
            voiced = sum(1 for f in frames if self.vad.is_speech(f, self.sample_rate))
            ratio = voiced / len(frames)
            return ratio >= self.voiced_ratio_threshold

        # Fallback energy-based VAD when webrtcvad is unavailable.
        merged = b"".join(frames)
        sig = np.frombuffer(merged, dtype=np.int16).astype(np.float32) / 32768.0
        if sig.size == 0:
            return False
        rms = float(np.sqrt(np.mean(sig**2)))
        return rms >= 0.01

    def _try_commit(self, speaker_id: str, now_ts: float) -> SegmentCommit | None:
        st = self.seg_states.get(speaker_id)
        if st is None or not st.chunks or st.start_ts is None or st.last_voice_ts is None:
            return None
        duration_ms = int((st.last_voice_ts - st.start_ts) * 1000)
        if duration_ms < self.min_commit_ms:
            return None
        return SegmentCommit(
            speaker_id=speaker_id,
            segment_bytes=b"".join(st.chunks),
            duration_ms=duration_ms,
        )

    def push_chunk(self, chunk_bytes: bytes) -> list[SegmentCommit]:
        commits: list[SegmentCommit] = []
        frames = self._split_frames(chunk_bytes)
        for frame in frames:
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) < self.frames_per_batch:
                continue

            batch = self.frame_buffer
            self.frame_buffer = []
            now_ts = time.time()
            batch_bytes = b"".join(batch)
            is_speech = self._is_speech_batch(batch)

            if is_speech:
                speaker_id = self.tracker.assign(batch_bytes)
                if speaker_id is not None:
                    st = self.seg_states.setdefault(speaker_id, _SpeakerSegmentState())
                    if st.start_ts is None:
                        st.start_ts = now_ts
                    st.last_voice_ts = now_ts
                    st.chunks.append(batch_bytes)
                    if st.start_ts is not None:
                        run_ms = int((now_ts - st.start_ts) * 1000)
                        if run_ms >= self.max_segment_ms:
                            c = self._try_commit(speaker_id, now_ts)
                            if c is not None:
                                commits.append(c)
                            self.seg_states[speaker_id] = _SpeakerSegmentState()

            # Silence-based commit checks for all speakers.
            for sid, st in list(self.seg_states.items()):
                if st.last_voice_ts is None:
                    continue
                silent_ms = int((now_ts - st.last_voice_ts) * 1000)
                if silent_ms < self.commit_silence_ms:
                    continue
                c = self._try_commit(sid, now_ts)
                if c is not None:
                    commits.append(c)
                self.seg_states[sid] = _SpeakerSegmentState()

        return commits


_session_pipelines: dict[str, RealtimeAudioPipeline] = {}


def get_pipeline(session_id: str) -> RealtimeAudioPipeline:
    p = _session_pipelines.get(session_id)
    if p is None:
        p = RealtimeAudioPipeline()
        _session_pipelines[session_id] = p
    return p


def remove_pipeline(session_id: str) -> None:
    _session_pipelines.pop(session_id, None)


async def process_audio_chunk(session: SessionState, chunk_bytes: bytes) -> list[TranscribedChunk]:
    pipeline = get_pipeline(session.session_id)
    commits = pipeline.push_chunk(chunk_bytes)
    transcribed: list[TranscribedChunk] = []
    for commit in commits:
        source_text, source_lang, source_confidence = await asyncio.to_thread(
            _transcribe_sync, commit.segment_bytes
        )
        if not source_text.strip():
            continue
        transcribed.append(
            TranscribedChunk(
                speaker_id=commit.speaker_id,
                source_text=source_text.strip(),
                source_lang=source_lang.lower(),
                source_confidence=source_confidence,
            )
        )
    return transcribed
