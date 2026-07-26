from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from config import (
    LANG_DISPLAY_NAMES,
    MANAGER_BACKEND,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    SINGLETRANS_ROOT,
    STUDENT_EDITOR_ACTION_MODEL_DIR,
    STUDENT_EDITOR_ALWAYS_RUN_EDIT,
    STUDENT_EDITOR_DEVICE,
    STUDENT_EDITOR_DRAFT_BACKEND,
    STUDENT_EDITOR_DRAFT_MODEL,
    STUDENT_EDITOR_EDIT_MODEL_DIR,
    STUDENT_EDITOR_RUN_ACTION,
    STUDENT_EDITOR_RUN_EDIT,
    STUDENT_EDITOR_TORCH_DTYPE,
    SUPPORTED_LANGS,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_TOP_P,
)
from services.manager_types import ContextPack, SourceState, WritePlan
from services.zh_text import to_simplified


PRESERVE_TERMS = ("ai agent",)


class OllamaManagerTranslatorEngine:
    backend_name = "ollama"

    def __init__(self) -> None:
        self.model_name = OLLAMA_MODEL
        self.base_url = OLLAMA_URL
        self.timeout_s = OLLAMA_TIMEOUT_SECONDS
        self.keep_alive = OLLAMA_KEEP_ALIVE
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        # Reuse one client (keep-alive connection pool) instead of reconnecting
        # on every call; matters now that interim segments are also translated.
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120.0),
            )
        return self._client

    async def generate(
        self,
        *,
        source_state: SourceState,
        write_plan: WritePlan,
        context_pack: ContextPack,
        source_lang: str,
        target_lang: str,
        previous_target_text: str = "",
    ) -> tuple[str, dict[str, object], float]:
        started = time.perf_counter()
        text = source_state.raw_text.strip()
        if not text:
            return "", self._meta(write_plan.mode, 0.0), 0.0

        if target_lang not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported target_lang: {target_lang}")

        same_lang = source_lang in SUPPORTED_LANGS and source_lang == target_lang
        mixed_same_lang = same_lang and _is_mixed_language_text(text)
        if same_lang and not mixed_same_lang:
            return text, self._meta(write_plan.mode, 0.0), 0.0

        text_for_prompt, placeholders = _apply_term_placeholders(text, PRESERVE_TERMS)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(source_lang, target_lang)},
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        source_state=source_state,
                        write_plan=write_plan,
                        context_pack=context_pack,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        source_text=text_for_prompt,
                        previous_target_text=previous_target_text,
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": TRANSLATION_TEMPERATURE,
                "top_p": TRANSLATION_TOP_P,
                "num_predict": max(96, TRANSLATION_MAX_NEW_TOKENS),
            },
        }
        client = self._get_client()
        response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        translated = str(data.get("message", {}).get("content", "")).strip()
        if not translated:
            raise RuntimeError("empty translation result from ollama manager engine")
        translated = _restore_term_placeholders(translated, placeholders)
        if target_lang == "zh":
            translated = to_simplified(translated)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return translated, self._meta(write_plan.mode, latency_ms), latency_ms

    def _build_system_prompt(self, source_lang: str, target_lang: str) -> str:
        source_name = LANG_DISPLAY_NAMES.get(source_lang, "Auto-detected language")
        target_name = LANG_DISPLAY_NAMES.get(target_lang, target_lang)
        preserve_terms_hint = ", ".join(PRESERVE_TERMS)
        zh_rule = (
            "Write only Simplified Chinese (简体中文), never Traditional. "
            if target_lang == "zh"
            else ""
        )
        # Kept short on purpose: a long system prompt inflates prefill/latency.
        return (
            f"You are a real-time interpreter. Translate from {source_name} to {target_name}. "
            + zh_rule
            + f"Use natural, fluent {target_name} word order and translate the meaning, not word for word. "
            "Preserve names, numbers, dates, units, and negation. "
            f"Keep these terms exactly: {preserve_terms_hint}. "
            "Output only the translation, nothing else."
        )

    def _build_user_prompt(
        self,
        *,
        source_state: SourceState,
        write_plan: WritePlan,
        context_pack: ContextPack,
        source_lang: str,
        target_lang: str,
        source_text: str,
        previous_target_text: str,
    ) -> str:
        new_segment = source_text.strip() or source_state.live_source_tail.strip() or source_state.raw_text.strip()

        # Translate each segment INDEPENDENTLY. Injecting the previous target as
        # context caused a small model to echo/repeat the previous sentence when
        # the new segment was a short fragment (a common cause of "garbled"
        # output after a few sentences), so it is intentionally omitted.
        return f"Translate this into {LANG_DISPLAY_NAMES.get(target_lang, target_lang)}:\n{new_segment}"

    def _meta(self, mode: str, latency_ms: float) -> dict[str, object]:
        return {
            "engine": "ollama",
            "model": self.model_name,
            "mode": mode,
            "latency_ms": latency_ms,
        }


class _ActionStudentRuntime:
    def __init__(self, model_dir: Path, *, max_source_length: int = 384) -> None:
        self.model_dir = model_dir
        self.max_source_length = max_source_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def predict(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        tokenizer, model, torch, device = self._ensure_loaded()
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_source_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probabilities_tensor = torch.softmax(logits.float(), dim=-1).cpu()
        id_to_label = getattr(model.config, "id2label", None) or {0: "KEEP", 1: "EDIT", 2: "HOLD"}
        probabilities = {
            str(id_to_label[int(idx)]): float(probabilities_tensor[int(idx)].item())
            for idx in range(probabilities_tensor.shape[0])
        }
        best_idx = int(probabilities_tensor.argmax().item())
        return {
            "ok": True,
            "label": str(id_to_label[best_idx]),
            "probabilities": probabilities,
            "latency_s": time.perf_counter() - started,
            "prompt": prompt,
            "device": device,
            "model_dir": str(self.model_dir),
        }

    def _ensure_loaded(self) -> tuple[Any, Any, Any, str]:
        with self._lock:
            if self._tokenizer is not None and self._model is not None and self._torch is not None and self._device is not None:
                return self._tokenizer, self._model, self._torch, self._device
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Action model directory does not exist: {self.model_dir}")

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            self._device = device
            return tokenizer, model, torch, device


class _EditStudentRuntime:
    def __init__(
        self,
        model_dir: Path,
        *,
        max_source_length: int = 384,
        max_target_length: int = 96,
        generation_num_beams: int = 2,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.1,
        length_penalty: float = 1.0,
    ) -> None:
        self.model_dir = model_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.generation_num_beams = generation_num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        tokenizer, model, torch, device = self._ensure_loaded()
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_source_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=self.max_target_length,
                num_beams=self.generation_num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return {
            "ok": True,
            "output_text": decoded,
            "latency_s": time.perf_counter() - started,
            "prompt": prompt,
            "device": device,
            "model_dir": str(self.model_dir),
            "generation": {
                "max_target_length": self.max_target_length,
                "num_beams": self.generation_num_beams,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "repetition_penalty": self.repetition_penalty,
                "length_penalty": self.length_penalty,
            },
        }

    def _ensure_loaded(self) -> tuple[Any, Any, Any, str]:
        with self._lock:
            if self._tokenizer is not None and self._model is not None and self._torch is not None and self._device is not None:
                return self._tokenizer, self._model, self._torch, self._device
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Edit model directory does not exist: {self.model_dir}")

            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=False, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir), local_files_only=True)
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            self._device = device
            return tokenizer, model, torch, device


class _SingleTransRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._loaded = False
        self._load_lock = threading.Lock()
        self._translator_backend_config: Any | None = None
        self._create_translator_engine: Any | None = None
        self._build_editor_input_prompt: Any | None = None
        self._derive_source_markers: Any | None = None
        self._infer_example_family_from_row: Any | None = None

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            if not self.root.exists():
                raise FileNotFoundError(f"singletrans root does not exist: {self.root}")
            root_text = str(self.root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            from src.eval.translator_dataset import (  # type: ignore
                _build_editor_input_prompt,
                _derive_source_markers,
                _infer_example_family_from_row,
            )
            from src.translator import TranslatorBackendConfig, create_translator_engine  # type: ignore

            self._translator_backend_config = TranslatorBackendConfig
            self._create_translator_engine = create_translator_engine
            self._build_editor_input_prompt = _build_editor_input_prompt
            self._derive_source_markers = _derive_source_markers
            self._infer_example_family_from_row = _infer_example_family_from_row
            self._loaded = True

    @property
    def TranslatorBackendConfig(self) -> Any:
        self.ensure_loaded()
        assert self._translator_backend_config is not None
        return self._translator_backend_config

    @property
    def create_translator_engine(self) -> Any:
        self.ensure_loaded()
        assert self._create_translator_engine is not None
        return self._create_translator_engine

    @property
    def build_editor_input_prompt(self) -> Any:
        self.ensure_loaded()
        assert self._build_editor_input_prompt is not None
        return self._build_editor_input_prompt

    @property
    def derive_source_markers(self) -> Any:
        self.ensure_loaded()
        assert self._derive_source_markers is not None
        return self._derive_source_markers

    @property
    def infer_example_family_from_row(self) -> Any:
        self.ensure_loaded()
        assert self._infer_example_family_from_row is not None
        return self._infer_example_family_from_row


class StudentEditorManagerTranslatorEngine:
    backend_name = "student_editor"

    def __init__(self) -> None:
        self.singletrans_root = Path(SINGLETRANS_ROOT)
        self.action_model_dir = Path(STUDENT_EDITOR_ACTION_MODEL_DIR)
        self.edit_model_dir = Path(STUDENT_EDITOR_EDIT_MODEL_DIR)
        self.draft_backend = STUDENT_EDITOR_DRAFT_BACKEND
        self.draft_model = STUDENT_EDITOR_DRAFT_MODEL
        self.device = STUDENT_EDITOR_DEVICE
        self.torch_dtype = STUDENT_EDITOR_TORCH_DTYPE
        self.run_action = STUDENT_EDITOR_RUN_ACTION
        self.run_edit = STUDENT_EDITOR_RUN_EDIT
        self.always_run_edit = STUDENT_EDITOR_ALWAYS_RUN_EDIT
        self.model_name = (
            f"draft={self.draft_model};"
            f"action={self.action_model_dir.parent.name};"
            f"edit={self.edit_model_dir.parent.name}"
        )
        self._runtime = _SingleTransRuntime(self.singletrans_root)
        self._action_runtime = _ActionStudentRuntime(self.action_model_dir)
        self._edit_runtime = _EditStudentRuntime(self.edit_model_dir)
        self._draft_engines: dict[tuple[str, str], Any] = {}
        self._draft_engine_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    async def generate(
        self,
        *,
        source_state: SourceState,
        write_plan: WritePlan,
        context_pack: ContextPack,
        source_lang: str,
        target_lang: str,
        previous_target_text: str = "",
    ) -> tuple[str, dict[str, object], float]:
        return await asyncio.to_thread(
            self._generate_sync,
            source_state,
            write_plan,
            context_pack,
            source_lang,
            target_lang,
        )

    def _generate_sync(
        self,
        source_state: SourceState,
        write_plan: WritePlan,
        context_pack: ContextPack,
        source_lang: str,
        target_lang: str,
    ) -> tuple[str, dict[str, object], float]:
        started = time.perf_counter()
        text = source_state.raw_text.strip()
        if not text:
            return "", self._empty_meta(write_plan.mode), 0.0

        same_lang = source_lang in SUPPORTED_LANGS and source_lang == target_lang
        mixed_same_lang = same_lang and _is_mixed_language_text(text)
        if same_lang and not mixed_same_lang:
            return text, self._same_lang_meta(write_plan.mode), 0.0

        normalized_source_lang = self._normalize_source_lang(source_lang, text)
        if not self._supports_pair(normalized_source_lang, target_lang):
            raise ValueError(
                "student editor backend currently supports English->Chinese only; "
                f"received {source_lang!r}->{target_lang!r}"
            )

        with self._generate_lock:
            draft_started = time.perf_counter()
            draft_result = self._draft_engine(normalized_source_lang, target_lang).generate(
                source_state,
                write_plan,
                context_pack,
            )
            draft_latency_s = time.perf_counter() - draft_started
            draft_text = str(draft_result.draft_text or "").strip()
            if not draft_text:
                raise RuntimeError("student editor draft engine returned empty text")

            row = self._build_prompt_row(
                source_state=source_state,
                write_plan=write_plan,
                context_pack=context_pack,
                draft_text=draft_text,
                source_lang=normalized_source_lang,
                target_lang=target_lang,
            )
            prompt = self._runtime.build_editor_input_prompt(row)

            student_action: dict[str, Any] = {"ok": False, "skip_reason": "disabled"}
            student_edit: dict[str, Any] = {"ok": False, "skip_reason": "disabled"}
            action_label: str | None = None

            if self.run_action:
                student_action = self._action_runtime.predict(prompt)
                action_label = str(student_action.get("label") or "")

            should_run_edit = self.run_edit and (self.always_run_edit or action_label in {None, "", "EDIT"})
            if should_run_edit:
                student_edit = self._edit_runtime.generate(prompt)
            else:
                student_edit = {"ok": False, "skipped": True, "skip_reason": f"action={action_label or '(none)'}"}

            if action_label:
                resolved_action = action_label
            elif student_edit.get("ok"):
                resolved_action = "EDIT_PREVIEW"
            else:
                resolved_action = "KEEP"

            hold_fallback_used = False
            guard_fallback_used = False
            fallback_reason = ""
            similarity = None
            length_ratio = None
            resolved_text = draft_text

            if resolved_action == "HOLD":
                if source_state.asr_is_final:
                    resolved_action = "KEEP_FINAL_FALLBACK"
                    resolved_text = draft_text
                    hold_fallback_used = True
                    fallback_reason = "student_hold_on_final_source"
                else:
                    resolved_text = ""
            elif resolved_action in {"EDIT", "EDIT_PREVIEW"} and student_edit.get("ok"):
                resolved_text = str(student_edit.get("output_text") or "").strip()
                safety_issue, similarity, length_ratio = self._student_edit_safety_issue(
                    source_text=source_state.raw_text,
                    draft_text=draft_text,
                    edited_text=resolved_text,
                )
                if safety_issue:
                    resolved_action = "EDIT_GUARDED_FALLBACK"
                    resolved_text = draft_text
                    guard_fallback_used = True
                    fallback_reason = safety_issue

            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            backend_meta = {
                "engine": "student-editor",
                "model": self.model_name,
                "mode": write_plan.mode,
                "resolved_action": resolved_action,
                "draft_source": self.draft_backend,
                "prompt_family": row.get("example_family"),
                "draft_latency_s": draft_latency_s,
                "draft_backend_meta": dict(draft_result.backend_meta),
                "action_latency_s": student_action.get("latency_s"),
                "action_label": action_label,
                "action_probabilities": dict(student_action.get("probabilities") or {}),
                "edit_latency_s": student_edit.get("latency_s"),
                "edit_output": student_edit.get("output_text"),
                "hold_fallback_used": hold_fallback_used,
                "guard_fallback_used": guard_fallback_used,
                "draft_edit_similarity": similarity,
                "draft_edit_length_ratio": length_ratio,
                "student_edit_safety_issue": fallback_reason if guard_fallback_used else "",
                "fallback_reason": fallback_reason,
                "latency_ms": latency_ms,
            }
            return resolved_text, backend_meta, latency_ms

    def _draft_engine(self, source_lang: str, target_lang: str) -> Any:
        key = (source_lang, target_lang)
        with self._draft_engine_lock:
            existing = self._draft_engines.get(key)
            if existing is not None:
                return existing
            TranslatorBackendConfig = self._runtime.TranslatorBackendConfig
            create_translator_engine = self._runtime.create_translator_engine
            engine = create_translator_engine(
                TranslatorBackendConfig(
                    backend=self.draft_backend,
                    model=self.draft_model,
                    source_language=_singletrans_language_name(source_lang),
                    target_language=_singletrans_language_name(target_lang),
                    device=self.device,
                    torch_dtype=self.torch_dtype,
                )
            )
            self._draft_engines[key] = engine
            return engine

    def _build_prompt_row(
        self,
        *,
        source_state: SourceState,
        write_plan: WritePlan,
        context_pack: ContextPack,
        draft_text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        source_markers = dict(self._runtime.derive_source_markers(source_state.raw_text))
        source_markers["asr_is_final"] = bool(source_state.asr_is_final)
        row = {
            "source_language": _display_language(source_lang),
            "target_language": _display_language(target_lang),
            "source_text": source_state.raw_text,
            "committed_source_text": source_state.committed_source_text,
            "live_source_tail": source_state.live_source_tail or source_state.raw_text,
            "write_mode": write_plan.mode,
            "blocked_spans_json": json.dumps(write_plan.blocked_spans, ensure_ascii=False),
            "glossary_json": json.dumps(context_pack.glossary, ensure_ascii=False, sort_keys=True),
            "entities_json": json.dumps(context_pack.entities, ensure_ascii=False),
            "retrieved_snippets_json": json.dumps(context_pack.retrieved_snippets, ensure_ascii=False),
            "tags_json": "[]",
            "draft_translation": draft_text,
            "extra_json": json.dumps({"source_markers": source_markers}, ensure_ascii=False, sort_keys=True),
        }
        row["example_family"] = self._runtime.infer_example_family_from_row(row)
        return row

    def _student_edit_safety_issue(
        self,
        *,
        source_text: str,
        draft_text: str,
        edited_text: str,
    ) -> tuple[str | None, float | None, float | None]:
        normalized_draft = self._normalize_guard_text(draft_text)
        normalized_edited = self._normalize_guard_text(edited_text)
        if not normalized_edited:
            return "student_edit_empty", None, None

        source_digits = re.findall(r"\d+", source_text)
        if source_digits and not all(digit in edited_text for digit in source_digits):
            return "student_edit_digit_missing", None, None

        if not normalized_draft:
            return None, None, None

        similarity = SequenceMatcher(None, normalized_draft, normalized_edited).ratio()
        length_ratio = len(normalized_edited) / max(1, len(normalized_draft))

        if len(normalized_draft) >= 24 and similarity < 0.18:
            return "student_edit_low_overlap", similarity, length_ratio
        if len(normalized_draft) >= 24 and length_ratio < 0.35:
            return "student_edit_too_short", similarity, length_ratio
        if len(normalized_draft) >= 24 and length_ratio > 1.8:
            return "student_edit_too_long", similarity, length_ratio
        return None, similarity, length_ratio

    @staticmethod
    def _normalize_guard_text(text: str) -> str:
        return re.sub(r"[\W_]+", "", str(text or "").lower())

    @staticmethod
    def _normalize_source_lang(source_lang: str, text: str) -> str:
        if source_lang in {"en", "zh", "fr", "es"}:
            return source_lang
        if re.search(r"[A-Za-z]", text):
            return "en"
        return source_lang

    @staticmethod
    def _supports_pair(source_lang: str, target_lang: str) -> bool:
        return source_lang == "en" and target_lang == "zh"

    def _empty_meta(self, mode: str) -> dict[str, object]:
        return {"engine": "student-editor", "model": self.model_name, "mode": mode, "latency_ms": 0.0}

    def _same_lang_meta(self, mode: str) -> dict[str, object]:
        return {
            "engine": "student-editor",
            "model": self.model_name,
            "mode": mode,
            "resolved_action": "KEEP_SAME_LANGUAGE",
            "latency_ms": 0.0,
        }


def build_manager_engine() -> OllamaManagerTranslatorEngine | StudentEditorManagerTranslatorEngine:
    if MANAGER_BACKEND == "ollama":
        return OllamaManagerTranslatorEngine()
    if MANAGER_BACKEND == "student_editor":
        return StudentEditorManagerTranslatorEngine()
    raise ValueError(f"Unsupported MANAGER_BACKEND: {MANAGER_BACKEND}")


def _display_language(lang: str) -> str:
    return LANG_DISPLAY_NAMES.get(lang, lang)


def _singletrans_language_name(lang: str) -> str:
    if lang == "en":
        return "English"
    if lang == "zh":
        return "Chinese"
    if lang == "fr":
        return "French"
    if lang == "es":
        return "Spanish"
    return _display_language(lang)


def _is_mixed_language_text(text: str) -> bool:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    return has_cjk and has_latin


def _apply_term_placeholders(text: str, terms: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    masked = text
    placeholder_to_term: dict[str, str] = {}
    for idx, term in enumerate(terms):
        placeholder = f"[[TERM_{idx}]]"
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(masked):
            masked = pattern.sub(placeholder, masked)
            placeholder_to_term[placeholder] = term
    return masked, placeholder_to_term


def _restore_term_placeholders(text: str, placeholder_to_term: dict[str, str]) -> str:
    restored = text
    for placeholder, term in placeholder_to_term.items():
        restored = restored.replace(placeholder, term)
    return restored
