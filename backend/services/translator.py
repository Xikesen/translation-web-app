import re

import httpx

from config import (
    LANG_DISPLAY_NAMES,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    SUPPORTED_LANGS,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_TOP_P,
)


async def _chat_completion(system_prompt: str, user_prompt: str, num_predict: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": TRANSLATION_TEMPERATURE,
            "top_p": TRANSLATION_TOP_P,
            "num_predict": num_predict,
        },
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        translated = str(data.get("message", {}).get("content", "")).strip()
        if not translated:
            raise RuntimeError("empty translation result from ollama")
        return translated


PRESERVE_TERMS = ("ai agent",)


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


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return text

    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported target_lang: {target_lang}")

    same_lang = source_lang in SUPPORTED_LANGS and source_lang == target_lang
    mixed_same_lang = same_lang and _is_mixed_language_text(text)
    if same_lang and not mixed_same_lang:
        return text

    source_name = LANG_DISPLAY_NAMES.get(source_lang, "Auto-detected language")
    target_name = LANG_DISPLAY_NAMES[target_lang]
    text_for_prompt, placeholders = _apply_term_placeholders(text, PRESERVE_TERMS)
    preserve_terms_hint = ", ".join(PRESERVE_TERMS)
    system_prompt = (
        "You are a business meeting translation engine.\n"
        "Rules:\n"
        "1) Translate exactly into the target language.\n"
        "2) Output translated text only.\n"
        "3) Do not explain, annotate, or add notes.\n"
        "4) Preserve intent, punctuation, numbers, and units.\n"
        "5) Keep names, product names, and abbreviations unchanged when appropriate.\n"
        f"6) Always preserve these terms exactly when present: {preserve_terms_hint}.\n"
        "7) If source and target languages are the same but the input is mixed-language, rewrite to natural target language while preserving listed terms."
    )
    user_prompt = (
        f"Source language: {source_name}\n"
        f"Target language: {target_name}\n"
        f"Text:\n{text_for_prompt}"
    )
    translated = await _chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_predict=TRANSLATION_MAX_NEW_TOKENS,
    )
    return _restore_term_placeholders(translated, placeholders)


async def translate_via_llm(text: str, source_lang: str, target_lang: str) -> str:
    return await translate_text(text=text, source_lang=source_lang, target_lang=target_lang)


async def polish_translation_candidate(
    source_text: str,
    source_lang: str,
    target_lang: str,
    candidate_translation: str,
) -> str:
    source_name = LANG_DISPLAY_NAMES.get(source_lang, "Auto-detected language")
    target_name = LANG_DISPLAY_NAMES[target_lang]
    system_prompt = (
        "You are a translation post-editor for business meetings.\n"
        "Rules:\n"
        "1) Output target-language text only.\n"
        "2) Use the candidate translation as first draft.\n"
        "3) Fix only mistranslations, grammar, and fluency.\n"
        "4) Keep names, numbers, units, and abbreviations stable."
    )
    user_prompt = (
        f"Source language: {source_name}\n"
        f"Target language: {target_name}\n"
        f"Source text:\n{source_text}\n\n"
        f"Candidate translation:\n{candidate_translation}\n\n"
        "Output only the improved final translation."
    )
    return await _chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        num_predict=max(32, int(TRANSLATION_MAX_NEW_TOKENS * 0.75)),
    )
