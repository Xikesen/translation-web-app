import httpx

from config import (
    LANG_DISPLAY_NAMES,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    SUPPORTED_LANGS,
)


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text.strip():
        return text

    if source_lang in SUPPORTED_LANGS and source_lang == target_lang:
        return text

    source_name = LANG_DISPLAY_NAMES.get(source_lang, "Auto-detected language")
    target_name = LANG_DISPLAY_NAMES[target_lang]
    system_prompt = (
        "You are a translation engine.\n"
        "Rules:\n"
        "1) Translate exactly into the target language.\n"
        "2) Output translated text only.\n"
        "3) Do not explain or add notes.\n"
        "4) Preserve intent and punctuation."
    )
    user_prompt = (
        f"Source language: {source_name}\n"
        f"Target language: {target_name}\n"
        f"Text:\n{text}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 128},
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        translated = str(data.get("message", {}).get("content", "")).strip()
        if not translated:
            raise RuntimeError("empty translation result from ollama")
        return translated
