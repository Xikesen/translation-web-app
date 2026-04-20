# Realtime Translation MVP (Current Runnable)

当前版本是最小可运行闭环：

- `WebRTC audio -> VAD -> faster-whisper ASR -> translation router -> Ollama (Gemma 4-E4B Q8_0)`
- 前端双窗口：`Transcript`（原文） + `Translation`（译文）
- 支持语种：`en/fr/zh/es`

## 1) Prerequisites

- macOS
- Python 3.11+
- [Ollama](https://ollama.com/)

## 2) Start Ollama + Model

```bash
brew install ollama
ollama serve
ollama pull gemma-4-e4b:q8_0
```

Quick check:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gemma-4-e4b:q8_0",
    "messages":[
      {"role":"system","content":"You are a translation engine. Output translated text only."},
      {"role":"user","content":"Translate to French: Hello everyone"}
    ],
    "stream":false
  }'
```

## 3) Start Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core envs (minimal):

```bash
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=gemma-4-e4b:q8_0
export ASR_BATCH_MS=320
export ASR_COMMIT_SILENCE_MS=650
export ASR_MAX_SEGMENT_MS=2200
export MERGE_SHORT_MAX_CHARS=24
export MERGE_MAX_WAIT_SECONDS=2.3
export MERGE_FORCE_CHARS=56
export TRANSLATION_ROUTER_ENABLE_EXACT_CACHE=1
export TRANSLATION_ROUTER_ENABLE_GLOSSARY=1
export TRANSLATION_ROUTER_ENABLE_FUZZY_TM=1
export TRANSLATION_ROUTER_ENABLE_TM_POLISH=1
export TRANSLATION_EVENT_INCLUDE_ROUTE_METRICS=1
```

Run server:

```bash
uvicorn main:app --reload --port 8000
```

## 4) Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/benchmark/translation
```

Open:

- `http://localhost:8000/webrtc`

Click `Start` and speak. `translation` events include:

- `route` (e.g. `glossary`, `tm_fuzzy(...)`, `tm_polish(...)`, `llm`)
- `latency_ms`

## 5) Key APIs

- `POST /session/start`
- `POST /session/{session_id}/stop`
- `POST /session/{session_id}/utterance` (text debug path)
- `WS /ws_audio/{session_id}` (audio path)
- `GET /benchmark/translation`
- `POST /benchmark/translation/reset`
- `POST /glossary/upload` (upload glossary JSON; hot reload)

## 6) Glossary Upload Format

- Upload entry point: `POST /glossary/upload` or `/webrtc` page upload button
- JSON only, top-level must be an object
- Key format: `"source_lang->target_lang"` (for example `en->zh`)
- Value format: `{ "source term": "target term" }`
- Reference template file: `backend/data/glossary.template.json`

Example:

```json
{
  "en->zh": {
    "ai agent": "ai agent",
    "real-time translation": "实时翻译"
  },
  "zh->en": {
    "实时翻译": "real-time translation"
  }
}
```

## 7) Translation Routing Modes (Explained)

Your understanding is mostly correct. Current routing has a fast-path layer before confidence-based routing:

1. `exact_cache`: exact sentence cache hit, return immediately
2. `glossary`: exact glossary hit, return immediately
3. `tm_exact`: exact TM hit, return immediately
4. confidence-based TM routing:
   - high confidence (`score >= TRANSLATION_ROUTER_FUZZY_THRESHOLD`): `tm_fuzzy`, directly reuse TM translation
   - medium confidence (`TRANSLATION_ROUTER_TM_POLISH_THRESHOLD <= score < TRANSLATION_ROUTER_FUZZY_THRESHOLD`): `tm_polish`, LLM post-edit on TM candidate
   - low confidence (`score < TRANSLATION_ROUTER_TM_POLISH_THRESHOLD`): skip TM reuse and go to full `llm` translation
5. `llm`: full fallback translation

Notes:

- Confidence score uses max of `Jaccard 3-gram` and `Levenshtein similarity`
- Default thresholds: `fuzzy=0.93`, `tm_polish=0.85`
- If `tm_polish` fails, system logs warning and falls back to `llm`
