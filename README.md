# Realtime Translation MVP (Local)

这是一个可运行的最小骨架，当前技术路径为：
- **WebRTC 音频 -> VAD -> faster-whisper 转写 -> Ollama(Gemma 4-E4B Q8_0) 文本翻译**

当前目标：
- 会话开始/结束（最多 10 分钟）
- 多说话人映射（A/B/C，最多 3 人）
- 实时消息处理
- 翻译到目标语言（英语/法语/中文/西班牙语）
- 说话窗口生命周期（结束后保留最后 2 句，3 秒后关闭）

## 1) 环境要求

- macOS (Apple Silicon / Intel)
- Python 3.11+
- [Ollama](https://ollama.com/)（本地 LLM 服务）

## 2) 安装并启动 Gemma 模型

安装 Ollama（Homebrew）：

```bash
brew install ollama
```

启动 Ollama 服务：

```bash
ollama serve
```

拉取模型：

```bash
ollama pull gemma-4-e4b:q8_0
```

测试模型可用性：

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

## 3) 启动后端 (FastAPI)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=gemma-4-e4b:q8_0
export OLLAMA_TIMEOUT_SECONDS=120
export OLLAMA_KEEP_ALIVE=30m
export TARGET_LANG_DEFAULT=en

export TRANSLATION_MAX_NEW_TOKENS=96
export TRANSLATION_TEMPERATURE=0.0
export TRANSLATION_TOP_P=0.9

export ASR_BATCH_MS=400
export ASR_COMMIT_SILENCE_MS=450
export ASR_MIN_COMMIT_MS=700
export ASR_MAX_SEGMENT_MS=1800

export LID_SMOOTH_WINDOW=5
export LID_MIN_CONFIDENCE=0.65

export MERGE_SHORT_MAX_CHARS=18
export MERGE_MAX_WAIT_SECONDS=2.0
export MERGE_FORCE_CHARS=48

uvicorn main:app --reload --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

## 4) 使用方式（WebRTC）

后端启动后，打开：

`http://localhost:8000/webrtc`

页面点击 **Start** 后会：
- 自动创建会话
- 通过浏览器麦克风采集音频（WebAudio）
- 下采样到 16k PCM 并通过 `ws_audio` 发送给 FastAPI
- 后端先回推 `transcript`，再异步回推 `translation`
- 内置 LID 平滑（减少 source_lang 抖动）
- 内置短句合并提交（减少切碎翻译）

页面固定两个窗口：
- Transcript（讲话者原文）
- Translation（可切换 target language）

目标语言支持：`en/fr/zh/es`，会话中可切换。

## 5) 翻译后端说明

- 当前仅使用 **Gemma 4-E4B Q8_0（Ollama）**
- 后端调用 `POST /api/chat`
- 若 Ollama 未启动或模型未拉取，页面 Events 会显示 `translation failed: ...`
- 首次请求可能较慢（模型冷启动）

## 6) 事件协议（核心）

后端推送事件：
- `connected`
- `audio_connected`
- `speaker_start`
- `transcript`
- `translation`
- `target_lang_updated`
- `speaker_end`
- `speaker_expire`
- `session_end`
- `error`

## 7) 下一步扩展

- 优化 WebRTC 前端（AudioWorklet 替代 ScriptProcessor）
- 持续调优 `VAD + faster-whisper` 参数
- 引入 benchmark（chunk 切分质量 + 翻译准确度）
# ooni-glass-v1
