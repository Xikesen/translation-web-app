# Realtime Translation MVP (Local)

这是一个可运行的最小骨架，翻译后端已切换为：
- **本地 Ollama + Qwen3-8B（唯一翻译后端）**

目标是先跑通：
- 会话开始/结束（最多 2 分钟）
- 多说话人映射（A/B/C/D，最多 4 人）
- 实时消息处理
- 翻译到目标语言（英语/法语/中文/日语）
- 说话窗口生命周期（结束后保留最后 2 句，3 秒后关闭）
- 屏幕最多显示 3 个说话窗口

> 当前版本用“文本输入”模拟语音流，方便先验证后端协议和 UI 行为；后续可接入 `streamlit-webrtc + ASR + 声纹识别`。

## 1) 环境要求

- macOS (Apple Silicon / Intel 均可)
- Python 3.11+
- [Ollama](https://ollama.com/)（本地 LLM 服务）

## 2) 安装并启动 Qwen3-8B

安装 Ollama（Homebrew）：

```bash
brew install ollama
```

启动 Ollama 服务（新终端）：

```bash
ollama serve
```

拉取模型（新终端）：

```bash
ollama pull qwen3:8b
```

测试模型是否可用：

```bash
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b","prompt":"Translate hello to French. Output only translation.","stream":false}'
```

## 3) 启动后端 (FastAPI)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3:8b
export OLLAMA_TIMEOUT_SECONDS=120
export OLLAMA_KEEP_ALIVE=30m
uvicorn main:app --reload --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

## 4) 启动前端 (Streamlit)

新开一个终端：

```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

打开浏览器：`http://localhost:8501`

## 5) 使用方式

1. 选择目标语言（`en/fr/zh/ja`）。
2. 点击 **Start** 创建会话并连接 WebSocket。
3. 在表单中选择 `Speaker Key`（如 `mic-1`）和输入文本，点击 **Send Utterance**。
4. 观察右侧“Speaker Windows”：
   - 同一个 `Speaker Key` 始终映射到固定角色（A/B/C/D）。
   - 同时显示最多 3 个窗口。
   - 某说话人停止输入后，窗口先保留最后 2 句，3 秒后自动消失。
5. 点击 **Stop** 结束会话（或等待 2 分钟自动结束）。

## 6) 翻译后端说明

- 当前仅使用 **Qwen3-8B（Ollama）**
- 后端调用 `POST /api/generate`
- 若 Ollama 未启动或模型未拉取，页面 Events 会显示 `translation failed: ...`
- 首次请求可能较慢（模型冷启动），建议等待 10-40 秒

## 7) 事件协议（核心）

前端发送：

```json
{
  "type": "utterance",
  "speaker_key": "mic-1",
  "text": "Hello everyone",
  "source_lang": "en"
}
```

后端推送事件：
- `connected`
- `speaker_start`
- `utterance`
- `speaker_end`
- `speaker_expire`
- `session_end`
- `error`

## 8) 下一步扩展（从 MVP 到真实语音）

- 接入 `streamlit-webrtc` 采集麦克风 PCM/Opus 分片
- 后端加入 `VAD`（webrtcvad/silero-vad）
- 接入 `faster-whisper` 做 ASR
- 接入 `pyannote.audio` 做说话人分离与声纹匹配
- 可选：接入真正的麦克风音频流后，把 source_lang 改为自动检测
# Realtime-translation-mvp
