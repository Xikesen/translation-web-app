#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

# Low-latency defaults tuned for Apple Silicon (M-series) with 16GB:
#   - translation: small fast LLM that stays resident on the GPU (~0.3s warm)
#   - ASR: Whisper on the Apple GPU via MLX/Metal (~0.35s per segment)
# qwen2.5:3b = low latency (default). Set OLLAMA_MODEL=qwen2.5:7b for "quality mode".
OLLAMA_MODEL_DEFAULT="qwen2.5:3b"
OLLAMA_MODEL="${OLLAMA_MODEL:-$OLLAMA_MODEL_DEFAULT}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL_CANDIDATES="${OLLAMA_MODEL_CANDIDATES:-qwen2.5:3b qwen2.5:7b gemma2:2b llama3.2:3b}"

ASR_BACKEND_DEFAULT="mlx_whisper_local"
ASR_BACKEND="${ASR_BACKEND:-$ASR_BACKEND_DEFAULT}"
if [[ "$ASR_BACKEND" == "mlx_whisper_local" ]]; then
  ASR_MODEL_NAME="${ASR_MODEL_NAME:-mlx-community/whisper-small-mlx}"
else
  ASR_MODEL_NAME="${ASR_MODEL_NAME:-small}"
fi
ASR_DEVICE="${ASR_DEVICE:-cpu}"
ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-int8}"
# Empty = auto-detect language (needed for zh/en/fr/es mix). Set e.g. "en" to force.
ASR_LANGUAGE_HINT="${ASR_LANGUAGE_HINT-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found."
    exit 1
  fi
}

print_step() {
  echo
  echo "==> $1"
}

ensure_ollama_installed() {
  if command -v ollama >/dev/null 2>&1; then
    return
  fi
  if ! command -v brew >/dev/null 2>&1; then
    echo "Error: 'ollama' not found and Homebrew is not installed."
    echo "Install Homebrew first, then rerun this script."
    exit 1
  fi
  print_step "Installing Ollama via Homebrew"
  brew install ollama
}

ensure_ollama_running() {
  if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    return
  fi
  print_step "Starting Ollama in background"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &

  for _ in {1..20}; do
    sleep 1
    if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      return
    fi
  done

  echo "Error: Ollama did not become ready at $OLLAMA_URL."
  echo "Check logs: /tmp/ollama-serve.log"
  exit 1
}

ensure_model_pulled() {
  local selected_model=""
  local requested_model="$OLLAMA_MODEL"

  print_step "Ensuring Ollama model is available: $requested_model"
  if ollama pull "$requested_model"; then
    selected_model="$requested_model"
  else
    echo "Warning: failed to pull requested model '$requested_model'."
    echo "Trying local installed models and fallback candidates..."

    local local_model
    local_model="$(ollama list | awk 'NR==2 {print $1}')"
    if [[ -n "$local_model" && "$local_model" != "NAME" ]]; then
      selected_model="$local_model"
      echo "Using already-installed local model: $selected_model"
    else
      local candidate
      for candidate in $OLLAMA_MODEL_CANDIDATES; do
        if ollama pull "$candidate"; then
          selected_model="$candidate"
          echo "Using fallback model: $selected_model"
          break
        fi
      done
    fi
  fi

  if [[ -z "$selected_model" ]]; then
    echo "Error: unable to prepare any Ollama model."
    echo "Try manually: ollama pull gemma3:4b"
    exit 1
  fi

  OLLAMA_MODEL="$selected_model"
}

ensure_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python3 is required."
    exit 1
  fi
}

setup_venv_and_install() {
  print_step "Setting up Python virtualenv"
  mkdir -p "$BACKEND_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  print_step "Installing backend dependencies"
  python -m pip install -U pip setuptools wheel
  pip install -r "$BACKEND_DIR/requirements.txt"

  # Apple Silicon GPU ASR (MLX/Metal). Only needed/available on arm64 macOS.
  if [[ "$ASR_BACKEND" == "mlx_whisper_local" ]]; then
    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
      print_step "Installing MLX Whisper (Apple GPU ASR)"
      pip install "mlx-whisper" "scipy==1.11.4"
    else
      echo "Warning: ASR_BACKEND=mlx_whisper_local requires Apple Silicon macOS."
      echo "Falling back to faster_whisper_local on CPU."
      ASR_BACKEND="faster_whisper_local"
      ASR_MODEL_NAME="small"
    fi
  fi
}

export_runtime_env() {
  export MANAGER_BACKEND="${MANAGER_BACKEND:-ollama}"
  export OLLAMA_URL="$OLLAMA_URL"
  export OLLAMA_MODEL="$OLLAMA_MODEL"
  export ASR_BACKEND="$ASR_BACKEND"
  export ASR_MODEL_NAME="$ASR_MODEL_NAME"
  export ASR_DEVICE="$ASR_DEVICE"
  export ASR_COMPUTE_TYPE="$ASR_COMPUTE_TYPE"
  export ASR_LANGUAGE_HINT="$ASR_LANGUAGE_HINT"
}

start_backend() {
  print_step "Starting backend"
  echo "Backend URL: http://$HOST:$PORT"
  echo "ASR backend: $ASR_BACKEND ($ASR_MODEL_NAME, device=$ASR_DEVICE, compute=$ASR_COMPUTE_TYPE)"
  echo "Manager backend: $MANAGER_BACKEND (Ollama model: $OLLAMA_MODEL)"
  echo "AR glasses page: http://$HOST:$PORT/   (also /glass)"
  echo "WebRTC page:     http://$HOST:$PORT/webrtc"
  echo
  exec uvicorn main:app --host "$HOST" --port "$PORT"
}

main() {
  if [[ ! -d "$BACKEND_DIR" ]]; then
    echo "Error: backend directory not found at $BACKEND_DIR"
    exit 1
  fi

  require_cmd curl
  ensure_ollama_installed
  require_cmd ollama
  ensure_ollama_running
  ensure_model_pulled

  ensure_python
  setup_venv_and_install
  export_runtime_env

  cd "$BACKEND_DIR"
  start_backend
}

main "$@"
