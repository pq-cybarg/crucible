#!/usr/bin/env bash
# Launch the full Crucible stack: llama-server (if a GGUF exists), backend, frontend.
# Set CRUCIBLE_HF_MODEL=<hf_path> before running to enable live torch abliteration.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

GGUF=$(find models -name "*.gguf" 2>/dev/null | head -1)
if [ -n "$GGUF" ]; then
  pkill -f "llama-server" 2>/dev/null || true
  sleep 1
  llama-server --model "$GGUF" --port 8081 --ctx-size 32768 --parallel 4 \
    --n-gpu-layers 999 --host 127.0.0.1 >/tmp/crucible-llama.log 2>&1 &
  echo "llama-server  -> http://127.0.0.1:8081   ($(basename "$GGUF"))"
fi

pkill -f "uvicorn crucible.app:app" 2>/dev/null || true
sleep 1
uvicorn crucible.app:app --host 127.0.0.1 --port 8400 >/tmp/crucible-api.log 2>&1 &
echo "backend       -> http://127.0.0.1:8400   ${CRUCIBLE_HF_MODEL:+(torch adapter: $CRUCIBLE_HF_MODEL)}"

( cd frontend && npm run dev >/tmp/crucible-web.log 2>&1 & )
echo "frontend      -> http://localhost:5273"
echo ""
echo "Crucible is up. Logs: /tmp/crucible-{llama,api,web}.log"
