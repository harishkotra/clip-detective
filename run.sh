#!/usr/bin/env bash
# Start Clip Detective: llama-server (LFM3-VL-2B) + the local web app.
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp-src/build/bin}"
MODEL="${MODEL:-$PWD/models/LFM3-VL-2B-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$PWD/models/mmproj-LFM3-VL-2B-Q8_0.gguf}"
PORT_MODEL="${PORT_MODEL:-9911}"
PORT_APP="${PORT_APP:-8787}"
export LLAMA_URL="http://127.0.0.1:$PORT_MODEL"

for f in "$MODEL" "$MMPROJ"; do
  [ -s "$f" ] || { echo "missing model file: $f"; echo "run ./setup.sh first"; exit 1; }
done
[ -x "$LLAMA_BIN/llama-server" ] || { echo "missing $LLAMA_BIN/llama-server"; echo "run ./setup.sh first"; exit 1; }

if ! curl -sf --noproxy '*' "http://127.0.0.1:$PORT_MODEL/health" >/dev/null 2>&1; then
  echo "starting LFM3-VL-2B on :$PORT_MODEL"
  "$LLAMA_BIN/llama-server" \
    -m "$MODEL" --mmproj "$MMPROJ" \
    --host 127.0.0.1 --port "$PORT_MODEL" \
    -c 16384 -np 1 -ngl 99 --no-warmup \
    --reasoning off \
    --temp 0.1 --top-p 1.0 --top-k 50 --min-p 0.0 --repeat-penalty 1.1 \
    > /tmp/clip-detective-model.log 2>&1 &
  for i in $(seq 1 300); do
    curl -sf --noproxy '*' "http://127.0.0.1:$PORT_MODEL/health" >/dev/null 2>&1 && break
    [ $((i % 15)) -eq 0 ] && printf "  still loading (%ss)\n" "$i"
    sleep 1
  done
  curl -sf --noproxy '*' "http://127.0.0.1:$PORT_MODEL/health" >/dev/null 2>&1 \
    || { echo "model server failed, see /tmp/clip-detective-model.log"; exit 1; }
fi

echo "app: http://127.0.0.1:$PORT_APP"
PORT=$PORT_APP exec python3 server.py
