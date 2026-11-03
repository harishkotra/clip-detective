#!/usr/bin/env bash
# One-time setup for Clip Detective: models + a llama.cpp build that supports LFM3.
# Safe to re-run; it skips whatever is already done.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-src}"

say() { printf "\n==> %s\n" "$1"; }

for t in git cmake python3 curl; do
  command -v "$t" >/dev/null || { echo "missing '$t'. install it first."; exit 1; }
done
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (brew install ffmpeg / apt install ffmpeg)"

say "1/3 models (1.2GB, skipped if present)"
mkdir -p models
BASE="https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/main"
for f in LFM3-VL-2B-Q4_K_M.gguf mmproj-LFM3-VL-2B-Q8_0.gguf; do
  if [ -s "models/$f" ]; then
    echo "have models/$f"
  else
    echo "downloading $f"
    curl -L --progress-bar -o "models/$f" "$BASE/$f"
  fi
done

say "2/3 llama.cpp (LFM3 support + Metal fix)"
if [ -x "$LLAMA_DIR/build/bin/llama-server" ]; then
  echo "have $LLAMA_DIR/build/bin/llama-server"
else
  if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  python3 "$ROOT/scripts/fix-metal-bf16.py" "$LLAMA_DIR"
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
  GPU_FLAG=""
  [ "$(uname -s)" = "Darwin" ] && GPU_FLAG="-DGGML_METAL=ON"
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" $GPU_FLAG -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release -j "$JOBS"
fi

say "3/3 done"
echo "start it with:  ./run.sh"
echo "then open:      http://127.0.0.1:8787"
