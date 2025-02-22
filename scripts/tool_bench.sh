#!/bin/bash
set -euo pipefail

cmake --build build -j

# Useful for ollama
# export LLAMA_SERVER_TEST_REQUEST_RETRIES=${RETRIES:-3}

export LLAMA_CACHE=${LLAMA_CACHE:-$HOME/Library/Caches/llama.cpp}
export LLAMA_SERVER_BIN_PATH=$PWD/build/bin/llama-server

if [ ! -x "$LLAMA_SERVER_BIN_PATH" ]; then
    echo "Could not find llama-server binary at $LLAMA_SERVER_BIN_PATH"
    exit 1
fi
if [ ! -d "$LLAMA_CACHE" ]; then
    echo "Could not find llama cache at $LLAMA_CACHE, please set LLAMA_CACHE explicitly."
    exit 1
fi

export ARGS=(
    --llama-baseline="$(which llama-server)"
    --n 30
    --temp -1  # Leaves temperature parameter unset (use the server's default, e.g. 0.6 for ollama)
    --temp 0
    --temp 0.5
    --temp 0.75
    --temp 1
    --temp 1.5
    --temp 2
    --temp 5
    "$@"
) 

./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 7B Q4_K_M"      --output ../qwenc7b.jsonl   --hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M   --ollama qwen2.5-coder:7b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 1.5B Q4_K_M"          --output ../qwen1.5b.jsonl  --hf bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M       --ollama qwen2.5:1.5b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.2 Instruct 1B Q4_K_M"  --output ../llama1b.jsonl   --hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M       --ollama llama3.2:1b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.2 Instruct 3B Q4_K_M"  --output ../llama3b.jsonl   --hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M       --ollama llama3.2:3b-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.1 Instruct 8B Q4_K_M"  --output ../llama8b.jsonl   --hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M  --ollama llama3.1:8b-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Mistral Nemo Q4_K_M"           --output ../nemo.jsonl      --hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M  --ollama mistral-nemo:12b-instruct-2407-q4_K_M

# ./scripts/tool_bench.py run ${ARGS[@]} --model "Phi 4 Instruct Q4_K_M"         --output ../phi4.jsonl      --hf bartowski/phi-4-GGUF:Q4_K_M                       # --ollama phi4 
# ./scripts/tool_bench.py run ${ARGS[@]} --model "Phi 3.5 Mini Instruct Q4_K_M"  --output ../phi3.5.jsonl    --hf bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M       # --ollama phi3.5:3.8b-mini-instruct-q4_K_M 

for f in *.jsonl; do
    ./scripts/tool_bench.py plot "$f" --output ${f%.jsonl}.png
done