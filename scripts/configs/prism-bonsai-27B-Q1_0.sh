# Prism Bonsai 27B (Q1_0)

MODEL_FILE="Bonsai-27B-Q1_0.gguf"

# This benchmark used a locally built llama-server
LLAMA_SERVER="${SCRIPT_DIR}/build/bin/llama-server"

SERVER_FLAGS=(
    --ctx-size 32768
    --temp 0.7
    --top-p 0.95
    --top-k 20
    -ngl 99
    -fa on
    --cache-type-k q4_0
    --cache-type-v q4_0
    -np 1
    --port 8080
)
