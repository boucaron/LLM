# Gemma 4 26B A4B (IQ2_XXS, reasoning off, quantized KV cache)

MODEL_FILE="gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf"

SERVER_FLAGS=(
    --ctx-size 32768
    --temp 1.0
    --top-p 0.95
    --top-k 64
    -fa on
    --cache-type-k q4_0
    --cache-type-v q4_0
    -np 1
    --reasoning off
    --port 8080
)
