# Qwen3.6 35B A3B (IQ2_XXS)

MODEL_FILE="Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"

SERVER_FLAGS=(
    -ngl 99
    --ctx-size 32768
    --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
    --repeat-penalty 1.00 --presence-penalty 1.5
    --chat-template-kwargs '{"enable_thinking":false}'
    -fa on
    --cache-type-k q4_0
    --cache-type-v q4_0
    -np 1
    --port 8080
)
