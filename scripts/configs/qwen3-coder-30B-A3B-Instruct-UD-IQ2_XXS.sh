# Qwen3 Coder 30B A3B (IQ2_XXS)

MODEL_FILE="Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf"

SERVER_FLAGS=(
    --jinja
    -ngl 99
    --ctx-size 32768
    --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --repeat-penalty 1.05
    -fa on
    --cache-type-k q4_0
    --cache-type-v q4_0
    -np 1
    --port 8080
)
