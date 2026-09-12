# Ornith 1.0 9B (Q4_K_M)

MODEL_FILE="Ornith-1.0-9B-Q4_K_M.gguf"

SERVER_FLAGS=(
    -ngl 99
    --ctx-size 32768
    --temp 0.6 --top-p 0.95 --top-k 20
    --chat-template-kwargs '{"enable_thinking":false}'
    -fa on
    -np 1
    --port 8080
)
