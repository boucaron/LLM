#!/bin/bash

set -euo pipefail

#####################################
# Configuration
#####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_SERVER="$SCRIPT_DIR/build/bin/llama-server"
MODEL="$SCRIPT_DIR/../Bonsai-27B-Q1_0.gguf"

SERVER_URL="http://127.0.0.1:8080"

OUTPUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

OUTPUT_FILE="$OUTPUT_DIR/llama_benchmark_$TIMESTAMP.csv"
TRANSCRIPT_FILE="$OUTPUT_DIR/llama_output_$TIMESTAMP.txt"


#####################################
# Prompts
#####################################

PROMPTS=(
"I think that the 42 answer is also a sarcastic way to what question matters and the importance of a good question. What do you think."
"Can you enumerate such questions."
"I would say the usage of a llm is a bit like asking such kind of question."
)


#####################################
# Helpers
#####################################

error()
{
    echo
    echo "ERROR: $1"
    echo
    exit 1
}


ok()
{
    echo "OK: $1"
}


#####################################
# Memory functions
#####################################

get_llama_memory()
{
    PID=$(pgrep -f "llama-server" | head -1 || true)

    if [[ -n "$PID" ]]; then
        RSS=$(ps -o rss= -p "$PID" | tr -d ' ')
        echo $((RSS / 1024))
    else
        echo "0"
    fi
}


get_system_memory()
{
    PAGE_SIZE=$(vm_stat | grep "page size" | awk '{print $8}')

    FREE=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
    ACTIVE=$(vm_stat | grep "Pages active" | awk '{print $3}' | tr -d '.')
    INACTIVE=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
    WIRED=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | tr -d '.')


    FREE_MB=$((FREE * PAGE_SIZE / 1024 / 1024))

    USED_MB=$(((ACTIVE + INACTIVE + WIRED) * PAGE_SIZE / 1024 / 1024))


    SWAP_USED=$(sysctl vm.swapusage \
        | awk '{print $8}' \
        | sed 's/M//')


    PRESSURE=$(memory_pressure 2>/dev/null \
        | grep "memory pressure" \
        | awk -F": " '{print $2}' \
        || echo "unknown")


    echo "$USED_MB,$FREE_MB,$SWAP_USED,$PRESSURE"
}


#####################################
# Environment checks
#####################################

echo
echo "=== llama.cpp chat benchmark ==="
echo


[[ "$(uname)" == "Darwin" ]] \
    || error "This script requires macOS."

ok "macOS detected"


for CMD in curl jq ps vm_stat sysctl memory_pressure; do
    command -v "$CMD" >/dev/null 2>&1 \
        || error "Missing command: $CMD"
done

ok "System tools available"


#####################################
# llama checks
#####################################

[[ -f "$LLAMA_SERVER" ]] \
    || error "llama-server missing: $LLAMA_SERVER"


[[ -x "$LLAMA_SERVER" ]] \
    || error "llama-server is not executable"


ok "llama-server ready"


#####################################
# Model checks
#####################################

[[ -f "$MODEL" ]] \
    || error "Model missing: $MODEL"


[[ -r "$MODEL" ]] \
    || error "Model not readable"


MODEL_NAME=$(basename "$MODEL")
MODEL_SIZE=$(du -h "$MODEL" | awk '{print $1}')


ok "Model: $MODEL_NAME"
ok "Size : $MODEL_SIZE"


#####################################
# llama version
#####################################

LLAMA_VERSION=$(
    "$LLAMA_SERVER" --version 2>/dev/null \
    | head -1 \
    || echo "unknown"
)


ok "Version: $LLAMA_VERSION"


#####################################
# Server checks
#####################################

echo
echo "Checking llama-server..."


curl -s --connect-timeout 5 "$SERVER_URL/health" >/dev/null \
|| error "
llama-server is not running.

Start:

$LLAMA_SERVER \
-m $MODEL \
--ctx-size 32768 \
--temp 0.7 \
--top-p 0.95 \
--top-k 20 \
-ngl 99 \
-fa on \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
-np 1 \
--port 8080
"


ok "Server reachable"


if ! curl -s "$SERVER_URL/v1/models" >/dev/null; then
    error "OpenAI compatible API not available."
fi


ok "Chat API available"



#####################################
# CSV
#####################################

echo \
"timestamp,llama_version,model,turn,prompt,prompt_tokens,predicted_tokens,prompt_ms,predicted_ms,prompt_tok_s,predicted_tok_s,total_ms,llama_memory_mb,system_used_mb,system_free_mb,swap_used_mb,memory_pressure" \
> "$OUTPUT_FILE"


#####################################
# Transcript
#####################################

cat > "$TRANSCRIPT_FILE" <<EOF
llama.cpp benchmark transcript

Timestamp:
$TIMESTAMP

Model:
$MODEL_NAME

Version:
$LLAMA_VERSION

=================================

EOF


#####################################
# Conversation state
#####################################

MESSAGES='[]'


#####################################
# Benchmark
#####################################

for i in "${!PROMPTS[@]}"; do

TURN=$((i+1))
PROMPT="${PROMPTS[$i]}"


echo
echo "--------------------------------"
echo "Turn $TURN"
echo "$PROMPT"


MESSAGES=$(echo "$MESSAGES" | jq \
--arg p "$PROMPT" \
'. + [{
    role:"user",
    content:$p
}]')


RESPONSE=$(curl -s "$SERVER_URL/v1/chat/completions" \
-H "Content-Type: application/json" \
-d "$(jq -n \
--argjson messages "$MESSAGES" \
'{
    model:"local-model",
    messages:$messages,
    max_tokens:32768
}')")


echo $RESPONSE

echo "$RESPONSE" | jq empty >/dev/null \
|| error "Invalid JSON returned"


LLM_OUTPUT=$(echo "$RESPONSE" \
| jq -r '.choices[0].message.content // "(empty response)"')


MESSAGES=$(echo "$MESSAGES" | jq \
--arg a "$LLM_OUTPUT" \
'. + [{
    role:"assistant",
    content:$a
}]')


PROMPT_TOKENS=$(echo "$RESPONSE" \
| jq -r '.usage.prompt_tokens // 0')


PRED_TOKENS=$(echo "$RESPONSE" \
| jq -r '.usage.completion_tokens // 0')


PROMPT_MS=$(echo "$RESPONSE" \
| jq -r '.timings.prompt_ms // 0')


PRED_MS=$(echo "$RESPONSE" \
| jq -r '.timings.predicted_ms // 0')


PROMPT_TS=$(echo "$RESPONSE" \
| jq -r '.timings.prompt_per_second // 0')


PRED_TS=$(echo "$RESPONSE" \
| jq -r '.timings.predicted_per_second // 0')


TOTAL_MS=$(awk "BEGIN {print $PROMPT_MS+$PRED_MS}")


LLAMA_MEM=$(get_llama_memory)


IFS=',' read \
SYSTEM_USED \
SYSTEM_FREE \
SWAP_USED \
PRESSURE <<< "$(get_system_memory)"


#####################################
# CSV output
#####################################

echo \
"\"$TIMESTAMP\",\"$LLAMA_VERSION\",\"$MODEL_NAME\",\"$TURN\",\"$PROMPT\",$PROMPT_TOKENS,$PRED_TOKENS,$PROMPT_MS,$PRED_MS,$PROMPT_TS,$PRED_TS,$TOTAL_MS,$LLAMA_MEM,$SYSTEM_USED,$SYSTEM_FREE,$SWAP_USED,\"$PRESSURE\"" \
>> "$OUTPUT_FILE"



#####################################
# Transcript
#####################################

cat >> "$TRANSCRIPT_FILE" <<EOF

=================================
TURN $TURN

PROMPT:
$PROMPT

ANSWER:
$LLM_OUTPUT


METRICS:

Prompt tokens:
$PROMPT_TOKENS

Generated tokens:
$PRED_TOKENS

Prompt speed:
$PROMPT_TS tok/s

Generation speed:
$PRED_TS tok/s

Total time:
$TOTAL_MS ms

llama memory:
${LLAMA_MEM} MB

System used:
${SYSTEM_USED} MB

System free:
${SYSTEM_FREE} MB

Swap:
${SWAP_USED} MB

Memory pressure:
$PRESSURE

EOF


echo "Generated : $PRED_TOKENS tokens"
echo "Speed     : $PRED_TS tok/s"
echo "Memory    : ${LLAMA_MEM} MB"


done


echo
echo "================================="
echo "Benchmark complete"
echo
echo "CSV:"
echo "$OUTPUT_FILE"
echo
echo "Transcript:"
echo "$TRANSCRIPT_FILE"
echo "================================="