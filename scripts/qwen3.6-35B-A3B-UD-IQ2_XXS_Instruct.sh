#!/bin/bash
# Benchmark runner for qwen3.6-35B-A3B-UD-IQ2_XXS_Instruct.
#
# Shared logic lives in bench.sh; model-specific settings live in
# configs/qwen3.6-35B-A3B-UD-IQ2_XXS_Instruct.sh.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench.sh" "$(basename "${BASH_SOURCE[0]}" .sh)"
