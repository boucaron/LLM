#!/bin/bash
# Benchmark runner for qwen3-coder-30B-A3B-Instruct-UD-IQ2_XXS.
#
# Shared logic lives in bench.sh; model-specific settings live in
# configs/qwen3-coder-30B-A3B-Instruct-UD-IQ2_XXS.sh.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench.sh" "$(basename "${BASH_SOURCE[0]}" .sh)"
