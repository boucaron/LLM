#!/bin/bash
# Benchmark runner for gemma-4-26B-A4B-it-UD-IQ2_XXS_reasoning_off_kvcache_tinybench.
#
# Shared logic lives in bench.sh; model-specific settings live in
# configs/gemma-4-26B-A4B-it-UD-IQ2_XXS_reasoning_off_kvcache_tinybench.sh.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench.sh" "$(basename "${BASH_SOURCE[0]}" .sh)"
