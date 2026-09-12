#!/bin/bash
# Benchmark runner for Ornith-1-9B-Q4_K_M.
#
# Shared logic lives in bench.sh; model-specific settings live in
# configs/Ornith-1-9B-Q4_K_M.sh.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench.sh" "$(basename "${BASH_SOURCE[0]}" .sh)"
