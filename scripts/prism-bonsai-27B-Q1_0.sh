#!/bin/bash
# Benchmark runner for prism-bonsai-27B-Q1_0.
#
# Shared logic lives in bench.sh; model-specific settings live in
# configs/prism-bonsai-27B-Q1_0.sh.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench.sh" "$(basename "${BASH_SOURCE[0]}" .sh)"
