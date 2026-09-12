# Benchmark scripts

Self-contained bash benchmark scripts for the **MacBook Air M4** run. They are not portable to Linux/Windows — memory and swap measurement relies on macOS-specific commands (`sysctl vm.swapusage`, `memory_pressure`, `vm_stat`).

## Layout

```
scripts/
├── bench.sh          # shared runner: all benchmark logic, once
├── configs/          # per-model settings (one .sh per model)
├── *.sh              # thin per-model wrappers, one per model
├── context_gen.py    # regenerates the long context-*.txt prompts
└── results/          # benchmark outputs (gitignored)
```

## Running a benchmark

Each benchmark expects its `llama-server` binary and model GGUF in the parent directory (e.g. `../<model>.gguf`), and a **running** `llama-server` instance on port 8080.

Two equivalent ways:

```sh
# via the per-model wrapper
./Ornith-1-9B-Q4_K_M.sh

# or directly, listing available configs with no argument
./bench.sh qwen3.6-35B-A3B-UD-IQ2_XXS_Instruct
```

The wrapper just execs `bench.sh <name>` with the matching config.

## Config files

Each `configs/<name>.sh` sets:

- `MODEL_FILE` — GGUF filename, relative to the parent directory
- `SERVER_FLAGS` — the `llama-server` flags (shown in the "Start:" hint if the server is not running)
- `LLAMA_SERVER` — optional binary path override (the Prism runs used a locally built `build/bin/llama-server`)

## Outputs

Per run, results are written to `scripts/results/` (gitignored):

- `llama_benchmark_<timestamp>.csv` — benchmark numbers
- `llama_output_<timestamp>.txt` — full transcript

## Context prompts

`context_gen.py` regenerates the deliberately long, repetitive `context-32k/64k/128k/256k.txt` prompts (gitignored), used to fill large context windows reproducibly:

```sh
cd scripts && python context_gen.py
```
