# LLM — Local LLM Experiments

A collection of practical local LLM experiments: benchmarking GGUF models with `llama.cpp`, reusable benchmark scripts, server configurations, and one-shot HTML generation experiments.

The focus is on **practical usability** of local models (inference speed, memory/VRAM usage, thermal behavior) rather than model quality comparisons.

**TL;DR:** Benchmarked 20+ GGUF models (7B–35B, dense and MoE) with `llama.cpp` on an RTX 5060 Ti 16 GB and a MacBook Air M4 24 GB. Key results: 25–35B MoE models run at **80–105 t/s fully in VRAM** at aggressive quantization, **~137 t/s with MTP**, and **100K–256K context** is possible via Q4 KV-cache + MoE CPU offloading. A 16 GB GPU is a genuinely practical local-LLM machine.

## Repository layout

```
.
├── macbook-air-m4-local-llm-benchmark.md
├── rtx5060ti-local-llm-benchmark.md
├── rtx5060ti-experimental-results.md
├── rtx5060ti-qwen3.8-27b-experiments.md
├── rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md
├── llamacpp/
│   └── model.ini
├── scripts/
├── games/
└── others/
```

## Benchmark reports

### [MacBook Air M4 — 24 GB](macbook-air-m4-local-llm-benchmark.md)
Practical local LLM performance on a fanless MacBook Air M4 (24 GB unified memory).
Tests of recent GGUF models (Gemma 4 26B A4B, Qwen 3.x, LFM, Prism Bonsai, Ornith, GLM…) under sustained operation, with attention to thermal throttling, memory usage, and quantized KV cache.

### [RTX 5060 Ti — 16 GB](rtx5060ti-local-llm-benchmark.md)
Practical local LLM performance on a dedicated 16 GB GPU.
The main report covers the methodology, key findings, trade-offs, and practical recommendations.
Highlights: ~80–105 t/s for 25–35B MoE models fully in VRAM, ~130+ t/s with MTP, and very large context (100K–256K) with CPU offloading of MoE layers.

Companion files with the detailed per-configuration measurements:

- [rtx5060ti-experimental-results.md](rtx5060ti-experimental-results.md) — Qwen 3.6 generation (27B, 35B A3B), Gemma 4 26B, Qwen 3 Coder 30B, other models, CPU offloading, reproducibility
- [rtx5060ti-qwen3.8-27b-experiments.md](rtx5060ti-qwen3.8-27b-experiments.md) — Qwen 3.8 27B across D2/D3 quantizations and ISTA DASLab variants

> Last updated **12/09/2026** — most recent additions: Qwen 3.8 27B in `IQ3_XXS GSQ RCO` (with MTP and DFlash2), Qwen 3.8 27B with Unsloth Dynamic Quant v3.0 (`UD-IQ3_S`), and Ornith 1.5 35B A3B.

### [RTX 5060 Ti — Qwen 3.8 27B with DFlash2 speculative decoding](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md)
Focused study of Qwen 3.8 27B (GSQ RCO `IQ3_XXS`) with a DFlash2 draft model (DFlash2 speculative decoding) on the 16 GB RTX 5060 Ti, across context sizes 32K–162K on real coding tasks (game development in several languages).
Highlights: ~60–65 t/s at 32K–64K, a ~60 t/s burst at 128K dropping to ~40 t/s as the context fills, and a drop to ~23 t/s at 162K where the combined dedicated + shared GPU memory footprint (~16.7 GB) exceeds the card's 16 GB VRAM and offloading becomes the dominant bottleneck.

## Benchmark scripts

### `scripts/`
Self-contained bash benchmark scripts for the **MacBook Air M4** run, one per model. They are not portable to Linux/Windows — memory and swap measurement relies on macOS-specific commands (`sysctl vm.swapusage`, `memory_pressure`, `vm_stat`). Each script:

All scripts are thin wrappers around the shared runner **`bench.sh`**: each one execs `bench.sh <name>`, and the per-model settings (model GGUF name, `llama-server` flags, optional binary path) live in **`scripts/configs/<name>.sh`**. Benchmarks can also be run directly with `./bench.sh <config>` (run with no argument to list the available configs).

Each run:

1. Locates its `llama-server` binary and model GGUF (expected in the parent directory, e.g. `../<model>.gguf`)
2. Starts a `llama-server` instance with the model-specific flags from its config
3. Runs a set of prompts and measures generation throughput, memory usage, and swap/pressure
4. Writes results to `scripts/results/` (gitignored):
   - `llama_benchmark_<timestamp>.csv` — benchmark numbers
   - `llama_output_<timestamp>.txt` — full transcript

| Script | Model |
| --- | --- |
| `Ornith-1-9B-Q4_K_M.sh` | Ornith 1.0 9B (Q4_K_M) |
| `Ornith-1-9B-Q4_K_M_MTP.sh` | Ornith 1.0 9B with MTP speculative decoding |
| `gemma-4-26B-A4B-it-UD-IQ2_XXS_reasoning_off_kvcache_tinybench.sh` | Gemma 4 26B A4B (IQ2_XXS, reasoning off, quantized KV cache) |
| `prism-bonsai-27B-Q1_0.sh` / `prism-bonsai-27B-Q2_0.sh` | Prism Bonsai 27B (Q1_0 / Q2_0) |
| `qwen3-coder-30B-A3B-Instruct-UD-IQ2_XXS.sh` | Qwen3 Coder 30B A3B (IQ2_XXS) |
| `qwen3.6-35B-A3B-UD-IQ2_XXS_Instruct.sh` | Qwen3.6 35B A3B (IQ2_XXS) |

### `scripts/context_gen.py`
Generates a deliberately long, repetitive text prompt used to fill large context windows (up to 256K tokens) reproducibly, so that benchmarks test sustained inference performance rather than content quality.

Running `cd scripts && python context_gen.py` regenerates all four `context-*.txt` files (32K/64K/128K/256K) in the `scripts/` directory; they are gitignored and kept out of the repository since they can be regenerated at any time.

## Llama.cpp configuration

### `llamacpp/model.ini`
`llama-server` configuration profiles (version 1) for Qwen 3.8 27B variants, tuned for the **RTX 5060 Ti 16 GB** run (VRAM-driven context sizes, CPU offloading of MoE layers), all using flash attention, q4_0 KV cache, and medium reasoning effort:

| Profile | Context | Speculative decoding |
| --- | ---: | --- |
| `Qwen3.8-27B-UD-IQ3_S` | 120K | MTP (`draft-mtp`, n-max 2) |
| `JB_...GSQ-RCO-IQ3_XXS` | 240K | none |
| `JBMTP_...GSQ-RCO-IQ3_XXS` | 170K | MTP (`draft-mtp`, n-max 2) |
| `JBMT2P_...GSQ-RCO-IQ3_XXS` | 162K | MTP (`draft-mtp`, n-max 2, q4_0 draft KV cache) |
| `JBDRAFT{Tiny,Small,Normal,Large,Big}_...GSQ-RCO-IQ3_XXS` | 32K / 64K / 96K / 128K / 162K | DFlash2 (`draft-dflash`, DFlash2 draft model, n-max 4) |

The `JBDRAFT*` profiles are the configurations used in the [DFlash2 speculative decoding report](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md) (`JBDRAFTBig` is the verbatim config quoted there).

## One-shot generation experiments

### `games/`
Single-page HTML games generated in one shot with Qwen 3.8 27B — a '90s neon aesthetic throughout:

- `neonpong.html` — Pong
- `neontetris.html` — Tetris
- `neonbreakout.html` — Breakout
- `neonsnake.html` — Snake

See [games/README.md](games/README.md) for the exact prompts and server flags used.

### `others/`
Single-page HTML experiments:

- `miniraytracer.html` — a software raytracer (no external API) rendering a reflective sphere, checkerboard ground, and clouds — a nod to early 1990s raytracing
- `pagodascene.html` — a Three.js scene: orbital camera around a pagoda at daytime, green ground, trees

See [others/README.md](others/README.md) for the prompts used.
