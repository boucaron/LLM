# LLM — Local LLM Experiments

A collection of practical local LLM experiments: benchmarking GGUF models with `llama.cpp`, reusable benchmark scripts, server configurations, and one-shot HTML generation experiments.

The focus is on **practical usability** of local models (inference speed, memory/VRAM usage, thermal behavior) rather than model quality comparisons.

## Repository layout

```
.
├── macbook-air-m4-local-llm-benchmark-july-2026.md
├── rtx5060ti-local-llm-benchmark-july-2026.md
├── llamacpp/
│   └── model.ini
├── scripts/
├── games/
└── others/
```

## Benchmark reports

### [MacBook Air M4 — 24 GB](macbook-air-m4-local-llm-benchmark-july-2026.md)
Practical local LLM performance on a fanless MacBook Air M4 (24 GB unified memory).
Tests of recent GGUF models (Gemma 4 26B A4B, Qwen 3.x, LFM, Prism Bonsai, Ornith, GLM…) under sustained operation, with attention to thermal throttling, memory usage, and quantized KV cache.

### [RTX 5060 Ti — 16 GB](rtx5060ti-local-llm-benchmark-july-2026.md)
Practical local LLM performance on a dedicated 16 GB GPU.
Explores aggressive 2-bit quantization, MTP speculative decoding, very large context windows (100K–256K), and CPU offloading of MoE layers. Highlights include ~80–105 t/s for 25–35B MoE models fully in VRAM and ~130+ t/s with MTP.

> Note: this is an extensive, detailed report (~3,000 lines). The summary above captures the key findings; the full report contains the complete methodology, per-configuration measurements, and analysis.
>
> Last updated **01/09/2026** — most recent additions: Qwen 3.8 27B in `IQ3_XXS GSQ RCO` (with MTP), Qwen 3.8 27B with Unsloth Dynamic Quant v3.0 (`UD-IQ3_S`), and Ornith 1.5 35B A3B.

## Benchmark scripts

### `scripts/`
Self-contained bash benchmark scripts, one per model. Each script:

1. Locates its `llama-server` binary and model GGUF (expected in the parent directory, e.g. `../<model>.gguf`)
2. Starts a `llama-server` instance with model-specific flags
3. Runs a set of prompts and measures generation throughput, memory usage, and swap/pressure
4. Writes results to `scripts/results/`:
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

## Llama.cpp configuration

### `llamacpp/model.ini`
`llama-server` configuration profiles (version 1) for Qwen 3.8 27B variants (UD-IQ3_S and GSQ-RCO-IQ3_XXS), covering large context sizes (120K/240K), flash attention, q4_0 KV cache, MTP speculative decoding, and reasoning effort settings.

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
