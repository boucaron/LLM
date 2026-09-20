# LLM — Local LLM Experiments

A collection of practical local LLM experiments: benchmarking GGUF models with `llama.cpp`, reusable benchmark scripts, server configurations, one-shot HTML generation experiments, and a practical study of context engineering for LLM agents.

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
├── context-engineering-practical-experiments.md
├── context-engineering-primitives.md
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

> Note: my daily driver on the Mac is [Tiel-Coder-35B-A3B-UD-IQ3-XXS](macbook-air-m4-local-llm-benchmark.md#tiel-coder-35b-a3b-gguf), a coding-tuned model running at ~26 t/s with a 120K context. It is a re-quantization of Ornith 1.5 35B A3B (Unsloth dynamic imatrix) with improvements for coding work and multi-turn conversations. It really pushes the memory envelope of the 24 GB MacBook Air — you need to be very lean on the remaining memory to run it.

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

## Context engineering

### [Context Engineering: Practical Experiments](context-engineering-practical-experiments.md)
A practical study of context engineering for LLMs and agent workflows: why context is a scarce resource, and the techniques for keeping it small and relevant rather than relying on compaction alone.

Key ideas covered:

- **Context as a working set** — a context should contain only what is needed *now*, not everything that might be needed later
- **Selection / filtering** — relevance filtering, deduplication, removing stale or superseded information *before* it enters the context
- **Progressive disclosure** — expose information only when the agent reaches the point where it needs it; prefer pointers to information over the information itself
- **Compression vs. representation** — summarization, extraction, structuring, compilation, and canonicalization as distinct transformations; the best representation depends on the operation that will consume it
- **Forking and context boundaries** — fork conversations to keep explorations and subtasks out of the parent context; boundaries act as both efficiency and correctness boundaries
- **Task specification, isolated execution and merge** — compile a compact task spec, run each work item in a fresh context, and merge only outcomes back into the parent (scale: parent context stays constant regardless of the number of work items)
- **Result passing instead of history passing** — hand over validated outputs and state, not full conversations
- **Information lifecycle** — treat context items as stateful (created → active → superseded → archived), with supersession replacing stale assumptions
- **Agent-system techniques** — externalized state, hierarchical context (progressive refinement), validation before promoting observations to the parent context, and **task checkpointing / state handoff** as a practical alternative to context compaction for long-running agents

### [Context Engineering Primitives (early draft)](context-engineering-primitives.md)
An early draft in which I explore how the LLM and the harness can be given a small set of
context primitives — `context.push()`, `context.mark()` / `context.deleteMarked()`,
and `context.call()` — such that the working context becomes an explicit, mutable
**projection** of an immutable session history, rather than a raw contiguous slice of the
conversation. This enables the agent to keep the context as small as it needs to be and
more relevant to the current computation — removing temporary work (e.g. debugging
trajectories) from the active context while retaining it in history — without ever
destructively modifying the session.

The draft covers the core model (immutable history + materialized projection, analogous to
an append-only event log with a materialized view), the minimal primitive set
(`push` / `mark`–`deleteMarked` / `call`), treating `call()` as a function whose argument
is context, recursion limits (depth and call budgets), and exposing the primitives to the
LLM as ordinary tool calls — with a note on how the projection must respect
message/tool-call/tool-result structure to stay provider-compatible.

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

## License

Released under the [0BSD (Zero-clause BSD) license](LICENSE): use, copy, modify, and/or distribute for any purpose, with no conditions and no warranty.

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
