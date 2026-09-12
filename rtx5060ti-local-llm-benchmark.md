# Practical Local LLM Performance on an RTX 5060 Ti 16GB

Tests conducted in July/August/September 2026.

## Introduction

Following my earlier experiments on the MacBook Air M4, I finally decided to add a dedicated GPU to my workstation.

This report is not intended to compare model quality or reasoning capabilities. The goal is simply to determine which GGUF models are practical to run on an RTX 5060 Ti 16 GB in terms of inference speed, VRAM usage, and overall usability.

The focus is on generation throughput. Prompt processing (prefill) was intentionally not measured separately as part of the primary benchmark.

The main question is:

> **How much LLM can you actually squeeze into a 16 GB GPU, and what trade-offs are required to do it?**

**TL;DR:** A 16 GB GPU runs several 25–35B-class MoE models at highly interactive speeds: **80–105 t/s** with aggressive 2-bit quantization (model fully in VRAM), **~137 t/s** with MTP (Qwen 3.6 35B A3B, 12.5 GB), and **100K–256K context** by keeping the KV cache quantized (q4_0) in VRAM and offloading MoE layers to CPU. Dense models of comparable size are 3–5× slower. Detailed per-configuration numbers are in the companion files.

The experiments started with very aggressive 2-bit quantization, then moved through different quantization levels, MTP, very large context sizes, and finally CPU offloading of MoE layers.

The results are surprisingly practical: a 16 GB GPU can run several 25–35B-class sparse models at highly interactive speeds, and with the right combination of quantization, KV-cache settings and CPU offloading, very large context sizes are possible as well.

---

# 1. What Can You Squeeze Out of 16 GB?

The short answer is: **quite a lot**.

With aggressive 2-bit quantization and a Q4 KV cache, the RTX 5060 Ti can run several 25–35B-class MoE models entirely on the GPU at roughly 80–105 t/s.

MTP can push some configurations significantly higher, with the Qwen3.6-35B-A3B reaching around 137 t/s in the tested configuration.

Even models that exceed the available VRAM at higher-quality quantization levels remain usable by offloading some MoE layers to CPU memory.

## Quick benchmark summary

*All entries use Q4 KV-cache quantization. Model quantization levels range from 2-bit to 4-bit depending on the configuration. For CPU-offloaded MoE models, VRAM reflects steady-state usage with MoE layers offloaded. The "headline" configuration is shown for each model. Avg t/s is the mean of three runs at 32K context; the larger-context rows show the start → end-of-context range instead. Star ratings are a subjective usability judgment on this system (5 = excellent daily driver, 3 = usable).*


| Model                                    |   Context   |      Avg t/s |         VRAM | Practical on 16 GB? |
| ---------------------------------------- | :---------: | -----------: | -----------: | :-----------------: |
| Gemma 4 26B A4B IQ2\_XXS                  |      32K    |          \~103 |      10.9 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 27B IQ2\_XXS                    |      32K    |             \~34 |      10.4 GB |     ⭐⭐⭐☆☆     |
| Qwen 3.6 27B MTP IQ2\_XXS (n-max 2)      |      32K    |             \~49 |      11.0 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.6 27B Q3\_K\_M MTP 2              |      32K    |             \~41 |      14.8 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3 Coder 30B A3B Instruct IQ2\_XXS   |      32K    |             \~82 |      11.5 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 35B A3B Instruct IQ2\_XXS       |      32K    |             \~86 |      11.2 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 35B A3B Instruct MTP IQ2\_XXS (n-max 2) | 32K | \~137 | 12.5 GB | ⭐⭐⭐⭐⭐ |
| Qwen 3.6 35B A3B IQ4\_NL (16 MoE off)    |      32K    |             \~59 |      12.5 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.6 35B A3B IQ3\_S MTP 2            |      32K    |            \~125 |      15.6 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.8 27B Q3\_K\_M MTP 2              |      32K    |        \~35–42 |      14.7 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.8 27B Q2\_K\_XL MTP 1             |     172K    |        \~39–42 |      15.3 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.8 27B IQ2\_XXS MTP 2              |     192K    |        \~42–57 |      14.7 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.8 27B D3 IQ3\_S MTP 2             |      32K    |        \~40–44 |      13.4 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.8 27B ISTA IQ3\_XXS MTP 2         |     170K    |        \~40–43 |      15.6 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.8 27B ISTA IQ3\_XXS DFlash2 (n-max 4) | 32K–64K | \~60–65 | 13.3–15.1 GB | ⭐⭐⭐⭐⭐ |
| Ornith 1.0 35B IQ2\_XXS                  |      32K    |            \~104 |      11.8 GB |     ⭐⭐⭐⭐⭐     |
| Ornith 1.0 9B                            |      32K    |             \~69 |       6.8 GB |     ⭐⭐⭐⭐☆     |
| Ornith 1.0 9B MTP (n-max 2)              |      32K    |             \~92 |       7.4 GB |     ⭐⭐⭐⭐⭐     |
| Ornith 1.5 35B A3B Q4\_K\_M              |      32K    |             \~72 |      15.4 GB |     ⭐⭐⭐⭐⭐     |
| Ornith 1.5 35B A3B AD-IQ4\_XS-IQ3\_S     |      32K    |             \~85 |      15.6 GB |     ⭐⭐⭐⭐⭐     |
| KAT-Coder-V2.5-Dev IQ2\_XXS              |      32K    |             \~87 |      10.5 GB |     ⭐⭐⭐⭐⭐     |
| Qwythos-9B-Claude-Mythos-5-1M MTP         |      32K    |             \~80 |       7.6 GB |     ⭐⭐⭐⭐⭐     |
| Muse-Glimmer-30B Q3\_K\_XL                |      32K    |             \~27 |      12.6 GB |     ⭐⭐⭐☆☆     |
| NVIDIA Nemotron 3.5 Lightning 30B A3B IQ4\_XS | 32K | \~90 | 15.4 GB | ⭐⭐⭐⭐⭐ |

## Practical configurations

The experiments suggest a few useful configurations depending on what matters most.

### Maximum throughput

For a 25–35B-class sparse model, aggressive 2-bit quantization with the model fully resident in VRAM is extremely effective.

MTP can provide another large increase in throughput when the model supports it.

The Qwen3.6-35B-A3B MTP configuration reached approximately **130–142 t/s** in individual runs, with around 12.5 GB of VRAM usage.

### Large context

If context size is more important than maximum throughput, the key is to keep the KV cache in VRAM.

Once the KV cache starts spilling into CPU memory, throughput can collapse.

The Qwen3.6-27B Q3\_K\_S experiment demonstrated that slightly reducing model size can be worthwhile because the saved VRAM can instead be used for the KV cache. It remained GPU-resident at approximately 100K context and was still generating at around 27 t/s near the end of the context.

### Very large context on a larger model

For the Qwen3.6-35B-A3B IQ3\_S configuration, 256K context was initially unusable because the KV cache was being offloaded to CPU memory.

However, offloading only a few MoE layers to CPU memory freed enough VRAM to keep the KV cache resident.

With 4 MoE layers offloaded, the model reached approximately **88–89 t/s** at 256K context without MTP.

With MTP 1 and 8 MoE layers offloaded, it reached approximately **82–86 t/s**.

This is one of the most interesting results of the experiments: **a small amount of model offloading can be preferable to allowing the KV cache to spill to CPU memory.**

---

# 2. The Main Trade-offs

The experiments show that VRAM is not simply a matter of fitting the model weights.

Several things compete for the same 16 GB:

* Model weights
* MTP overhead
* KV cache
* Context size
* Runtime overhead
* GPU-resident layers

For sparse MoE models, CPU offloading is particularly interesting because it is possible to move some MoE layers out of VRAM while leaving the rest of the model and the KV cache on the GPU.

This leads to an important practical rule:

> **When VRAM is tight, it can be better to offload a few MoE layers to CPU memory than to let the KV cache spill to CPU memory.**

The experiments repeatedly show that once the KV cache is forced into system memory, throughput can drop dramatically.

This effect is particularly visible with very large context sizes.

---

# 3. Why Buy an RTX 5060 Ti?

Until recently I never installed a dedicated GPU in my workstation because most of my workloads were CPU-bound.

The recent generation of sparse Mixture-of-Experts models has changed that. There are now several capable LLMs that comfortably fit within 16 GB of VRAM.

The RTX 5060 Ti provides over 400 GB/s of memory bandwidth, compared with roughly 120 GB/s for the MacBook Air M4.

The substantially higher memory bandwidth suggested that decode throughput should improve significantly, although actual performance would also depend on model architecture, quantization, llama.cpp kernels and whether inference remained entirely on the GPU.

From a practical point of view, this moves local LLMs from small experiments to tools that can be used productively every day. Running coding assistants and autonomous agents locally becomes genuinely feasible.

---

# 4. Test Setup

## Hardware

* **Motherboard:** MinisForum AMD Ryzen 9 7945HX BD795M
* **Memory:** Corsair CMSX64GX5M2A5200C44 (2 × 32 GB)
* **SSD:** Lexar SSD NQ790 2 TB
* **GPU:** ASUS Prime GeForce RTX 5060 Ti 16 GB GDDR7 OC Edition
* **PSU:** Seasonic Core GX-650 V2

The CPU was configured with a conservative power limit of approximately 75 W peak. I describe the rationale and configuration in a separate note.

During testing, the CPU was never the limiting factor.

## Software

* Windows 11 25H2
* NVIDIA Studio Driver 610.62 (CUDA 13)
  * **Important**: The GPU is deliberately kept quiet and is thermally limited to around 60 °C, so the GPU will throttle rather aggressively once it reaches this temperature. These results therefore prioritize a quiet, sustained local-LLM experience rather than maximum possible throughput. Linux, Game Ready drivers, higher power limits, or a higher thermal target should produce higher numbers.
* llama.cpp b10069 (CUDA 13.3)
* llama-b10360 (CUDA 13.3) starting from Muse Gleemer
* llama-b10472 (CUDA 13.3) starting from Qwen 3.8 27B
* llama-b10658 - release *0.4.0* (CUDA 13.3) starting from DFlash 2 experiments

## Methodology

Each model was given the same three prompts. Models were tested under similar interactive settings, with model-specific parameters adjusted when required for correct chat behavior.

The reported token count is the number of output tokens actually generated by the model before it stopped naturally.

Since different models produce different response lengths—especially reasoning models—the generation time is not directly comparable across models.

The primary metric of interest is sustained generation throughput (t/second), while the output token counts illustrate how verbose each model is.

Unless otherwise noted, the tests use a 4-bit KV cache together with Flash Attention to maximise the available context while keeping VRAM usage low.

### Prompts

* Prompt 1: I think that the 42 answer is also a sarcastic way to what question matters and the importance of a good question. What do you think
* Prompt 2: Can you enumerate such questions
* Prompt 3: I would say the usage of a llm is a bit like asking such kind of question

---

# 5. MTP: A Surprisingly Effective Lever

The RTX 5060 Ti responds well to MTP with `spec-draft-n-max 2`.

MTP increased throughput by approximately 44% on the 27B dense model and 60% on the 35B A3B model in the initial tests.

This is a major improvement:

* For the dense 27B model, the user experience moves from experimental to the low comfort zone.
* For the sparse 35B model, the throughput is high enough that agentic jobs become possible locally.

The throughput increase is good enough to start thinking about tuning the model in other directions: using less aggressive quantization to improve model quality, or using more precision in the KV cache.

I hope we will also see more smaller dense models with MTP.

However, MTP is not universally beneficial. When CPU offloading becomes the dominant bottleneck, MTP provides only a modest improvement. Some small models also show little or no gain.

### DFlash 2: A Separate Draft Model

A different speculative-decoding approach is available through **DFlash 2**, which uses a separate small draft model (here: `Qwen 3.8 27B DFlash2 Q2_K_S MIX`) running alongside the main model.

On the Qwen 3.8 27B ISTA `IQ3_XXS` configuration, DFlash 2 with `spec-draft-n-max 4` reaches approximately **60–65 t/s** at 32K–64K context, compared to roughly **40–43 t/s** with MTP 2 on the same main model — a gain of around 40–50%.

Throughput erodes as context grows, in line with the KV-cache pressure discussed elsewhere: around 40 t/s at 128K, dropping to ~23 t/s at 162K where the combined dedicated + shared GPU memory footprint (~16.7 GB) exceeds the 16 GB budget.

See the dedicated report: [rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md).

---

# 6. Dense Models: 27B Is Possible, But VRAM Matters

The RTX 5060 Ti 16 GB is surprisingly capable of local 27B-class inference.

For dense models, however, the decisive factor is not simply whether the model can be loaded. Keeping both model weights and KV cache in VRAM is critical.

Fully GPU-resident Q3 configurations can reach around 40 t/s at 32K context and remain usable beyond 100K context, whereas crossing into CPU memory can reduce throughput to single-digit tokens per second.

This makes dense models particularly sensitive to the exact VRAM balance.

Recent improvements in speculative drafters have changed this picture for dense models as well. With **DFlash 2** speculative decoding, the Qwen 3.8 27B dense model now reaches around **60 t/s** (ISTA `IQ3_XXS` main model with a DFlash2 draft and `spec-draft-n-max 4`, ~15.1 GB VRAM), compared with roughly **40–43 t/s** using MTP 2 on the same model — a gain of about 40–50%. Dense 27B models are therefore no longer capped at the ~40 t/s level, as long as the combined draft-model and KV-cache footprint stays within the 16 GB budget.

---

# 7. Quantization Improvements: Non-Uniform Quantization Has Gotten Good

This document is about throughput and usability, not model quality, but the quality of low-bit quantization is important enough to mention, because it directly changes which configuration is worth running.

The capabilities of smaller quantizations have slightly increased in recent months. The main driver is **non-uniform quantization**: instead of applying a single quantization level to the entire model, each tensor or layer receives its own level, with the more sensitive parts kept at higher precision. The result is that a 2–3 bit model now retains noticeably more of the original model quality than a uniformly quantized model of the same nominal size.

Two sources in particular provide such models, both tested here:

* **Unsloth Dynamic Quant V3.0** — for example `Qwen3.8-27B-UD-IQ3_S`
* **ISTA-DASLab (GSQ-RCO)** — for example `Qwen3.8-27B-GSQ-RCO-IQ3_XXS` and its MTP variant

This is also why I ran experiments with **4-bit models plus CPU offloading** (for instance `Qwen3.6-35B-A3B IQ4_NL` with 16 MoE layers offloaded, and the `Qwen3.8-27B` IQ4 experiments): rather than accepting a heavy quality penalty to fit a model entirely in 16 GB, a higher-quality non-uniform quantization combined with a small amount of MoE offloading is often the better trade-off.

In practice, these non-uniformly quantized models have become my daily drivers: the quality gain is real enough to matter day to day, while the throughput cost is small as long as the KV cache stays in VRAM.

---

# 8. Experimental Results

The detailed experiments and measurements behind the practical recommendations above are in three companion files:

* [rtx5060ti-experimental-results.md](rtx5060ti-experimental-results.md) — Qwen 3.6 generation, other models, CPU offloading and reproducibility
* [rtx5060ti-qwen3.8-27b-experiments.md](rtx5060ti-qwen3.8-27b-experiments.md) — Qwen 3.8 27B (D2, D3, ISTA DASLab)
* [rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md) — Qwen 3.8 27B with DFlash2 speculative decoding (32K–162K)

---

# 9. Models and Files Tested

**Unless otherwise noted, GGUF models were downloaded from Unsloth.ai**.

## Gemma 4 26B A4B

* `gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf`

## Qwen 3.6 27B

* `Qwen3.6-27B-UD-IQ2_XXS.gguf`

## Qwen 3 Coder 30B A3B Instruct

* `Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf`

## Qwen 3.6 35B A3B

* `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`

## Qwen 3.6 27B MTP

* `Qwen3.6-27B-UD-IQ2_XXS.gguf`

The MTP variant uses the same filename, with a small difference in the context.

## Ornith 1.0 35B

* `Ornith-1.0-35B-UD-IQ2_XXS.gguf`

## Ornith 1.0 9B

* `Ornith-1.0-9B-Q4_K_M.gguf`

## KAT-Coder-V2.5-Dev

* `Kwaipilot_KAT-Coder-V2.5-Dev-IQ2_XXS.gguf`

The bartowski-provided model was used.

## Qwythos-9B-Claude-Mythos-5-1M

* `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf`

The model was provided by empero-ai.

## Muse-Glimmer 30B

* `Muse-Glimmer-30B-UD-Q3_K_XL.gguf`

## NVIDIA-Nemotron-3.5-Lightning-30B-A3B

* `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS`

## Qwen 3.8 27B

Dynamic Quant V2.0

* `Qwen3.8-27B-IQ4_NL`
* `Qwen3.8-27B-Q3_K_M`
* `Qwen3.8-27B-Q3_K_S`
* `Qwen3.8-27B-UD-Q2_K_XL`
* `Qwen3.8-27B-UD-IQ2_XXS`

Dynamic Quant V3.0

- `Qwen3.8-27B-UD-IQ3_S`

Other variant ISTA-DASLab

- `Qwen3.8-27B-GSQ-RCO-IQ3_XXS`
- `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp`

## Ornith 1.5 35B A3B

- `ornith-ai/Ornith-1.5-35B-Q4_K_M`
- `AtomicChat/Ornith-1.5-35B-A3B-GGUF:AD-IQ4_XS-IQ3_S`

# 10. Practical Recommendations

After all these experiments, a few rules stand out.

### 1. Keep the KV cache in VRAM if possible

This is probably the most important rule from the experiments.

A small amount of model offloading can be acceptable.

KV-cache offloading can be devastating to throughput.

### 2. For MoE models, offload MoE layers rather than the whole model

MoE models behave much better under CPU offloading than dense models.

The Qwen3.6-35B-A3B IQ4\_NL experiments show that even with a substantial number of MoE layers on the CPU, around 50–65 t/s is still possible.

### 3. Don't automatically maximise GPU layer count

The goal is not necessarily to put every possible layer on the GPU.

The goal is to find the best balance between:

* GPU-resident model weights
* KV-cache capacity
* CPU-offloaded MoE layers
* MTP overhead

For a large context, leaving a little more VRAM available for the KV cache can produce a much better result.

### 4. MTP is most valuable when the model is GPU-resident

MTP can provide very large throughput improvements when the model is entirely on the GPU.

For the Qwen3.6-35B-A3B IQ2\_XXS configuration, it increased throughput from around 80 t/s to roughly 130–140 t/s.

Once CPU offloading becomes the dominant bottleneck, the benefit becomes much smaller.

### 5. DFlash 2 can provide additional throughput, but adoption is limited

DFlash 2 speculative decoding, which uses a separate small draft model alongside the main model, is an alternative speculative-decoding approach to MTP. In the tested configuration (Qwen 3.8 27B ISTA `IQ3_XXS` with a DFlash2 `Q2_K_S` draft, `n-max 4`), it reached around **60–65 t/s** at 32K–64K context, versus roughly **40–43 t/s** with MTP 2 on the same main model.

A notable advantage over MTP is that DFlash 2 tends to hold its throughput better as context grows, whereas MTP gains erode more quickly once the KV cache starts putting pressure on available VRAM.

However, DFlash 2 is not yet a common or standard feature. Most implementations require a separate draft model — an "addon" model — that must be downloaded and configured alongside the main model, and finding a draft model small enough to fit within a 16 GB budget alongside a 27B–35B main model and a Q4 KV cache is non-trivial.

See the dedicated [DFlash 2 report](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md) for details.

### 6. A slightly smaller quantization can be better than expected

Moving from 4-bit to 3-bit or down to 2-bit is not only about making the model fit.

The additional VRAM headroom can be converted into:

* more context,
* a larger KV cache,
* less CPU offloading,
* or some combination of the three.

The Qwen3.6-35B-A3B 3-bit experiments are a good example of this.

### 7. 16 GB is a very interesting capacity

16 GB is enough to run surprisingly large sparse models at very high generation speeds.

At the same time, it is also exactly where memory management becomes important.

A model that fits in 16 GB at 32K context may no longer behave the same way at 128K or 256K.

---

# 11. Experiment Log

The experiments started from a simple question: how far can a 16 GB GPU be pushed for local LLM inference?

The first experiments focused on very aggressive 2-bit quantization. This made it possible to run 25–35B-class sparse models entirely on the GPU at highly interactive speeds.

That led to the next question: if the GPU is already fast enough, can MTP provide another significant improvement?

Yes, it did.

The Qwen3.6-35B-A3B configuration went from roughly 80 t/s to around 130–140 t/s with MTP, making the model feel extremely responsive.

The next question was whether it was possible to move toward higher-quality quantization.

At 4-bit, the 35B model no longer fits entirely in 16 GB. CPU offloading therefore became necessary.

This worked surprisingly well for the MoE model. Rather than being completely unusable, the model could still generate at around 50–65 t/s depending on how many MoE layers were moved to CPU memory.

The experiments then became more interesting.

The 3-bit Qwen3.6-35B-A3B configuration fits fully in VRAM at smaller context sizes. At 32K context it reaches roughly 103 t/s without MTP and around 126 t/s with MTP 1.

But when increasing the context to 256K, the KV cache becomes too large.

At that point llama.cpp starts moving the KV cache to CPU memory, and throughput collapses to around 22–23 t/s.

The obvious solution was to move some of the model to CPU memory instead.

With only 4 MoE layers offloaded, the 256K configuration reached around 88–89 t/s.

That was one of the more interesting results of the whole experiment.

Instead of thinking about CPU offloading as something that should always be avoided, it became clear that **the location of the offloaded data matters**.

A few MoE layers on the CPU can be much better than having the KV cache on the CPU.

This also explains why the optimal configuration changes with context size.

At small context sizes, putting as much of the model as possible on the GPU is generally the best approach.

At very large context sizes, however, VRAM becomes valuable for the KV cache.

Moving a few MoE layers to CPU memory can therefore free enough VRAM to keep the KV cache on the GPU.

This produces a much better overall result.

---

# 12. Conclusions

The RTX 5060 Ti 16 GB proved to be an excellent entry point for local LLM inference.

Sparse models in the 25–35B class now run at speeds that make interactive usage genuinely comfortable, with slightly above 50 t/s already feeling very usable and MTP pushing some configurations into a much more responsive range.

Compared with the MacBook Air M4, generation throughput improved by roughly three to five times depending on the model.

These comparisons are based on my earlier M4 experiments using comparable model/configuration combinations; they are not a controlled same-day hardware comparison.

Perhaps the biggest surprise was that even the Qwen 3.6 27B reasoning model remained usable at around 34 t/s and around 45–50 t/s with MTP.

While not as fast as the sparse A3B models, it is responsive enough that reasoning no longer feels like a bottleneck.

The experiments also showed that the 16 GB VRAM boundary is not as simple as “model fits” versus “model does not fit”.

For MoE models, carefully chosen CPU offloading can extend what is possible considerably.

The most important lesson is probably that **VRAM should be managed as a budget shared between model weights and KV cache**.

For large-context workloads, it can be better to sacrifice a few MoE layers to CPU memory than to sacrifice the KV cache.

These results suggest that 16 GB GPUs can be a practical sweet spot for local inference, particularly when using aggressively quantized MoE models.

A single mid-range consumer GPU is now sufficient to run several state-of-the-art sparse models at highly interactive speeds.

16 GB is an excellent entry point, but it is also exactly where dense-model inference starts becoming constrained by the boundary between GPU memory and system memory.

---

# ChangeLog

**27/07/2026**

* Initial test

**28/07/2026**

* MTP tests

**30/07/2026**

* Added Ornith-1.0-35B

**01/08/2026**

* Added Ornith-1.0-9B

**02/08/2026**

* Added Ornith-1.0-9B MTP variants
* Added Kwaipilot\_KAT-Coder-V2.5-Dev
* Added Qwythos-9B-Claude-Mythos-5-1M

**03/08/2026**

* Additional experiments with offloading to check throughput effects on MoE models

**04/08/2026**

* Additional experiments with MoE offloading and MTP

**07/08/2026**

* Experiment with offloading on Qwen 3.6 27B Dense and MTP

**10/08/2026**

* Added Qwen 3.6 35B A3B in 3-bit quantization

**11/08/2026**

- Added Muse Gleemer in 3-bit quantization

**12/08/2026**

- Added NVIDIA Nemotron 3.5 Lightning 30B A3B

**15/08/2026**

- Added Qwen 3.8 27B in 4 and 3-bit quantization

**16/08/2026**

- Additional experiments on Qwen 3.8 27B to explore context size alternatives.

**17/08/2026**

- Add MTP tuning results for Qwen 3.8 27B `Qwen3.8-27B-UD-IQ2_XXS`

**19/08/2026**

- Add note regarding Unsloth changes with Dynamic Quant V3 models => some models are used are no more available

**21/08/2026**

- Evaluate new Unsloth Qwen 3.8 27B `Qwen3.8-27B-UD-IQ3_S`

**22/08/2026**

- Added Ornith 1.5 35B A3B `Q4_K_M` and `AtomicChat:AD-IQ4_XS-IQ3_S`

**31/08/2026**

- Add Qwen 3.8 27B variant for `IQ3_XXS GSQ RCO`

**01/09/2026**

- Add Qwen 3.8 27B variant for `IQ3_XXS GSQ RCO` MTP

**11/09/2026**
- Integration DFlash 2 experiments + Update tables
- Updated section 6 (Dense Models) with the latest speculative-drafter figures: Qwen 3.8 27B reaches around 60 t/s with DFlash 2
- Added section on non-uniform quantization improvements (Unsloth Dynamic Quant V3.0, ISTA-DASLab) and 4-bit offloading experiments 
