# Practical Local LLM Performance on an RTX 5060 Ti 16GB

Tests conducted in July/August 2026.

## Introduction

Following my earlier experiments on the MacBook Air M4, I finally decided to add a dedicated GPU to my workstation.

This report is not intended to compare model quality or reasoning capabilities. The goal is simply to determine which GGUF models are practical to run on an RTX 5060 Ti 16 GB in terms of inference speed, VRAM usage, and overall usability.

The focus is on generation throughput. Prompt processing (prefill) was intentionally not measured separately as part of the primary benchmark.

The main question is:

> **How much LLM can you actually squeeze into a 16 GB GPU, and what trade-offs are required to do it?**

The experiments started with very aggressive 2-bit quantization, then moved through different quantization levels, MTP, very large context sizes, and finally CPU offloading of MoE layers.

The results are surprisingly practical: a 16 GB GPU can run several 25–35B-class sparse models at highly interactive speeds, and with the right combination of quantization, KV-cache settings and CPU offloading, very large context sizes are possible as well.

---

# 1. What Can You Squeeze Out of 16 GB?

The short answer is: **quite a lot**.

With aggressive 2-bit quantization and a Q4 KV cache, the RTX 5060 Ti can run several 25–35B-class MoE models entirely on the GPU at roughly 80–105 tokens/s.

MTP can push some configurations significantly higher, with the Qwen3.6-35B-A3B reaching around 137 tokens/s in the tested configuration.

Even models that exceed the available VRAM at higher-quality quantization levels remain usable by offloading some MoE layers to CPU memory.

## Quick benchmark summary

*Numbers below use very aggressive 2-bit quantization and Q4 KV-cache quantization, except for the 9B models, which use 4-bit quantization and their default KV-cache configuration.*


| Model                                  | Avg tok/s |    VRAM | Practical on 16 GB? |
| -------------------------------------- | --------: | ------: | :-----------------: |
| Gemma 4 26B A4B IQ2\_XXS               |      \~90 | 10.9 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 27B IQ2\_XXS                  |      \~30 | 10.4 GB |     ⭐⭐⭐☆☆     |
| Qwen 3 Coder 30B A3B Instruct IQ2\_XXS |      \~75 | 11.5 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 35B A3B Instruct IQ2\_XXS     |      \~80 | 11.2 GB |     ⭐⭐⭐⭐⭐     |
| Qwen 3.6 27B MTP IQ2\_XXS              |      \~45 | 11.0 GB |     ⭐⭐⭐⭐☆     |
| Qwen 3.6 35B A3B Instruct MTP IQ2\_XXS |     \~120 | 12.5 GB |     ⭐⭐⭐⭐⭐     |
| Ornith 1.0 35B IQ2\_XXS                |      \~90 | 11.8 GB |     ⭐⭐⭐⭐⭐     |
| Ornith 1.0 9B                          |      \~60 |  6.8 GB |     ⭐⭐⭐⭐☆     |
| Ornith 1.0 9B MTP                      |      \~80 |  7.4 GB |     ⭐⭐⭐⭐⭐     |
| KAT-Coder-V2.5-Dev IQ2\_XXS            |      \~80 | 10.5 GB |     ⭐⭐⭐⭐⭐     |
| Qwythos-9B-Claude-Mythos-5-1M MTP      |      \~80 |  7.7 GB |     ⭐⭐⭐⭐⭐     |

## Practical configurations

The experiments suggest a few useful configurations depending on what matters most.

### Maximum throughput

For a 25–35B-class sparse model, aggressive 2-bit quantization with the model fully resident in VRAM is extremely effective.

MTP can provide another large increase in throughput when the model supports it.

The Qwen3.6-35B-A3B MTP configuration reached approximately **130–142 tok/s** in individual runs, with around 12.5 GB of VRAM usage.

### Large context

If context size is more important than maximum throughput, the key is to keep the KV cache in VRAM.

Once the KV cache starts spilling into CPU memory, throughput can collapse.

The Qwen3.6-27B Q3\_K\_S experiment demonstrated that slightly reducing model size can be worthwhile because the saved VRAM can instead be used for the KV cache. It remained GPU-resident at approximately 100K context and was still generating at around 27 tok/s near the end of the context.

### Very large context on a larger model

For the Qwen3.6-35B-A3B IQ3\_S configuration, 256K context was initially unusable because the KV cache was being offloaded to CPU memory.

However, offloading only a few MoE layers to CPU memory freed enough VRAM to keep the KV cache resident.

With 4 MoE layers offloaded, the model reached approximately **88–89 tok/s** at 256K context without MTP.

With MTP 1 Way and 8 MoE layers offloaded, it reached approximately **82–86 tok/s**.

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

## Methodology

Each model was given the same three prompts. Models were tested under similar interactive settings, with model-specific parameters adjusted when required for correct chat behavior.

The reported token count is the number of output tokens actually generated by the model before it stopped naturally.

Since different models produce different response lengths—especially reasoning models—the generation time is not directly comparable across models.

The primary metric of interest is sustained generation throughput (tokens/second), while the output token counts illustrate how verbose each model is.

Unless otherwise noted, the tests use a 4-bit KV cache together with Flash Attention to maximise the available context while keeping VRAM usage low.

### Prompts

* Prompt 1: I think that the 42 answer is also a sarcastic way to what question matters and the importance of a good question. What do you think
* Prompt 2: Can you enumerate such questions
* Prompt 3: I would say the usage of a llm is a bit like asking such kind of question

---

# 5. MTP: A Surprisingly Effective Lever

The RTX 5060 Ti responds well to MTP with two-token MTP/speculative decoding.

MTP increased throughput by approximately 44% on the 27B dense model and 60% on the 35B A3B model in the initial tests.

This is a major improvement:

* For the dense 27B model, the user experience moves from experimental to the low comfort zone.
* For the sparse 35B model, the throughput is high enough that agentic jobs become possible locally.

The throughput increase is good enough to start thinking about tuning the model in other directions: using less aggressive quantization to improve model quality, or using more precision in the KV cache.

I hope we will also see more smaller dense models with MTP.

However, MTP is not universally beneficial. When CPU offloading becomes the dominant bottleneck, MTP provides only a modest improvement. Some small models also show little or no gain.

---

# 6. Dense Models: 27B Is Possible, But VRAM Matters

The RTX 5060 Ti 16 GB is surprisingly capable of local 27B-class inference.

For dense models, however, the decisive factor is not simply whether the model can be loaded. Keeping both model weights and KV cache in VRAM is critical.

Fully GPU-resident Q3 configurations can reach around 40 tok/s at 32K context and remain usable beyond 100K context, whereas crossing into CPU memory can reduce throughput to single-digit tokens per second.

This makes dense models particularly sensitive to the exact VRAM balance.

---

# 7. Experimental Results

The remainder of the document contains the detailed experiments and measurements behind the practical recommendations above.

---

## 7.1 Gemma 4 26B A4B

### 4-bit KV Cache, Flash Attention On

```bash
llama-server.exe
-m ..\gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf
--ctx-size 32768
--temp 1.0
--top-p 0.95
--top-k 64
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-ngl 99
```

Prompt 1: Output 522 tokens, 5.1s, 102.45 t/s

Prompt 2: Output 741 tokens, 7.0s, 105.35 t/s

Prompt 3: Output 661 tokens, 6.5s, 101.35 t/s

VRAM used: 10.9 GB

### 4-bit KV Cache, Flash Attention On, Parallel Seq 1

```bash
llama-server.exe
-m ..\gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf
--ctx-size 32768
--temp 1.0
--top-p 0.95
--top-k 64
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-ngl 99
-np 1
```

Prompt 1: Output 569 tokens, 5.1s, 102.33 t/s

Prompt 2: Output 871 tokens, 8.2s, 106.23 t/s

Prompt 3: Output 764 tokens, 7.7s, 98.62 t/s

VRAM used: 10.7 GB

Gemma 4 A4B was the fastest model tested, consistently exceeding 100 tokens/s while remaining below 11 GB of VRAM.

Subjectively, it felt effectively instantaneous during interactive use.

---

## 7.2 Qwen 3.6 27B

```bash
llama-server.exe
    -m ..\Qwen3.6-27B-UD-IQ2_XXS.gguf
    --ctx-size 32768
    --temp 1.0
    --top-p 0.95
    --top-k 20
    --min-p 0.00
    -fa on
    --cache-type-k q4_0
    --cache-type-v q4_0
    --chat-template-kwargs "{\"enable_thinking\":false}"
    -np 1
    -ngl 99
```

Prompt 1: Output 542 tokens, 15s, 34.56 t/s

Prompt 2: Output 822 tokens, 23s, 34.41 t/s

Prompt 3: Output 766 tokens, 22s, 34.23 t/s

VRAM used: 10.4 GB

### Observations

Despite being significantly slower than the sparse models, it still delivers a comfortable interactive experience for reasoning tasks.

For comparison, the same benchmark achieved only about 4–6 tokens/s on the MacBook Air M4.

### MTP Variant

Prompt 1: Output 586 tokens, 12s, 47.42 t/s

Prompt 2: Output 864 tokens, 17s, 50.10 t/s

Prompt 3: Output 876 tokens, 17s, 49.32 t/s

VRAM used: 11.0 GB

This is an interesting boost in performance, around 40% more throughput. This is still an important improvement from a user-experience point of view.

The best results were with 2-way MTP; above that, it does not bring more throughput.

---

## 7.3 Qwen 3.6 27B: 4-Bit and 3-Bit Quantization

The 2-bit models are fast, but there is a slight reduction in their capabilities.

I took the MTP variants from the Qwen3.6-27B-MTP-GGUF repository:

* 4-bit: IQ4\_NL with 16.3 GB
* 3-bit:
  * Q3\_K\_M with 13.8 GB
  * Q3\_K\_S with 12.6 GB

### 4-Bit with Offloading

This dense model has **64 layers**, which is important for the offloading experiment.

Template used for this experiment:

```bash
llama-server.exe
-m ..\Qwen3.6-27B-IQ4_NL.gguf
--ctx-size 32768
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00
-fa on --cache-type-k q4_0 --cache-type-v q4_0
--chat-template-kwargs "{\"enable_thinking\":false}"
-np 1 --spec-type draft-mtp --spec-draft-n-max 2
-ngl LAYERS_TO_KEEP_ON_GPU
```

The last parameter, `-ngl`, is the number of layers we are going to keep on the GPU.

Don't forget that the KV cache tries to stay on the GPU to be faster, but if it is too big, llama.cpp will put it in CPU memory, which can severely reduce inference throughput.


| Layers in GPU | VRAM Usage | Throughput |
| ------------: | ---------: | ---------: |
|            48 |    13.1 GB | \~13 tok/s |
|            51 |    13.7 GB | \~14 tok/s |
|            59 |    15.4 GB | \~23 tok/s |
|            60 |    15.5 GB | \~24 tok/s |

### Observations

First I kept 48 of the 64 layers on the GPU. The VRAM usage was fine, but the throughput was not great.

Then I pushed a little further, with only a small throughput gain.

Once I pushed as much as possible onto the GPU, at the cost of VRAM and context headroom, the difference became much more significant.

Offloading dense models like this one puts a lot of stress on the relatively slow PCIe link and the CPU/memory subsystem.

### 3-Bit Experiments

Following the previous case, where it matters a lot to fit as much as possible of the dense model on the GPU, I took a 3-bit version of the model.

#### Q3\_K\_M

```bash
llama-server.exe
-m ..\Qwen3.6-27B-Q3_K_M.gguf
--ctx-size 32768
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00
-fa on --cache-type-k q4_0 --cache-type-v q4_0
--chat-template-kwargs "{\"enable_thinking\":false}"
-np 1 --spec-type draft-mtp --spec-draft-n-max 2
-ngl 99
```

I performed three runs: one without MTP, one with MTP set to 1, and one with MTP set to 2.


|  MTP | VRAM Used | Throughput |
| ---: | --------: | ---------: |
| None |   14.1 GB | \~25 tok/s |
|    1 |   14.5 GB | \~36 tok/s |
|    2 |   14.8 GB | \~41 tok/s |

The model fits in VRAM, and there is a major gain in throughput.

MTP-2 produced a very large improvement on the fully GPU-resident Q3\_K\_M configuration, raising throughput from roughly 25 to over 40 tok/s.

It is an important gain to use MTP in this context, with some cost in VRAM.

The context is limited for this variant of the 3-bit model. I managed to put above 75K tokens in context, but above a threshold the KV cache is offloaded to CPU RAM and performance degrades a lot, to around 9 tok/s.

#### Q3\_K\_S

```bash
llama-server.exe
-m ..\Qwen3.6-27B-Q3_K_S.gguf
--ctx-size 110000
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--chat-template-kwargs "{\"enable_thinking\":false}" -np 1 -ngl 99
--spec-type draft-mtp --spec-draft-n-max 2
```

Here the experiment is to check how much context we can put on the GPU without killing throughput.


| Context |       VRAM Used |                      Throughput |
| ------: | --------------: | ------------------------------: |
|     32K | 13.5 → 13.7 GB |  Start:\~40 → @28K: \~36 tok/s |
|     64K | 14.4 → 14.6 GB |  Start:\~40 → @52K: \~31 tok/s |
|     96K | 15.5 → 15.5 GB |  Start:\~40 → @77K: \~27 tok/s |
|    128K | 15.6 → 15.6 GB | Start:\~37 → @111K: \~19 tok/s |
|    110K | 15.5 → 15.5 GB |  Start:\~40 → @77K: \~27 tok/s |

The smaller Q3\_K\_S variant can maintain full GPU residency beyond 100K context, although generation throughput progressively decreases as the KV cache grows.

The smaller Q3\_K\_S variant leaves enough VRAM headroom for a very large Q4 KV cache. In testing, it remained GPU-resident at approximately 100K context and still generated at 27 tok/s near the end of the context.

This demonstrates that reducing model size slightly can be more valuable than expected when the saved VRAM is converted into KV-cache capacity.

---

## 7.4 Qwen 3 Coder 30B A3B Instruct

```bash
llama-server.exe
-m ..\Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf
--jinja
-ngl 99
--ctx-size 32768
--temp 0.7
--min-p 0.0
--top-p 0.80
--top-k 20
--repeat-penalty 1.05
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
```

Prompt 1: Output 264 tokens, 3.1s, 83.84 t/s

Prompt 2: Output 279 tokens, 3.4s, 81.49 t/s

Prompt 3: Output 296 tokens, 3.7s, 79.78 t/s

VRAM used: 11.5 GB

The coding model maintained around 80 tokens/s while producing relatively concise responses, making the measured generation speed particularly suitable for interactive coding workloads.

---

## 7.5 Qwen 3.6 35B A3B

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
-ngl 99
--ctx-size 32768
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
```

Prompt 1: Output 495 tokens, 5.5s, 90.08 t/s

Prompt 2: Output 1224 tokens, 14s, 82.96 t/s

Prompt 3: Output 685 tokens, 8.1s, 84.46 t/s

VRAM used: 11.2 GB

This model offers an excellent compromise between model size and generation speed in this configuration.

Despite its larger size, it consistently maintained over 80 tokens/s while remaining comfortably within the 16 GB VRAM budget.

### MTP Variant

Prompt 1: Output 729 tokens, 5.6s, 130.87 t/s

Prompt 2: Output 721 tokens, 5.1s, 142.11 t/s

Prompt 3: Output 778 tokens, 5.7s, 137.68 t/s

VRAM used: 12.5 GB

This is a big improvement in throughput. Without MTP it is already very fast, but there is a good 1 GB increase in VRAM usage due to 2-way MTP.

The throughput increase is around 40–60%, which is significant.

There is no further improvement in throughput beyond two draft tokens.

---

# 8. Qwen 3.6 35B A3B: Higher-Quality 4-Bit Quantization

The IQ4\_NL model is about 18 GB without any KV cache, so it cannot fit entirely in the 16 GB GPU.

This makes it a good candidate for testing MoE CPU offloading.

Model:

Qwen3.6-35B-A3B-UD-IQ4\_NL from Unsloth.

## Offloading 16 MoE Layers

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--n-cpu-moe 16
```

Prompt 1: Output 486 tokens, 8.2s, 59.49 t/s

Prompt 2: Output 702 tokens, 5.1s, 59.75 t/s

Prompt 3: Output 784 tokens, 13s, 59.56 t/s

VRAM used: 12.5 GB

## Offloading 12 MoE Layers

Prompt 1: Output 653 tokens, 9.9s, 66.24 t/s

Prompt 2: Output 1090 tokens, 16s, 65.77 t/s

Prompt 3: Output 924 tokens, 14s, 65.60 t/s

VRAM used: 14 GB

## Offloading 20 MoE Layers

Prompt 1: Output 487 tokens, 8.9s, 54.69 t/s

Prompt 2: Output 874 tokens, 15s, 55.02 t/s

Prompt 3: Output 696 tokens, 12s, 55.13 t/s

VRAM used: 11.2 GB

## Offloading 24 MoE Layers

Prompt 1: Output 754 tokens, 15s, 49.84 t/s

Prompt 2: Output 963 tokens, 19s, 50.48 t/s

Prompt 3: Output 1031 tokens, 20s, 50.80 t/s

VRAM used: 9.7 GB

### Observations

Offloading 16 layers reaches about 59 tokens/s. This is not extremely fast, but the user experience is still very good.

Offloading 12 layers gives slightly higher throughput, around 65 tokens/s, but VRAM usage is already 14 GB, so there is not much margin for a small increase in throughput.

Offloading 20 layers gives around 55 tokens/s. The user experience is still good and only 11.2 GB of VRAM is used, making it an interesting configuration.

Offloading 24 layers reduces throughput to around 50 tokens/s. The user experience is still acceptable, and the 9.7 GB VRAM usage leaves a lot of headroom for experimentation.

The results suggest that PCIe transfers and/or CPU memory bandwidth become important bottlenecks once a significant portion of the MoE computation is moved to the CPU.

---

# 9. Qwen 3.6 35B A3B: CPU Offloading + MTP

I then took the MTP variant of the previous model.

## Offloading 16 MoE Layers — 1 Way MTP

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL_MTP.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--spec-type draft-mtp --spec-draft-n-max 1
--n-cpu-moe 16
```

Prompt 1: Output 689 tokens, 10s, 62.70 t/s

Prompt 2: Output 966 tokens, 14s, 64.96 t/s

Prompt 3: Output 762 tokens, 11s, 64.94 t/s

VRAM used: 13.3 GB

## Offloading 16 MoE Layers — 2 Way MTP

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL_MTP.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--spec-type draft-mtp --spec-draft-n-max 2
--n-cpu-moe 16
```

Prompt 1: Output 446 tokens, 6.7s, 66.10 t/s

Prompt 2: Output 908 tokens, 13s, 65.77 t/s

Prompt 3: Output 781 tokens, 12s, 64.38 t/s

VRAM used: 13.3 GB

Small improvement in throughput with additional memory usage.

## Offloading 16 MoE Layers — 2 Way MTP + Q4 KV Cache

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL_MTP.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--spec-type draft-mtp --spec-draft-n-max 2
--n-cpu-moe 16
--cache-type-k q4_0
--cache-type-v q4_0
```

Prompt 1: Output 514 tokens, 8.4s, 61.27 t/s

Prompt 2: Output 933 tokens, 13s, 68.50 t/s

Prompt 3: Output 1044 tokens, 15s, 65.90 t/s

VRAM used: 12.9 GB

Similar throughput with larger context headroom.

### Observations

MTP in this case improved throughput by about 10% with around 800–900 MB of additional VRAM.

From a user-experience point of view it is noticeable, but not a big deal.

When CPU offloading is already the dominant constraint, MTP no longer provides the huge gains seen when the model is entirely GPU-resident.

---

# 10. CPU Offloading and Context Size

The next experiment was to see how CPU offloading behaves as context size increases.

## 24K Context

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --n-cpu-moe 16
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 24608 tokens, 28s, 859.18 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 2,155 tokens, 40s, 52.67 t/s

## 48K Context

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 65536
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --n-cpu-moe 16
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 49184 tokens, 58s, 843.86 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 136 tokens, 2.9s, 47.46 t/s

## 96K Context

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 131072
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --n-cpu-moe 16
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 97825 tokens, 2min 2s, 799.61 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 97 tokens, 2.4s, 39.72 t/s

## 200K Context

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 262144
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --n-cpu-moe 16
--cache-type-k q4_0 --cache-type-v q4_0
```

This experiment measures runtime behavior at large context sizes, not the model's ability to retrieve or reason over information located 200K tokens into the context.

Prefill: 195105 tokens, 4min 34s, 711.45 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 7,230 tokens, 4min 8s, 29.06 t/s

VRAM used: 14.2 GB at the end of the inference.

### 200K Context — 15 MoE Layers Offloaded

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 262144
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --n-cpu-moe 15
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 195105 tokens, 4min 28s, 727.78 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 204 tokens, 7.0s, 29.19 t/s

VRAM used: 14.3 GB at the end of the inference.

The main observation is that very large context is possible, but throughput progressively decreases as the KV cache grows.

---

# 11. Qwen 3.6 35B A3B: 3-Bit Quantization

This is the most interesting combination for squeezing the 35B A3B model into a 16 GB GPU.

I used the Qwen3.6-35B-A3B-UD-IQ3\_S model from Unsloth.

The 3-bit version fits fully on the GPU at smaller context sizes, leaving more VRAM available for the KV cache than the 4-bit version.

## No MTP

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 32768
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
```

Prompt 1: Output 690 tokens, 6.6s, 104.72 t/s

Prompt 2: Output 933 tokens, 9.0s, 103.95 t/s

Prompt 3: Output 940 tokens, 9.1s, 102.96 t/s

VRAM used: 15.4 GB

### 128K Context

Prompt 1: Output 619 tokens, 6.2s, 99.04 t/s

Prompt 2: Output 896 tokens, 8.8s, 98.74 t/s

Prompt 3: Output 685 tokens, 7.0s, 98.18 t/s

### 256K Context

Prompt 1: Output 565 tokens, 24s, 23.52 t/s

Prompt 2: Output 704 tokens, 30s, 23.06 t/s

Prompt 3: Output 766 tokens, 34s, 22.19 t/s

At 256K context there is major offloading of the KV cache to CPU memory, which kills performance.

---

## 11.1 256K Context — Offloading MoE Layers

The interesting question is whether we can recover the lost throughput by moving a small number of MoE layers to CPU memory instead of allowing the KV cache to spill into CPU memory.

### 8 MoE Layers Offloaded

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 262144
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
--n-cpu-moe 8
```

Prompt 1: Output 602 tokens, 7.6s, 78.79 t/s

Prompt 2: Output 983 tokens, 12s, 80.07 t/s

Prompt 3: Output 798 tokens, 10.0s, 80.01 t/s

VRAM used: 14.7 GB

This means we can either increase the context size on the GPU or reduce the offloading to the CPU to increase speed.

### 4 MoE Layers Offloaded

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 262144
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
--n-cpu-moe 4
```

Prompt 1: Output 637 tokens, 7.2s, 88.38 t/s

Prompt 2: Output 793 tokens, 8.8s, 89.62 t/s

Prompt 3: Output 688 tokens, 7.7s, 89.16 t/s

VRAM used: 15.4 GB

This gives about a 10% throughput increase while keeping the KV cache in VRAM.

---

## 11.2 MTP 1 Way

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 32768
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompt 1: Output 505 tokens, 4.0s, 126.17 t/s

Prompt 2: Output 703 tokens, 5.5s, 128.71 t/s

Prompt 3: Output 791 tokens, 6.3s, 125.95 t/s

VRAM used: 15.6 GB

### 128K Context

Prompt 1: Output 564 tokens, 5.6s, 100.48 t/s

Prompt 2: Output 669 tokens, 6.8s, 98.31 t/s

Prompt 3: Output 776 tokens, 8.6s, 89.96 t/s

There is already some offloading happening from GPU memory. Throughput decreases slightly as the context grows.

### 256K Context

I did not run the 256K configuration without MoE offloading because we already know that KV-cache offloading to CPU memory will severely reduce performance.

### 8 MoE Layers Offloaded

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 262144
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
--spec-type draft-mtp --spec-draft-n-max 1
--n-cpu-moe 8
```

Prompt 1: Output 550 tokens, 6.7s, 82.45 t/s

Prompt 2: Output 1126 tokens, 12s, 86.93 t/s

Prompt 3: Output 753 tokens, 8.9s, 84.96 t/s

VRAM used: 15.3 GB

This is already a bit borderline for the available VRAM. The No MTP version with less MoE offloading is actually a tiny bit faster.

### 6 MoE Layers Offloaded

Trying to push further:

Prompt 1: Output 641 tokens, 7.7s, 83.03 t/s

Prompt 2: Output 1074 tokens, 12s, 84.47 t/s

Prompt 3: Output 739 tokens, 8.9s, 83.27 t/s

VRAM used: 15.6 GB

We are past the threshold. It is a bit too much, and throughput starts to degrade.

---

## 11.3 MTP 2 Way

```bash
llama-server
-m ..\Qwen3.6-35B-A3B-UD-IQ3_S.gguf
-ngl 99
--ctx-size 32768
--temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on
--cache-type-k q4_0
--cache-type-v q4_0
-np 1
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompt 1: Output 587 tokens, 4.9s, 119.97 t/s

Prompt 2: Output 1024 tokens, 7.9s, 128.81 t/s

Prompt 3: Output 851 tokens, 6.8s, 125.20 t/s

VRAM used: 15.6 GB

### 128K Context

This was a validation run to see the throughput drop.

Prompt 1: Output 636 tokens, 6.7s, 94.42 t/s

Prompt 2: Output 941 tokens, 10s, 89.19 t/s

Prompt 3: Output 732 tokens, 8.6s, 84.68 t/s

There is no change in the overall trend: the KV cache is offloaded to CPU memory and throughput is slightly reduced.

I did not run MoE offloading here because the trend was already clear from the previous experiments.

### 3-Bit Observations

The 3-bit experiment is similar in some ways to the 4-bit experiment.

To maximise throughput for this model, I would suggest keeping the KV cache fully in VRAM and offloading the minimum number of MoE layers necessary to achieve that.

The 3-bit quantization is faster because more VRAM is available for the KV cache, allowing us to reduce the amount of MoE offloading to CPU memory.

The throughput remains very good even at 128K and 256K context.

---

# 12. Offloading to CPU Memory with 4-Bit Quantization

The 4-bit Qwen3.6-35B-A3B model does not fit entirely in GPU VRAM.

Offloading works reasonably well for MoE models.

For this test we use non-MTP and MTP variants with Q4 KV-cache quantization and a small 32K context.

Models:

* Qwen3.6-35B-A3B-UD-IQ4\_NL
* Qwen3.6-35B-A3B-UD-IQ4\_NL\_MTP

## Non-MTP


| CPU MoE Layers Offloaded |    VRAM |   No MTP + KV Q4 |
| -----------------------: | ------: | ---------------: |
|                       12 | 13.5 GB | **65–66 tok/s** |
|                       14 | 12.7 GB | **62–63 tok/s** |
|                       16 | 12.1 GB | **59–60 tok/s** |
|                       18 | 11.3 GB | **57–59 tok/s** |

## MTP 2 Way


| CPU MoE Layers Offloaded |    VRAM | MTP 2 Way + KV Q4 |
| -----------------------: | ------: | ----------------: |
|                       14 | 13.5 GB |  **65–67 tok/s** |
|                       16 | 12.8 GB |  **59–67 tok/s** |
|                       18 | 12.1 GB |  **58–61 tok/s** |

When the model is CPU-offloaded, MTP provides only a modest improvement because the CPU/PCIe path becomes a dominant part of the inference cost.

The context size was set to 32K tokens.

These figures show how much model can be moved to CPU memory while retaining usable throughput.

---

# 13. Reproducibility: Suspend/Resume Can Affect Performance

On this system, Windows suspend/resume can reduce inference performance.

Final benchmark results were therefore collected after a fresh reboot.

### Tests done after suspend


| CPU MoE |    VRAM | MTP 2 Way + KV Q4 |
| ------: | ------: | ----------------: |
|      14 | 13.5 GB |  **55–57 tok/s** |
|      16 | 12.8 GB |  **52–57 tok/s** |
|      18 | 12.1 GB |  **52–54 tok/s** |


| CPU MoE |    VRAM |   No MTP + KV Q4 |
| ------: | ------: | ---------------: |
|      12 | 13.5 GB | **53–54 tok/s** |
|      14 | 12.8 GB | **52–53 tok/s** |
|      16 | 12.0 GB |   **\~50 tok/s** |
|      18 | 11.3 GB | **48–49 tok/s** |

### Tests done after fresh reboot


| CPU MoE |    VRAM | MTP 2 Way + KV Q4 |
| ------: | ------: | ----------------: |
|      14 | 13.5 GB |  **65–67 tok/s** |
|      16 | 12.8 GB |  **59–67 tok/s** |
|      18 | 12.1 GB |  **58–61 tok/s** |


| CPU MoE |    VRAM |   No MTP + KV Q4 |
| ------: | ------: | ---------------: |
|      12 | 13.5 GB | **65–66 tok/s** |
|      14 | 12.7 GB | **62–63 tok/s** |
|      16 | 12.1 GB | **59–60 tok/s** |
|      18 | 11.3 GB | **57–59 tok/s** |

For CPU-offloaded Qwen3.6-35B-A3B IQ4\_NL, MTP provides little additional benefit when CPU offloading is already the dominant performance constraint.

---

# 14 Qwen 3.8 27B

Early Test Date: 15/08/2026

I would say this is a model that nearly all local LLM nerds have been waiting for, and it does not disappoint.

I used the following Unsloth variants:

* Qwen3.8-27B-IQ4\_NL
* Qwen3.8-27B-Q3\_K\_M
* Qwen3.8-27B-Q3\_K\_S

## Q3\_K\_M quantization

I performed a few quick tests. I have not yet done extensive testing across different context sizes and model variants.

Thinking mode is enabled by default, and it can be disabled with `--reasoning off` (if not set, reasoning is on).

### No MTP

```
llama-server 
-m ..\Qwen3.8-27B-Q3_K_M.gguf 
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 tokens/s

VRAM used: 14 GB

### 64 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_M.gguf 
--ctx-size 65536 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 tokens/s

VRAM used: 14.8 GB

Prompts additional 20 Ktokens context: prefill about 720 tokens/s, inference 22 tokens/s

VRAM used @ 28.3 KTokens : 14.8GB ( 200 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 610 tokens/s, inference 19 tokens/s

VRAM used @ 53.5 KTokens : 14.8 GB ( 200 MB Offloaded)

#### 96 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_M.gguf 
--ctx-size 98304 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 tokens/s

VRAM used: 15.5 GB (300 MB Offloaded) We are already on the edge

Prompts additional 40 Ktokens context: prefill about 630 tokens/s, inference 18 tokens/s

VRAM used @ 52.59 KTokens : 15.6 GB ( 300 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 470 tokens/s, inference 16 tokens/s

VRAM used @ 77.48 KTokens : 15.6 GB ( 300 MB Offloaded)

### MTP 1 Way

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 
--spec-type draft-mtp --spec-draft-n-max 1
```

#### 32 KTok Context

Prompts with the 3 classic questions, one after another: 32 to 35 tokens/s

VRAM used: 14.5 GB

Prompts additional 20 Ktokens context: prefill about 700 tokens/s, inference 31 tokens/s

VRAM used @ 27.5 KTokens : 14.8 GB

#### 64 KTok Context

Prompts with the 3 classic questions, one after another: 32 to 35 tokens/s

VRAM used: 15.5 GB (we are on the edge already, a bit of offloading)

Prompts additional 20 Ktokens context: prefill about 680 tokens/s, inference 30 tokens/s

VRAM used @ 31.6 KTokens : 15.6 GB ( 200 to 400 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 570 tokens/s, inference 28 tokens/s

VRAM used @ 52.7 KTokens : 15.6 GB ( 300 to 400 MB Offloaded)

### MTP 2 Way

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 
--spec-type draft-mtp --spec-draft-n-max 2
```

#### 32 KTok Context

Prompts with the 3 classic questions, one after another: 35 to 42 tokens/s

VRAM used: 14.7 GB

Prompts additional 20 Ktokens context: prefill about 700 tokens/s, inference 34 tokens/s

VRAM used @ 27.5 KTokens : 15.0 GB

#### 64 KTok Context

```bash
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 65536 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 33 to 39 tokens/s

VRAM used: 15.6 GB (we are on the edge already, a bit of offloading)

Prompts additional 20 Ktokens context: prefill about 660 tokens/s, inference 33 tokens/s

VRAM used @ 31.3 KTokens : 15.6 GB (400 to 500 MB offloading on CPU/Memory)

Prompts additional 20 Ktokens context: prefill about 550 tokens/s, inference 29 tokens/s

VRAM used @ 56.8 KTokens : 15.6 GB (500 MB offloading on CPU/Memory)

### MTP 3 Way

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 
--spec-type draft-mtp --spec-draft-n-max 3
```

Prompts with the 3 classic questions, one after another: 33 to 40 tokens/s (large swings in throughput)

VRAM used: 14.8 GB

### Chat/Instruct Mode

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf 
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0 
--spec-type draft-mtp --spec-draft-n-max 2 --reasoning off
```

Similar behavior in terms of memory usage and throughput. The answers are a similar length to Qwen 3.6 27B, so this is a good mode for interactive use.

### Observations

The default thinking effort is **very high**, and it burns a lot of tokens. Even for a relatively short question and a small context, let's say around 1K tokens, at some point I had about 7.6K tokens of thinking and answer combined. I mean, this is big—you need to have enough context.

Hopefully, it is possible to play with the template and tune the effort. This is a cool feature.

For the chat/instruct mode, the latency is low, and the length of the answers is similar to Qwen 3.6 27B.

MTP also seems to have improved compared with Qwen 3.6. The throughput gain remains strong when increasing the speculation depth, with MTP-2 currently looking like the sweet spot. MTP-3 is still faster than the baseline, but shows larger fluctuations. This may indicate improvements in the MTP routing/acceptance behavior, although more testing is needed to confirm this.

This is only preliminary, but it already means that this is very usable in 3-bit. The best 3-bit quantization variant will of course depend on the context size.

In the experiments, the model Qwen3.8-27B-Q3_K_M.gguf is good up to 64 KTok of context with a MTP 2 Way with a throughput in inference starting around 39 tokens/s going down to 29 tokens/s when the context is nearly full: this is very usable, especially for such a dense model. You can push a 96 KTok of context without MTP, then the throughput starts around 25 tokens/s going down to 16 tokens/s when the context is nearly full: it is on the slow side but still acceptable.

## Q3_K_S

The previous experiments are interesting, but the context is a bit too small. So we try another variant with less VRAM used to have a large context.

```bash
hf download hf://unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_S.gguf
```

We will start directly with 96 KTok context size.

### No MTP

#### 96 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 98304 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 26 tokens/s

VRAM used:  14.5GB (300 MB Offloaded)

Prompts additional 40 Ktokens context: prefill about 620 tokens/s, inference 20 tokens/s

VRAM used @ 52.8 KTokens : 14.6 GB ( 300 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 490 tokens/s, inference 18 tokens/s

VRAM used @  77.6 KTokens : 14.6 GB ( 300 MB Offloaded)

### 128 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 131072 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 tokens/s

VRAM used:  15.3GB (300 MB Offloaded)

Prompts additional 80 Ktokens context: prefill about 530 tokens/s, inference 16 tokens/s

VRAM used @ 102.6 KTokens : 15.3 GB ( 300 MB Offloaded)

### MTP 1 Way

#### 96 KTok Context

````bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 98304 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 1
````

Prompts with the 3 classic questions, one after another: 35 tokens/s

VRAM used:  15.4GB (400 MB Offloaded)

Prompts additional 40 Ktokens context: prefill about 580 tokens/s, inference 27 tokens/s

VRAM used @ 55.5 KTokens : 15.4 GB ( 400 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 490 tokens/s, inference 24 tokens/s

VRAM used @  80.1 KTokens : 15.4 GB ( 300 MB Offloaded)

#### 128 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 131072 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 1
```

Prompts with the 3 classic questions, one after another: 31 tokens/s

VRAM used:  15.6GB (800 MB Offloaded) => Already above the threshold

Prompts additional 80 Ktokens context: prefill about 255 tokens/s, inference 9 tokens/s

VRAM used @ 104.1 KTokens : 15.6 GB (800 MB Offloaded)

#### 112 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 112000 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 1
```

This is a limit for this setup to achieve good throughput

Prompts with the 3 classic questions, one after another: 35 tokens/s

VRAM used:  15.5GB (400 MB Offloaded)

Prompts additional 80 Ktokens context: prefill about 510 tokens/s, inference 23 tokens/s

VRAM used @ 101.8 KTokens : 15.5 GB (400 MB Offloaded)

#### Observations

The sweat spot for the max Context capacity for this specific model using a MTP 1 way is the 112 KTok Context variant. It starts at 35 tokens/s and finish around 23 tokens/s, this is not fast but usable.


### MTP 2 Way

#### 96 KTok Context

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 98304 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 39 tokens/s

VRAM used:  15.5GB (400 MB Offloaded)

Prompts additional 40 Ktokens context: prefill about 590 tokens/s, inference 28 tokens/s

VRAM used @ 54.6 KTokens : 15.5 GB ( 400 MB Offloaded)

Prompts additional 20 Ktokens context: prefill about 470 tokens/s, inference 23 tokens/s

VRAM used @ 79.49 KTokens : 15.5 GB ( 400 MB Offloaded)

#### 112 KTok Context

I did not run on purpose the 128 KTok because we already know the issue.

````bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 112000 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 2

````

Prompts with the 3 classic questions, one after another: 36 tokens/s

VRAM used:  15.5GB (500 MB Offloaded)

Prompts additional 80 Ktokens context: prefill about 470 tokens/s, inference 19 tokens/s

VRAM used @ 101.6 KTokens : 15.4 GB (700 MB Offloaded)

This interesting we have about 100 MB more offloaded at the start and about 300 MB more than the MTP 1 way model, there is a slight reduction in inference all along, meaning too much communication with the CPU/MEM. We need to reduce a bit further to achieve good throughput.

#### 106 KTok Context

Since the last experiment shown already a bit too much offloading reducing the throughput, let's try to restore a bit.

```bash
llama-server 
-m ..\Qwen3.8-27B-Q3_K_S.gguf 
--ctx-size 106000 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
 --spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 38 tokens/s

VRAM used:  15.5GB (500 MB Offloaded)

Prompts additional 80 Ktokens context: prefill about 500 tokens/s, inference 24 tokens/s

VRAM used @ 102.4 KTokens : 15.4 GB (500 MB Offloaded)

We are still on the edge but the throughput reduction is lower when the context is nearly full.

#### Observations

There is no real winner there for the MTP 2 ways: for larger context it is better to stick to a MTP 1 way that has the same throughput and allows to have a bit more context.

## 4-bit quantization

We are testing a 4-bit quantization that is really at the limit or above what the 16 GB can handle.

### Very early test

```
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf 
--ctx-size 32768 -fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --parallel 1 
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 
--presence-penalty 0.0 --repeat-penalty 1.0
```

Too big for the GPU.

700 MB already spilled over to CPU memory.

9 tokens/s

### Offloading Model

### 51 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 51

```

Throughput 8 to 9 tokens/s

VRAM Used 12.8 GB

### 59 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59
```

Throughput 12 to 13 tokens/s

VRAM Used 14.6 GB

### 62 Layers

llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59

Throughput 15 to 16 tokens/s

VRAM Used 15.3 GB

### 59 Layers MTP 1 Way

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59 --spec-type draft-mtp --spec-draft-n-max 1
```

Throughput 18 to 19 tokens/s

VRAM Used 15.2 GB

### 59 Layers MTP 2 Way

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59 --spec-type draft-mtp --spec-draft-n-max 2
```

Throughput 20 to 22 tokens/s

VRAM Used 15.3 GB

### 62 Layers MTP 1 Way

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 62 --spec-type draft-mtp --spec-draft-n-max 1
```

Throughput 19 to 22 tokens/s

VRAM Used 15.6 GB

Memory offloading we are above the edge, when the context starts to fill in the throughput reduces slightly faster.

### Observations

Pretty similar behaviour like the Qwen 3.6 27B in 4-bits quantization. MTP enables additional throughput, with the offloading and the MTP together we achieved around 20 tokens/s in burst mode.

# 15. Other Tested Models

## Ornith 1.0 35B

```bash
llama-server.exe
-m ..\Ornith-1.0-35B-UD-IQ2_XXS.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20
--min-p 0.00 --repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--cache-type-k q4_0 --cache-type-v q4_0
```

Prompt 1: Output 698 tokens, 6.7s, 104.83 t/s

Prompt 2: Output 796 tokens, 7.7s, 103.90 t/s

Prompt 3: Output 856 tokens, 8.3s, 103.38 t/s

VRAM used: 11.8 GB

---

## Ornith 1.0 9B

I am interested in this smaller model in the case an agent runs some command lines, for instance.

It is not really intended for coding, even though it is a nice model.

A Q4\_K\_M variant is used, so VRAM capacity is not a limiting factor.

```bash
llama-server.exe
-m ..\Ornith-1.0-9B-Q4_K_M.gguf
--ctx-size 32768
--temp 0.6 --top-p 0.95 --top-k 20
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
```

Prompt 1: Output 433 tokens, 6.2s, 69.39 t/s

Prompt 2: Output 1081 tokens, 15s, 68.75 t/s

Prompt 3: Output 775 tokens, 11s, 68.30 t/s

VRAM used: 6.8 GB

### MTP Variant

The variant used is provided by protoLabsAI.

```bash
llama-server.exe
-m ..\Ornith-1.0-9B-MTP-Q4_K_M.gguf
--ctx-size 32768
--temp 0.6 --top-p 0.95 --top-k 20
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompt 1: Output 430 tokens, 5.2s, 82.69 t/s

Prompt 2: Output 421 tokens, 4.7s, 89.09 t/s

Prompt 3: Output 268 tokens, 3.2s, 84.16 t/s

VRAM used: 7.3 GB

Good improvement in throughput, and the answers are short.

### Second run with 2 draft tokens

Prompt 1: Output 571 tokens, 6.6s, 86.66 t/s

Prompt 2: Output 1196 tokens, 12s, 98.78 t/s

Prompt 3: Output 801 tokens, 8.9s, 89.79 t/s

VRAM used: 7.4 GB

Another small improvement in throughput, with somewhat longer answers.

I tried values above 2 and saw diminishing returns, so 2 seems to be the best candidate.

---

## Kwaipilot KAT-Coder-V2.5-Dev

This variant is interesting because it is specifically intended for coding and agentic workloads.

The bartowski variant is used; there was no Unsloth implementation at the time of this test.

```bash
llama-server.exe
-m ..\Kwaipilot_KAT-Coder-V2.5-Dev-IQ2_XXS.gguf
--ctx-size 32768 --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1 --cache-type-k q4_0 --cache-type-v q4_0
```

Prompt 1: Output 349 tokens, 4.3s, 81.29 t/s

Prompt 2: Output 609 tokens, 7.4s, 82.00 t/s

Prompt 3: Output 430 tokens, 4.4s, 97.85 t/s

VRAM used: 10.5 GB

---

## Qwythos-9B-Claude-Mythos-5-1M

This is a variant of Qwen 3.5 post-trained on an uncensored model.

```bash
llama-server.exe
-m ..\Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf
--ctx-size 32768
--temp 0.6 --top-p 0.95 --top-k 20
--repeat-penalty 1.05
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompt 1: Output 132 tokens, 1.6s, 83.36 t/s

Prompt 2: Output 280 tokens, 3.4s, 83.12 t/s

Prompt 3: Output 280 tokens, 3.4s, 81.48 t/s

VRAM used: 7.5 GB

### MTP 2

Prompt 1: Output 182 tokens, 1.6s, 75.30 t/s

Prompt 2: Output 473 tokens, 5.6s, 83.79 t/s

Prompt 3: Output 319 tokens, 4.0s, 80.51 t/s

VRAM used: 7.6 GB

MTP does not automatically improve throughput.

## Muse-Glimmer-30B

I used the Unsloth variant in Muse-Glimmer-30B-UD-Q3_K_XL.gguf, the thinking mode is active, so it uses slightly more tokens and it has more latency.

```bash
llama-server 
-m ..\Muse-Glimmer-30B-UD-Q3_K_XL.gguf  
--ctx-size 32768 
--temp 1.0 --top-p 0.95 --top-k 64  
-fa on  
--cache-type-k q4_0 --cache-type-v q4_0 
--n-gpu-layers all --n-gpu-layers-draft all 
--spec-type draft-dflash --spec-draft-p-min 0.2 --spec-draft-n-min 0 --spec-draft-n-max 3 
--parallel 1 --jinja
```

Prompt 1: Output 949 tokens, 33s, 28.21 t/s

Prompt 2: Output 1,011 tokens, 36s, 27.66 t/s

Prompt 3: Output 674 tokens, 24s, 27.23 t/s

VRAM used: 12.6 GB

It means there a big margin on the size of the context.

I will wait a bit to retry it further because the software is pretty new and it needs to mature a bit.

## NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF

I used the Bartowski variant in 4-bit IQ4\_XS for this one. We do some CPU/MEM offloading to retain most of its capabilities. The Bartowski variant also supports MTP if needed.Burst Mode

Burst mode with a small context and minimal offloading, I get around **95 tok/s**:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 
--temp 0.6 --top-p 0.95  --min-p 0.01 
-c 32768 
--n-cpu-moe 8 --reasoning off
```

Another Burst Mode variant gives around **90 tok/s**, with slightly more offloading:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  --jinja -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 --temp 1.0 --top-p 0.95 
-c 32768 --n-cpu-moe 10
```

Larger context 256 KToken:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  
--jinja -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 
--temp 1.0 --top-p 0.95 -c 262144 
--n-cpu-moe 13
```

Burst Mode: 85 tok/s

With 112K Context: 54 tok/s

VRAM used in previous experiments: 15.4 to 15.5 GB

It behaves similarly to other MoE models of this size. MTP does not provide additional throughput on this hardware because the model is already partially offloaded to CPU/MEM.

The throughput is decent, and I find it to be an interesting model. It behaves similarly on the tuning settings to the recipes for the Qwen 3.6 35B-A3B.

What I find particularly interesting is that it remains quite fast for a 4-bit model with offloading. I will probably experiment with it further.

---

# 15. Models and Files Tested

Unless otherwise noted, GGUF models were downloaded from Unsloth.ai.

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

- `Muse-Glimmer-30B-UD-Q3_K_XL.gguf`

---

## NVIDIA-Nemotron-3.5-Lightning-30B-A3B

- `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS`

# 16. Practical Recommendations

After all these experiments, a few rules stand out.

### 1. Keep the KV cache in VRAM if possible

This is probably the most important rule from the experiments.

A small amount of model offloading can be acceptable.

KV-cache offloading can be devastating to throughput.

### 2. For MoE models, offload MoE layers rather than the whole model

MoE models behave much better under CPU offloading than dense models.

The Qwen3.6-35B-A3B IQ4\_NL experiments show that even with a substantial number of MoE layers on the CPU, around 50–65 tok/s is still possible.

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

For the Qwen3.6-35B-A3B IQ2\_XXS configuration, it increased throughput from around 80 tok/s to roughly 130–140 tok/s.

Once CPU offloading becomes the dominant bottleneck, the benefit becomes much smaller.

### 5. A slightly smaller quantization can be better than expected

Moving from 4-bit to 3-bit is not only about making the model fit.

The additional VRAM headroom can be converted into:

* more context,
* a larger KV cache,
* less CPU offloading,
* or some combination of the three.

The Qwen3.6-35B-A3B 3-bit experiments are a good example of this.

### 6. 16 GB is a very interesting capacity

16 GB is enough to run surprisingly large sparse models at very high generation speeds.

At the same time, it is also exactly where memory management becomes important.

A model that fits in 16 GB at 32K context may no longer behave the same way at 128K or 256K.

---

# 17. Experiment Log

The experiments started from a simple question: how far can a 16 GB GPU be pushed for local LLM inference?

The first experiments focused on very aggressive 2-bit quantization. This made it possible to run 25–35B-class sparse models entirely on the GPU at highly interactive speeds.

That led to the next question: if the GPU is already fast enough, can MTP provide another significant improvement?

It did.

The Qwen3.6-35B-A3B configuration went from roughly 80 tok/s to around 130–140 tok/s with MTP, making the model feel extremely responsive.

The next question was whether it was possible to move toward higher-quality quantization.

At 4-bit, the 35B model no longer fits entirely in 16 GB. CPU offloading therefore became necessary.

This worked surprisingly well for the MoE model. Rather than being completely unusable, the model could still generate at around 50–65 tok/s depending on how many MoE layers were moved to CPU memory.

The experiments then became more interesting.

The 3-bit Qwen3.6-35B-A3B configuration fits fully in VRAM at smaller context sizes. At 32K context it reaches roughly 103 tok/s without MTP and around 126 tok/s with MTP 1 Way.

But when increasing the context to 256K, the KV cache becomes too large.

At that point llama.cpp starts moving the KV cache to CPU memory, and throughput collapses to around 22–23 tok/s.

The obvious solution was to move some of the model to CPU memory instead.

With only 4 MoE layers offloaded, the 256K configuration reached around 88–89 tok/s.

That was one of the more interesting results of the whole experiment.

Instead of thinking about CPU offloading as something that should always be avoided, it became clear that **the location of the offloaded data matters**.

A few MoE layers on the CPU can be much better than having the KV cache on the CPU.

This also explains why the optimal configuration changes with context size.

At small context sizes, putting as much of the model as possible on the GPU is generally the best approach.

At very large context sizes, however, VRAM becomes valuable for the KV cache.

Moving a few MoE layers to CPU memory can therefore free enough VRAM to keep the KV cache on the GPU.

This produces a much better overall result.

---

# 18. Conclusions

The RTX 5060 Ti 16 GB proved to be an excellent entry point for local LLM inference.

Sparse models in the 25–35B class now run at speeds that make interactive usage genuinely comfortable, with slightly above 50 tokens/s already feeling very usable and MTP pushing some configurations into a much more responsive range.

Compared with the MacBook Air M4, generation throughput improved by roughly three to five times depending on the model.

These comparisons are based on my earlier M4 experiments using comparable model/configuration combinations; they are not a controlled same-day hardware comparison.

Perhaps the biggest surprise was that even the Qwen 3.6 27B reasoning model remained usable at around 34 tokens/s and around 45–50 tokens/s with MTP.

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