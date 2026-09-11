# Experimental Results: Qwen 3.6 Generation and Other Models

Detailed experiments and measurements behind the practical recommendations in [rtx5060ti-local-llm-benchmark.md](rtx5060ti-local-llm-benchmark.md).

---

# 1. Gemma 4 26B A4B

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

Gemma 4 A4B was one of the fastest models tested, consistently exceeding 100 t/s while remaining below 11 GB of VRAM.

Subjectively, it felt effectively instantaneous during interactive use.

---

# 2. Qwen 3.6 27B

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

For comparison, the same benchmark achieved only about 4–6 t/s on the MacBook Air M4.

### MTP Variant

Prompt 1: Output 586 tokens, 12s, 47.42 t/s

Prompt 2: Output 864 tokens, 17s, 50.10 t/s

Prompt 3: Output 876 tokens, 17s, 49.32 t/s

VRAM used: 11.0 GB

This is an interesting boost in performance, around 40% more throughput. This is still an important improvement from a user-experience point of view.

The best results were with 2-way MTP; beyond that, it does not bring more throughput.

---

# 3. Qwen 3.6 27B: 4-Bit and 3-Bit Quantization

The 2-bit models are fast, but there is a slight reduction in their capabilities.

I took the MTP variants from the Qwen3.6-27B-MTP-GGUF repository:

* 4-bit: IQ4\_NL with 16.3 GB
* 3-bit:
  * Q3\_K\_M with 13.8 GB
  * Q3\_K\_S with 12.6 GB

## 4-Bit with Offloading

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

Note that llama.cpp prioritizes keeping the KV cache in VRAM for performance; however, if VRAM is exhausted, the cache is offloaded to CPU memory, significantly reducing inference throughput.


| Layers in GPU | VRAM Usage | Throughput |
| ------------: | ---------: | ---------: |
|            48 |    13.1 GB |   \~13 t/s |
|            51 |    13.7 GB |   \~14 t/s |
|            59 |    15.4 GB |   \~23 t/s |
|            60 |    15.5 GB |   \~24 t/s |

### Observations

First I kept 48 of the 64 layers on the GPU. The VRAM usage was fine, but the throughput was not great.

Then I pushed a little further, with only a small throughput gain.

Once I pushed as much as possible onto the GPU, at the cost of VRAM and context headroom, the difference became much more significant.

Offloading dense models like this one puts a lot of stress on the relatively slow PCIe link and the CPU/memory subsystem.

## 3-Bit Experiments

Following the previous case, where it matters a lot to fit as much as possible of the dense model on the GPU, I took a 3-bit version of the model.

### Q3\_K\_M

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
| None |   14.1 GB |   \~25 t/s |
|    1 |   14.5 GB |   \~36 t/s |
|    2 |   14.8 GB |   \~41 t/s |

The model fits in VRAM, and there is a major gain in throughput.

MTP-2 produced a very large improvement on the fully GPU-resident Q3\_K\_M configuration, raising throughput from roughly 25 to over 40 t/s.

It is an important gain to use MTP in this context, with some cost in VRAM.

The context is limited for this variant of the 3-bit model. I managed to put above 75K tokens in context, but above a threshold the KV cache is offloaded to CPU RAM and performance degrades a lot, to around 9 t/s.

### Q3\_K\_S

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


| Context |       VRAM Used |                    Throughput |
| ------: | --------------: | ----------------------------: |
|     32K | 13.5 → 13.7 GB |  Start:\~40 → @28K: \~36 t/s |
|     64K | 14.4 → 14.6 GB |  Start:\~40 → @52K: \~31 t/s |
|     96K | 15.5 → 15.5 GB |  Start:\~40 → @77K: \~27 t/s |
|    128K | 15.6 → 15.6 GB | Start:\~37 → @111K: \~19 t/s |
|    110K | 15.5 → 15.5 GB |  Start:\~40 → @77K: \~27 t/s |

The smaller Q3\_K\_S variant can maintain full GPU residency beyond 100K context, although generation throughput progressively decreases as the KV cache grows.

The smaller Q3\_K\_S variant leaves enough VRAM headroom for a very large Q4 KV cache. In testing, it remained GPU-resident at approximately 100K context and still generated at 27 t/s near the end of the context.

This demonstrates that reducing model size slightly can be more valuable than expected when the saved VRAM is converted into KV-cache capacity.

---

# 4. Qwen 3 Coder 30B A3B Instruct

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

The coding model maintained around 80 t/s while producing relatively concise responses, making the measured generation speed particularly suitable for interactive coding workloads.

---

# 5. Qwen 3.6 35B A3B

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

Despite its larger size, it consistently maintained over 80 t/s while remaining comfortably within the 16 GB VRAM budget.

### MTP Variant

Prompt 1: Output 729 tokens, 5.6s, 130.87 t/s

Prompt 2: Output 721 tokens, 5.1s, 142.11 t/s

Prompt 3: Output 778 tokens, 5.7s, 137.68 t/s

VRAM used: 12.5 GB

This is a big improvement in throughput. Without MTP it is already very fast, but there is a good 1 GB increase in VRAM usage due to 2-way MTP.

The throughput increase is around 40–60%, which is significant.

There is no further improvement in throughput beyond two draft tokens.

---

# 6. Qwen 3.6 35B A3B: Higher-Quality 4-Bit Quantization

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

Offloading 16 layers reaches about 59 t/s. This is not extremely fast, but the user experience is still very good.

Offloading 12 layers gives slightly higher throughput, around 65 t/s, but VRAM usage is already 14 GB, so there is not much margin for a small increase in throughput.

Offloading 20 layers gives around 55 t/s. The user experience is still good and only 11.2 GB of VRAM is used, making it an interesting configuration.

Offloading 24 layers reduces throughput to around 50 t/s. The user experience is still acceptable, and the 9.7 GB VRAM usage leaves a lot of headroom for experimentation.

The results suggest that PCIe transfers and/or CPU memory bandwidth become important bottlenecks once a significant portion of the MoE computation is moved to the CPU.

---

# 7. Qwen 3.6 35B A3B: CPU Offloading + MTP

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

# 8. CPU Offloading and Context Size

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

Prefill: 24608 tokens, 28s, 859.18 t/s

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

Prefill: 49184 tokens, 58s, 843.86 t/s

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

Prefill: 97825 tokens, 2min 2s, 799.61 t/s

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

Prefill: 195105 tokens, 4min 34s, 711.45 t/s

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

Prefill: 195105 tokens, 4min 28s, 727.78 t/s

Prompt: Count the number of repetitions in this file

Inference: 204 tokens, 7.0s, 29.19 t/s

VRAM used: 14.3 GB at the end of the inference.

The main observation is that very large context is possible, but throughput progressively decreases as the KV cache grows.

---

# 9. Qwen 3.6 35B A3B: 3-Bit Quantization

This is the most interesting combination for squeezing the 35B A3B model into a 16 GB GPU.

I used the Qwen3.6-35B-A3B-UD-IQ3\_S model from Unsloth.

The 3-bit version fits fully on the GPU at smaller context sizes, leaving more VRAM available for the KV cache than the 4-bit version.

## 9.1 No MTP

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

## 9.2 256K Context — Offloading MoE Layers

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

## 9.3 MTP 1

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

## 9.4 MTP 2

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

# 10. Offloading to CPU Memory with 4-Bit Quantization

The 4-bit Qwen3.6-35B-A3B model does not fit entirely in GPU VRAM.

Offloading works reasonably well for MoE models.

For this test we use non-MTP and MTP variants with Q4 KV-cache quantization and a small 32K context.

Models:

* Qwen3.6-35B-A3B-UD-IQ4\_NL
* Qwen3.6-35B-A3B-UD-IQ4\_NL\_MTP

## Non-MTP


| CPU MoE Layers Offloaded |    VRAM | No MTP + KV Q4 |
| -----------------------: | ------: | -------------: |
|                       12 | 13.5 GB | **65–66 t/s** |
|                       14 | 12.7 GB | **62–63 t/s** |
|                       16 | 12.1 GB | **59–60 t/s** |
|                       18 | 11.3 GB | **57–59 t/s** |

## MTP 2


| CPU MoE Layers Offloaded |    VRAM |  MTP 2 + KV Q4 |
| -----------------------: | ------: | -------------: |
|                       14 | 13.5 GB | **65–67 t/s** |
|                       16 | 12.8 GB | **59–67 t/s** |
|                       18 | 12.1 GB | **58–61 t/s** |

When the model is CPU-offloaded, MTP provides only a modest improvement because the CPU/PCIe path becomes a dominant part of the inference cost.

The context size was set to 32K tokens.

These figures show how much model can be moved to CPU memory while retaining usable throughput.

---

# 11. Reproducibility: Suspend/Resume Can Affect Performance

On this system, Windows suspend/resume can reduce inference performance.

Final benchmark results were therefore collected after a fresh reboot.

## 11.1 Post-Suspend Performance


| CPU MoE |    VRAM |  MTP 2 + KV Q4 |
| ------: | ------: | -------------: |
|      14 | 13.5 GB | **55–57 t/s** |
|      16 | 12.8 GB | **52–57 t/s** |
|      18 | 12.1 GB | **52–54 t/s** |


| CPU MoE |    VRAM | No MTP + KV Q4 |
| ------: | ------: | -------------: |
|      12 | 13.5 GB | **53–54 t/s** |
|      14 | 12.8 GB | **52–53 t/s** |
|      16 | 12.0 GB |   **\~50 t/s** |
|      18 | 11.3 GB | **48–49 t/s** |

## 11.2 Post-Reboot Performance


| CPU MoE |    VRAM |  MTP 2 + KV Q4 |
| ------: | ------: | -------------: |
|      14 | 13.5 GB | **65–67 t/s** |
|      16 | 12.8 GB | **59–67 t/s** |
|      18 | 12.1 GB | **58–61 t/s** |


| CPU MoE |    VRAM | No MTP + KV Q4 |
| ------: | ------: | -------------: |
|      12 | 13.5 GB | **65–66 t/s** |
|      14 | 12.7 GB | **62–63 t/s** |
|      16 | 12.1 GB | **59–60 t/s** |
|      18 | 11.3 GB | **57–59 t/s** |

For CPU-offloaded Qwen3.6-35B-A3B IQ4\_NL, MTP provides little additional benefit when CPU offloading is already the dominant performance constraint.

---

# 12. Other Tested Models

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

This indicates significant VRAM headroom for context expansion.

Pending further maturation of the software stack, these results are preliminary.
*Update 21/08/2026*: Daily use Qwen 3.8 27B - more consistent for my tasks.

## NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF

I used the Bartowski variant in 4-bit IQ4\_XS for this one. CPU/MEM offloading is required to fit the model to retain most of its capabilities. The Bartowski variant also supports MTP if needed.

Burst Mode (High Throughput)

Burst mode with a small context and minimal offloading, I get around **95 t/s**:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 
--temp 0.6 --top-p 0.95  --min-p 0.01 
-c 32768 
--n-cpu-moe 8 --reasoning off
```

Another Burst Mode variant gives around **90 t/s**, with slightly more offloading:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  --jinja -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 --temp 1.0 --top-p 0.95 
-c 32768 --n-cpu-moe 10
```

Larger context 256 K:

```bash
llama-server.exe 
-hf bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:IQ4_XS 
-ngl 99  
--jinja -np 1 
--cache-type-k q4_0 --cache-type-v q4_0 
--temp 1.0 --top-p 0.95 -c 262144 
--n-cpu-moe 13
```

Burst Mode: 85 t/s

With 112K Context: 54 t/s

VRAM used in previous experiments: 15.4 to 15.5 GB

It behaves similarly to other MoE models of this size. MTP does not provide additional throughput on this hardware because the model is already partially offloaded to CPU/MEM.

The throughput is decent, and I find it to be an interesting model. It behaves similarly on the tuning settings to the recipes for the Qwen 3.6 35B-A3B.

What I find particularly interesting is that it remains quite fast for a 4-bit model with offloading. I will probably experiment with it further.

## Ornith-1.5-35B-A3B

### 4-bit quantization

I used the official 4-bit quantization from ornith-ai.

#### No MTP

```bash
llama-server -m ..\Ornith-1.5-35B-Q4_K_M.gguf
 --ctx-size 32768 -fa on  --cache-type-k q4_0 --cache-type-v q4_0  
--parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --n-cpu-moe 13
```

##### 32K

Prompt 1: Output 1,549 tokens, 21s, 72.11 t/s

Prompt 2: Output 1,013 tokens, 13s, 72.46 t/s

Prompt 3: Output 1,412 tokens, 19s, 71.59 t/s

VRAM used: 15.4 GB (200 MB Offload)

##### 128 K

```bash
llama-server -m ..\Ornith-1.5-35B-Q4_K_M.gguf
 --ctx-size 131072 -fa on  --cache-type-k q4_0 --cache-type-v q4_0  
--parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --n-cpu-moe 14
```

Prompt 1: Output 1,544 tokens, 21s, 70.76 t/s

Prompt 2: Output 996 tokens, 13s, 71.29 t/s

Prompt 3: Output 1,015 tokens, 14s, 70.64 t/s

VRAM used: 15.4 GB (200 MB Offload)

Adding 49182 tokens

Prefill 1min 2s, 789.96 tokens/s

53.7K Output 882 tokens 16s 53.45 t/s

##### 240 K

```bash
llama-server -m ..\Ornith-1.5-35B-Q4_K_M.gguf
 --ctx-size 240000 -fa on  --cache-type-k q4_0 --cache-type-v q4_0  
--parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --n-cpu-moe 15
```

VRAM used: 15.6 GB (400 MB Offload)

Adding 97821 tokens

Prefill 2min 15s, 723.77 tokens/s

98.49K Output 666 tokens 15s 43.22 t/s

Adding 97821 tokens

Prefill 2min 45s, 589.77 tokens/s

196.7K Output 419 tokens 15s 31.78 t/s

#### MTP

I did not run with MTP.

**There is a performance issue on the MTP heads** it seems those are not trained and the throughput is not good.

Actually it is not a problem, because the model does not fit in the GPU and with MoE offloading you have to balance the GPU VRAM to keep the KV Cache in the GPU so not having the MTP gives more context, or you can offload less MoE on the CPU/RAM and have higher throughput.

### Observations

It is very usable without MTP and a large context. There is a large throughput slowdown when using the 240K context.

### 3-bit quantization

I am using a variant from `AtomicChat/Ornith-1.5-35B-A3B-GGUF`

- Ornith-1.5-35B-A3B-AD-IQ4_XS-IQ3_S (*NB*: MTP not shipped with the model, separated.)

#### IQ4_XS-IQ3_S

The model does not fit in VRAM few layers are off-loaded.

##### 32K

```bash
llama-server -m ..\Ornith-1.5-35B-A3B-AD-IQ4_XS-IQ3_S.gguf"  
--ctx-size 32768 -fa on  --cache-type-k q4_0 --cache-type-v q4_0  
--parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --n-cpu-moe 4
```

Prompt 1: Output 2,911 tokens, 34s, 84.94 t/s

Prompt 2: Output 1,049 tokens, 12s, 84.83 t/s

Prompt 3: Output 1,400 tokens, 16s, 84.25 t/s

VRAM used: 15.6 GB (200 MB Offload)

*Note*: Lot of thinking, lot of tokens.

##### 128K

```bash
llama-server -m ..\Ornith-1.5-35B-A3B-AD-IQ4_XS-IQ3_S.gguf  
--ctx-size 131072 -fa on  --cache-type-k q4_0 --cache-type-v q4_0  
--parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --n-cpu-moe 7
```

Prompt 1: Output 1,392 tokens, 18s, 76.15 t/s

Prompt 2: Output 910 tokens, 11s, 77.40 t/s

Prompt 3: Output 2,448 tokens, 32s, 75.89 t/s

VRAM used: 15.6 GB (200 MB Offload)

Add 49667 tokens 40s 1229.67 tokens/s

51.6K Output: 929 tokens, 16s, 56.50 t/s

Add 25032 tokens 23s 1045.43 tokens/s

76.2K Output: 494 tokens,9.7s, 50.87 t/s

#### Observations

A bit faster but the throughput there is not a huge gap with respect to the official Q4 when the offloading is enabled. We still have a small offloading to the CPU/MEM. For larger context, you need to increase further the number of MoE offloaded.

