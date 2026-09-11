# Qwen 3.8 27B — DFlash2 Speculative Decoding on RTX 5060 Ti 16 GB

**TL;DR:** DFlash2 speculative decoding (Qwen 3.8 27B GSQ RCO `IQ3_XXS` + DFlash2 draft) sustains ~60–65 t/s on real coding tasks at 32K–64K context on a 16 GB RTX 5060 Ti. Throughput erodes as context grows; at 162K the combined dedicated + shared GPU memory footprint (~16.7 GB) exceeds the card's VRAM, offloading becomes the bottleneck, and speed can fall to ~23 t/s.

## 1. Test Configuration

### Hardware

* **GPU:** NVIDIA RTX 5060 Ti — 16 GB
* **OS:** Windows 11
* **Hardware reserved memory:** 334 MB

### Software

* **Inference engine:** llama.cpp 0.4.0
* **CUDA:** 13
* **NVIDIA Driver:** Studio 616.56

### Models

**Main model:** Qwen 3.8 27B GSQ RCO — IQ3_XXS [Link](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)

**Draft model:** HermiHg Qwen 3.8 27B DFlash2 Q2_K_S MIX [Link](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF)

### DFlash2 Configuration

The following is a **verbatim configuration example** used for the 162K / Big Context test:

```text
[JBDRAFTBig_Qwen3.8-27B-GSQ-RCO-IQ3_XXS]
model = C:\Users\admin\.cache\huggingface\hub\models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF\snapshots\858aa201794c92fbd49778ec95efe14752c09759\Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
model-draft = C:\Users\admin\.cache\huggingface\hub\models--HermiHg--Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF\snapshots\db38269d5ce14f887b4d4d10b30edc9b5fd7e408\Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf
ctx-size = 162000
n-gpu-layers = 999
flash-attn = on
cache-type-k = q4_0
cache-type-v = q4_0
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
reasoning-effort = medium
spec-type = draft-dflash
spec-draft-n-max = 4
cache-type-k-draft = q4_0
cache-type-v-draft = q4_0
spec-draft-type-k = q4_0
spec-draft-type-v = q4_0
n-gpu-layers-draft = 999
```

The same main model, draft model, and speculative-decoding parameters were used throughout the experiment. The primary variable was the **context size**.

### Context Profiles

| Profile    | Context |
| ---------- | ------: |
| **Tiny**   |     32K |
| **Small**  |     64K |
| **Normal** |     96K |
| **Large**  |    128K |
| **Big**    |    162K |

---

## 2. Experiment Goal

Evaluate inference performance on **coding and code-generation tasks** using DFlash2 speculative decoding.

The objective is to observe:

* Tokens/second
* Throughput as context increases
* Dedicated GPU VRAM usage
* Shared GPU memory usage
* Memory pressure and apparent offloading
* Throughput degradation near and beyond the 16 GB VRAM limit

The workload consists primarily of game-development and code-generation prompts in several programming languages.

---

## 3. Test Method

The workload was progressively extended as the context window increased.

Initial task:

> In Python, write me a Dungeon Crawler — a text-based roguelike game with about 1,000 lines of code.

The generated game was then recreated in other languages, followed by increasingly demanding graphics and game-development tasks.

This provides a practical coding workload rather than a synthetic token-generation benchmark.

---

## 4. Results

## 32K Context — Tiny

**Context:** 32,768 tokens

| Task                     | Tokens |   Time |         Speed |
| ------------------------ | -----: | -----: | ------------: |
| Dungeon Crawler — Python | 11,931 | 3m 00s | **66.05 t/s** |
| Dungeon Crawler — C      | 14,371 | 3m 40s | **65.07 t/s** |

**Context usage:** 26.68K / 32.77K

**Dedicated VRAM:** 13.3–13.4 GB
**Shared GPU memory:** 700–800 MB

**Observation:** Extremely fast, maintaining approximately **65 t/s**.

---

## 64K Context — Small

**Context:** 65,536 tokens

| Task                         | Tokens |   Time |         Speed |
| ---------------------------- | -----: | -----: | ------------: |
| Dungeon Crawler — Python     | 11,033 | 2m 48s | **65.61 t/s** |
| Dungeon Crawler — C          | 12,727 | 3m 16s | **64.91 t/s** |
| Dungeon Crawler — Java       | 14,163 | 3m 55s | **60.06 t/s** |
| Dungeon Crawler — JavaScript | 12,761 | 3m 43s | **57.14 t/s** |

**Context usage:** 51.10K / 65.54K

**Dedicated VRAM:** 14.1–14.2 GB
**Shared GPU memory:** ~1 GB

**Observation:** Still extremely fast, with throughput ranging from **57–66 t/s**.

---

## 96K Context — Normal

**Context:** 98,308 tokens

| Task                         | Tokens |   Time |         Speed |
| ---------------------------- | -----: | -----: | ------------: |
| Dungeon Crawler — Python     | 10,618 | 2m 40s | **66.10 t/s** |
| Dungeon Crawler — C          | 13,837 | 3m 29s | **66.17 t/s** |
| Dungeon Crawler — Java       | 11,501 | 3m 03s | **62.79 t/s** |
| Dungeon Crawler — JavaScript | 11,395 | 3m 09s | **60.17 t/s** |
| Breakout — TypeScript        | 11,723 | 3m 50s | **50.76 t/s** |
| Pong — JavaScript            | 10,984 | 3m 41s | **49.63 t/s** |
| 3D Tetris — JavaScript       | 10,614 | 4m 02s | **43.80 t/s** |

**Context usage:** 81.15K / 98.56K

**Dedicated VRAM:** 14.8–14.9 GB
**Shared GPU memory:** ~1 GB

**Observation:** Still fast, but memory pressure is becoming increasingly important. Even near the end of the context, throughput remains **above 40 t/s**.

---

## 128K Context — Large

**Context:** 128,000 tokens

| Task                         | Tokens |   Time |         Speed |
| ---------------------------- | -----: | -----: | ------------: |
| Dungeon Crawler — Python     |  9,408 | 2m 28s | **63.38 t/s** |
| Dungeon Crawler — C          | 12,190 | 3m 08s | **64.51 t/s** |
| Dungeon Crawler — Java       | 10,982 |    ~3m | **60.90 t/s** |
| Dungeon Crawler — JavaScript | 11,240 | 3m 10s | **59.14 t/s** |
| Breakout — TypeScript        | 11,922 | 3m 51s | **51.59 t/s** |
| Pong — JavaScript            | 11,435 | 3m 56s | **48.27 t/s** |
| 3D Tetris — JavaScript       |  9,832 | 3m 50s | **42.69 t/s** |
| Pagoda / Three.js experiment | 10,246 | 4m 08s | **41.20 t/s** |
| 3D Pac-Man — JavaScript      | 10,378 | 4m 16s | **40.49 t/s** |

**Context usage:** 98.17K / 128K

**Dedicated VRAM:** 15.0–15.1 GB
**Shared GPU memory:** ~1.1 GB

**Observation:** Still usable and surprisingly fast, but memory pressure is now a significant limitation.

---

## 162K Context — Big

**Context:** 162,000 tokens

| Task                         | Tokens |    Time |         Speed |
| ---------------------------- | -----: | ------: | ------------: |
| Dungeon Crawler — Python     |  9,889 |  2m 52s | **57.33 t/s** |
| Dungeon Crawler — C          | 12,389 |  3m 29s | **59.21 t/s** |
| Dungeon Crawler — Java       | 10,822 |  3m 16s | **55.17 t/s** |
| Dungeon Crawler — JavaScript | 12,152 |       — | **50.54 t/s** |
| Breakout — TypeScript        | 20,975 |      8m | **39.07 t/s** |
| Pong — JavaScript            | 13,608 |  6m 05s | **37.22 t/s** |
| 3D Tetris — JavaScript       | 29,868 | 16m 26s | **30.26 t/s** |
| Pagoda / Three.js experiment | 25,504 | 16m 08s | **26.33 t/s** |
| 3D Pac-Man — JavaScript      | 26,309 |       — | **23.07 t/s** |

**Context usage:** 162.05K / 162.05K (slight overflow beyond the 162,000 `ctx-size`)

**Dedicated VRAM:** 15.5 GB
**Shared GPU memory:** 1.2 GB
**Hardware reserved:** 334 MB

**Observation:** The additional context pushes the system further into memory pressure and offloading. Throughput now drops substantially, reaching approximately **23 t/s** on the final task.

**Remark:** Too much offloading. Inference throughput becomes noticeably slower earlier in the context.

---

## 5. Memory Behavior and Offloading

Dedicated VRAM does **not** represent the complete GPU memory footprint in this experiment.

| Context  | Dedicated VRAM | Shared GPU Memory | Approx. Combined |
| -------- | -------------: | ----------------: | ---------------: |
| **32K**  |   13.3–13.4 GB |        0.7–0.8 GB |    ~14.0–14.2 GB |
| **64K**  |   14.1–14.2 GB |           ~1.0 GB |    ~15.1–15.2 GB |
| **96K**  |   14.8–14.9 GB |           ~1.0 GB |    ~15.8–15.9 GB |
| **128K** |   15.0–15.1 GB |           ~1.1 GB |    ~16.1–16.2 GB |
| **162K** |        15.5 GB |           ~1.2 GB |         ~16.7 GB |

The 162K test makes the memory limitation particularly clear.

At 162K context, dedicated VRAM reaches approximately **15.5 GB**, while shared GPU memory rises to approximately **1.2 GB**. The combined reported memory footprint is therefore around **16.7 GB**, already beyond the RTX 5060 Ti's 16 GB of dedicated VRAM.

As the context grows, the system increasingly relies on memory outside the GPU's dedicated VRAM. This creates additional memory-bandwidth and latency overhead and causes inference throughput to deteriorate.

The progression is particularly visible when comparing the end of the 128K and 162K tests:

* **128K:** ~40.49 t/s on the final 3D Pac-Man task
* **162K:** ~23.07 t/s on the final 3D Pac-Man task

This is a substantial additional performance penalty as the context approaches the 162K limit.

The exact division between llama.cpp offloading and Windows/NVIDIA memory management cannot be determined from these measurements alone. However, the increasing shared-memory usage and corresponding throughput reduction strongly indicate that **memory capacity and memory movement have become the dominant bottlenecks**.

---

## 6. Performance Summary

| Context  | Performance Range | End-of-Context Behavior |
| -------- | ----------------: | ----------------------- |
| **32K**  |           ~65 t/s | ~65 t/s                 |
| **64K**  |        ~57–66 t/s | ~57 t/s                 |
| **96K**  |        ~44–66 t/s | ~44 t/s                 |
| **128K** |        ~40–65 t/s | ~40 t/s                 |
| **162K** |        ~23–59 t/s | ~23 t/s                 |

The overall trend is now much clearer:

**32K → 64K:** Very high throughput
**96K:** Still highly usable
**128K:** Noticeable memory pressure and gradual slowdown
**162K:** Heavy memory pressure/offloading with significant throughput degradation

The 162K test demonstrates that simply increasing the context window does not come for free on a 16 GB GPU. Once the workload exceeds the practical dedicated-VRAM capacity, the performance penalty becomes increasingly severe.

---

## 7. Context vs. Memory

The experiment shows a clear relationship between context size, memory usage, and throughput.

```text
Context        Dedicated VRAM    Shared GPU Memory
--------------------------------------------------
32K            13.3–13.4 GB      0.7–0.8 GB
64K            14.1–14.2 GB      ~1.0 GB
96K            14.8–14.9 GB      ~1.0 GB
128K           15.0–15.1 GB      ~1.1 GB
162K           15.5 GB           ~1.2 GB
```

The dedicated VRAM usage increases steadily, but the more important signal is that **shared GPU memory also increases as context grows**.

This suggests that the GPU is not simply using more of its available VRAM. It is progressively relying on memory outside the physical VRAM pool.

---

## 8. Conclusion

This experiment demonstrates that **Qwen 3.8 27B GSQ RCO with DFlash2 speculative decoding can achieve exceptionally high coding throughput on an RTX 5060 Ti 16 GB**.

> See also: [RTX 5060 Ti — 16 GB general benchmark report](rtx5060ti-local-llm-benchmark.md).

At moderate context sizes, the system can sustain approximately **60–65 t/s**, which is an impressive result for a 27B model on a 16 GB consumer GPU.

The larger-context tests reveal the practical limitation of the hardware.

At 128K context:

* Dedicated VRAM: **15.0–15.1 GB**
* Shared GPU memory: **~1.1 GB**
* Final observed throughput: **~40 t/s**

At 162K context:

* Dedicated VRAM: **15.5 GB**
* Shared GPU memory: **~1.2 GB**
* Final observed throughput: **~23 t/s**

The 162K test therefore makes the offloading effect much more obvious. The workload is operating beyond the practical dedicated-VRAM capacity, and increasing amounts of memory are being handled through shared/mapped memory.

The resulting performance loss becomes significant.

### Bottom Line

> **DFlash2 makes Qwen 3.8 27B remarkably fast on a 16 GB RTX 5060 Ti, reaching roughly 65 t/s at moderate context sizes. 32K–64K is the clear performance sweet spot. At 96K–128K, memory pressure begins to reduce throughput, while at 162K the system enters heavy offloading and throughput can fall to around 23 t/s.**

The experiment therefore shows that **DFlash2 speculative decoding is highly effective, but VRAM capacity remains the fundamental constraint when pushing a 27B model toward extremely large context windows.**

The 162K result is particularly useful because it demonstrates that **the limiting factor is no longer primarily compute or speculative decoding efficiency—it is memory capacity, memory movement, and the resulting offloading overhead.**
