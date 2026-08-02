# Practical Local LLM Performance on an RTX 5060 Ti 16GB

Tests conducted in July/August 2026.

Following my earlier experiments on the MacBook Air M4, I finally decided to add a dedicated GPU to my workstation.

This report is not intended to compare model quality or reasoning capabilities. The goal is simply to determine which GGUF models are practical to run on an RTX 5060 Ti 16 GB in terms of inference speed, VRAM usage, and overall usability.

This report focuses on generation throughput. Prompt processing (prefill) was intentionally not measured separately.

## Why buy an RTX 5060 Ti ?

Until recently I never installed a dedicated GPU in my workstation because most of my workloads were CPU-bound.

The recent generation of sparse Mixture-of-Experts models has changed that. There are now several capable LLMs that comfortably fit within 16 GB of VRAM.

The RTX 5060 Ti provides over 400 GB/s of memory bandwidth, compared with roughly 120 GB/s for the MacBook Air M4. Since LLM inference is largely memory-bandwidth bound, I expected approximately a threefold improvement in generation throughput.

From a practical point of view, this moves local LLMs from small experiments to tools that can be used productively every day. Running coding assistants and autonomous agents locally becomes genuinely feasible.

## Setup

### Hardware

* **Motherboard:** MinisForum AMD Ryzen 9 7945HX BD795M
* **Memory:** Corsair CMSX64GX5M2A5200C44 (2 × 32 GB)
* **SSD:** Lexar SSD NQ790 2 TB
* **GPU**: ASUS Prime GeForce RTX 5060 Ti 16 GB GDDR7 OC Edition
* **PSU**: Seasonic Core GX-650 V2

The CPU was configured with a conservative power limit of approximately 75 W peak. I describe the rationale and configuration in a separate note. During testing, the CPU was never the limiting factor.

### Software

- Windows 11 25H2
- NVIDIA Studio Driver 610.62 (CUDA 13)
- llama.cpp b10069 (CUDA 13.3)

# Executive Summary

Overall the RTX 5060 Ti exceeded my expectations. Every tested model fit comfortably within 16 GB of VRAM using aggressive quantisation, and most sparse models delivered between 75 and 90 tokens/s without MTP. Even larger reasoning-oriented models remained responsive enough for interactive use.


| Model                         | Avg tok/s | VRAM    | Practical on 16 GB? |
| ----------------------------- | --------- | ------- | -------------------- |
| Gemma 4 26B A4B               | \~90      | 10.9 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 27B                  | \~30      | 10.4 GB | ⭐⭐⭐☆☆           |
| Qwen 3 Coder 30B A3B Instruct | \~75      | 11.5 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 35B A3B Instruct     | \~80      | 11.2 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 27B MTP              | \~45      | 11.0 GB | ⭐⭐⭐⭐☆           |
| Qwen 3.6 35B A3B Instruct MTP | \~120     | 12.5 GB | ⭐⭐⭐⭐⭐           |
| Ornith 1.0 35B                | \~90      | 11.8 GB | ⭐⭐⭐⭐⭐           |
| Ornith 1.0 9B                 | \~60      | 6.8 GB  | ⭐⭐⭐⭐☆           |
| Ornith 1.0 9B MTP             | \~80      | 7.4 GB  | ⭐⭐⭐⭐⭐           |
| KAT-Coder-V2.5-Dev            | \~80      | 10.5 GB | ⭐⭐⭐⭐⭐           |

## MTP

The RTX 5060 TI responds well to MTP with 2 ways.

There is at least 40% throughput increase on Qwen 3.6 27B Dense and Qwen 3.6 35B A3B, it is a major improvement:

- For the dense 27B the user experience is moving from experiments to the low confort zone.
- For the sparse 35B the throughput is such that agentic job is possible locally.

The MTP throughput increase is good enough, to start to thing about tuning the model to improve 'knowledge' with less quantification, or having more 'precision' in the KV cache.

I hope we will see also more smaller dense models with MTP.

# Tested Models

Unless otherwise noted, all GGUF models were downloaded from Unsloth.ai.

* Gemma 4 26B A4B

  * gemma-4-26B-A4B-it-UD-IQ2\_XXS.gguf [Link](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf)
* Qwen 3.6 27B

  * Qwen3.6-27B-UD-IQ2\_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF?show_file_info=Qwen3.6-27B-UD-IQ2_XXS.gguf)
* Qwen 3 Coder 30B A3B Instruct

  * Qwen3-Coder-30B-A3B-Instruct-UD-IQ2\_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/blob/main/Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf)
* Qwen 3.6 35B A3B

  * Qwen3.6-35B-A3B-UD-IQ2\_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf)
* Qwen 3.6 27B MTP (same filename - small difference in the context)

  * Qwen3.6-27B-UD-IQ2_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF?show_file_info=Qwen3.6-27B-UD-IQ2_XXS.gguf)
* Ornith-1.0-35B

  * Ornith-1.0-35B-UD-IQ2\_XXS.gguf [Link](https://huggingface.co/unsloth/Ornith-1.0-35B-GGUF?show_file_info=Ornith-1.0-35B-UD-IQ2_XXS.gguf)
* Ornith-1.0-9B

  * Ornith-1.0-9B-Q4_K_M.gguf [Link](https://huggingface.co/unsloth/Ornith-1.0-9B-GGUF?show_file_info=Ornith-1.0-9B-Q4_K_M.gguf)
* KAT-Coder-V2.5-Dev

  * bartowski provided model: Kwaipilot\_KAT-Coder-V2.5-Dev-IQ2\_XXS.gguf [Link](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF?show_file_info=Kwaipilot_KAT-Coder-V2.5-Dev-IQ2_XXS.gguf)

## Prompts Used

* Prompt 1: I think that the 42 answer is also a sarcastic way to what question matters and the importance of a good question. What do you think
* Prompt 2: Can you enumerate such questions
* Prompt 3: I would say the usage of a llm is a bit like asking such kind of question

**Methodology:** Each model was given the same three prompts. Models were tested under similar interactive settings, with model-specific parameters adjusted when required for correct chat behavior. The reported token count is the number of output tokens actually generated by the model before it stopped naturally. Since different models produce different response lengths—especially reasoning models—the generation time is not directly comparable across models. The primary metric of interest is sustained generation throughput (tokens/second), while the output token counts illustrate how verbose each model is.

Unless otherwise noted, the tests use a 4-bit KV cache together with Flash Attention to maximise the available context while keeping VRAM usage low.

# Models

## Gemma 4 26B A4B

### 4-bit KV Cache, FA On

````bash
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
````

Prompt 1: Output 522 tokens 5.1s 102.45 t/s

Prompt 2: Output 741 tokens 7.0s 105.35 t/s

Prompt 3: Output 661 tokens 6.5s 101.35 t/s

VRAM used: 10.9 GB

### 4-bit KV Cache, FA On, Parallel Seq 1

````bash
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
````

Prompt 1: Output 569 tokens 5.1s 102.33 t/s

Prompt 2: Output 871 tokens 8.2s 106.23 t/s

Prompt 3: Output 764 tokens 7.7s 98.62 t/s

VRAM used: 10.7 GB

Gemma 4 A4B was the fastest model tested, consistently exceeding 100 tokens/s while remaining below 11 GB of VRAM. Subjectively, it felt effectively instantaneous during interactive use.

## Qwen 3.6 27B

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

Prompt 1: Output 542 tokens 15s 34.56 t/s

Prompt 2: Output 822 tokens 23s 34.41 t/s

Prompt 3: Output 766 tokens 22s 34.23 t/s

VRAM used: 10.4 GB

### Observations

Despite being significantly slower than the sparse models, it still delivers a comfortable interactive experience for reasoning tasks. For comparison, the same benchmark achieved only about 4–6 tokens/s on the MacBook Air M4.

### MTP Variant

Prompt 1: Output 586 tokens 12s 47.42 t/s

Prompt 2: Output 864 tokens 17s 50.10 t/s

Prompt 3: Output 876 tokens 17s 49.32t/s

VRAM used: 11.0 GB

#### Observations

It is an interesting boost in performances, about 40% more in throughput. This is still an important improvement from a user experience point of view. Best results were with 2 way MTP, above it does not bring more throughput.

## Qwen 3 Coder 30B A3B Instruct

```shell
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

Prompt 1: Output 264 tokens 3.1s 83.84 t/s

Prompt 2: Output 279 tokens 3.4s 81.49 t/s

Prompt 3: Output 296 tokens 3.7s 79.78 t/s

VRAM used: 11.5 GB

The coding model maintained around 80 tokens/s while producing relatively concise responses, making it particularly pleasant for programming tasks.

## Qwen 3.6 35B A3B

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

Prompt 1: Output 495 tokens 5.5s 90.08 t/s

Prompt 2: Output 1224 tokens 14s 82.96 t/s

Prompt 3: Output 685 tokens 8.1s 84.46 t/s

VRAM used: 11.2 GB

### Observations

This model offers an excellent compromise between reasoning capability and generation speed. Despite its larger size, it consistently maintained over 80 tokens/s while remaining comfortably within the 16 GB VRAM budget.

### MTP Variant

Prompt 1: Output 729 tokens 5.6s 130.87 t/s

Prompt 2: Output 721 tokens 5.1s 142.11 t/s

Prompt 3: Output 778 tokens 5.7s  137.68 t/s

VRAM used: 12.5 GB

#### Observations

It is a big improvement in throughput, without the MTP it is already very fast. There is a good 1 GB increase due to the 2 ways MTP. About 40 to 60 % throughput increase, this is big. No further improvement in throughput above 2 ways.

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

Prompt 1: Output 698 tokens 6.7s 104.83 t/s

Prompt 2: Output 796 tokens 7.7s 103.90 t/s

Prompt 3: Output 856 tokens 8.3s  103.38 t/s

VRAM used: 11.8 GB

## Ornith 1.0 9B

I am interested in this smaller model in the case an agent runs some cmd lines for instance. It is not really for coding, even it is a nice model. A K4_M variant is used, memory size is not an issue.

```bash
llama-server.exe 
-m ..\Ornith-1.0-9B-Q4_K_M.gguf 
--ctx-size 32768 
--temp 0.6 --top-p 0.95 --top-k 20 
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1 
```

Prompt 1: Output 433 tokens 6.2s 69.39 t/s

Prompt 2: Output 1081 tokens 15s 68.75 t/s

Prompt 3: Output 775 tokens 11s  68.30 t/s

VRAM used: 6.8 GB

### MTP Variant

The variant used is provided by protoLabsAI [Link](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF?show_file_info=Ornith-1.0-9B-MTP-Q4_K_M.gguf)

```bash
llama-server.exe 
-m ..\Ornith-1.0-9B-MTP-Q4_K_M.gguf 
--ctx-size 32768 
--temp 0.6 --top-p 0.95 --top-k 20 
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1 
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompt 1: Output 430 tokens 5.2 s 82.69 t/s

Prompt 2: Output 421 tokens 4.7 s 89.09 t/s

Prompt 3: Output 268 tokens 3.2 s  84.16 t/s

VRAM used:  7.3 GB

Good improvement on the throughput and the answers are short.

#### Second run with --spec-draft-n-max 2

Prompt 1: Output 571 tokens 6.6 s 86.66 t/s

Prompt 2: Output 1196 tokens 12 s 98.78 t/s

Prompt 3: Output 801 tokens 8.9 s  89.79 t/s

VRAM used:  7.4 GB

An another small improvement on the throughput, the answers are a bit longer.

*NB*: I tried above 2 and there is a diminishing return. So 2 seems to be the best candidate.

## Kwaipilot_KAT-Coder-V2.5-Dev

A Qwen 35B A3B variant with better benchmarks on agentic coding, which is the interesting part where the base model has some weaknesses.

*NB*: The bartowski variant is used, there is no unsloth implementation at the time of this test.

```bash
llama-server.exe 
-m ..\Kwaipilot_KAT-Coder-V2.5-Dev-IQ2_XXS.gguf 
--ctx-size 32768 --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1 --cache-type-k q4_0 --cache-type-v q4_0
```

Prompt 1: Output 349 tokens 4.3 s 81.29 t/s

Prompt 2: Output 609 tokens 7.4 s 82.00 t/s

Prompt 3: Output 430 tokens 4.4 s  97.85 t/s

VRAM used:  10.5 GB

# Conclusions

The RTX 5060 Ti 16 GB proved to be an excellent entry point for local LLM inference. Sparse models in the 25–35B class now run at speeds that make interactive usage genuinely comfortable (slightly above 50 tokens/s), to super comfortable when the MTP is enabled.

Compared with the MacBook Air M4, generation throughput improved by roughly three to five times depending on the model, which fundamentally changes the user experience. Instead of waiting for responses, local agents become practical for everyday work.

Perhaps the biggest surprise was that even the Qwen 3.6 27B reasoning model remained usable at around 34 tokens/s and with MTP around 45 tokens/s. While not as fast as the sparse A3B models, it is responsive enough that reasoning no longer feels like a bottleneck.

Overall, these results suggest that 16 GB GPUs have become a practical sweet spot for local inference. A single mid-range consumer GPU is now sufficient to run several state-of-the-art sparse models at highly interactive speeds.

# ChangeLog

27/07/2026:

- Initial test

28/07/2026:

- MTP Tests

30/07/2026:

- Add Ornith-1.0-35B

01/08/2026:

- Add Ornith-1.0-9B

02/08/2026:

- Add Ornith-1.0-9B MTP Variants
- Add Kwaipilot_KAT-Coder-V2.5-Dev