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

*NB*: Numbers are for **very aggressive 2 bits quantification and KV cache 4 bits quantification**. Except for 9B models using 4 bits quantification and default KV cache quantification.


| Model                             | Avg tok/s | VRAM    | Practical on 16 GB? |
| --------------------------------- | --------- | ------- | -------------------- |
| Gemma 4 26B A4B                   | \~90      | 10.9 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 27B                      | \~30      | 10.4 GB | ⭐⭐⭐☆☆           |
| Qwen 3 Coder 30B A3B Instruct     | \~75      | 11.5 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 35B A3B Instruct         | \~80      | 11.2 GB | ⭐⭐⭐⭐⭐           |
| Qwen 3.6 27B MTP                  | \~45      | 11.0 GB | ⭐⭐⭐⭐☆           |
| Qwen 3.6 35B A3B Instruct MTP     | \~120     | 12.5 GB | ⭐⭐⭐⭐⭐           |
| Ornith 1.0 35B                    | \~90      | 11.8 GB | ⭐⭐⭐⭐⭐           |
| Ornith 1.0 9B                     | \~60      | 6.8 GB  | ⭐⭐⭐⭐☆           |
| Ornith 1.0 9B MTP                 | \~80      | 7.4 GB  | ⭐⭐⭐⭐⭐           |
| KAT-Coder-V2.5-Dev                | \~80      | 10.5 GB | ⭐⭐⭐⭐⭐           |
| Qwythos-9B-Claude-Mythos-5-1M MTP | \~80      | 7.7 GB  | ⭐⭐⭐⭐⭐           |

## MTP

The RTX 5060 TI responds well to MTP with 2 ways.

There is at least 40% throughput increase on Qwen 3.6 27B Dense and Qwen 3.6 35B A3B, it is a major improvement:

- For the dense 27B the user experience is moving from experiments to the low confort zone.
- For the sparse 35B the throughput is such that agentic job is possible locally.

The MTP throughput increase is good enough, to start to thing about tuning the model to improve 'knowledge' with less quantification, or having more 'precision' in the KV cache.

I hope we will see also more smaller dense models with MTP.

## Offloading to CPU Memory

The model is too big to fit the GPU VRAM.
Offloading works reasonability well for MOE models.
For this test we are using non MTP and MTP variants with KV cache 4 bits quantification and small 32K tokens : 
- Qwen3.6-35B-A3B-UD-IQ4_NL_MTP [Link](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-IQ4_NL.gguf)
- Qwen3.6-35B-A3B-UD-IQ4_NL [Link](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-IQ4_NL.gguf)


| CPU MoE |    VRAM |  MTP 2 Way + KV Q4 |
| ------: | ------: | ---------------:   |
|     14  | 13.5 GB | **65–67 tok/s**    |
|     16  | 12.8 GB | **59–67 tok/s**    |
|     18  | 12.1 GB | **58–61 tok/s**    |


| CPU MoE |    VRAM |  No MTP + KV Q4  |
| ------: | ------: | ---------------: |
|      12 | 13.5 GB | **65–66 tok/s**  |
|      14 | 12.7 GB | **62–63 tok/s**  |
|      16 | 12.1 GB | **59–60 tok/s**  |
|      18 | 11.3 GB | **57–59 tok/s**  |


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
* Qwythos-9B-Claude-Mythos-5-1M

  * empero-ai provided model: Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf  [Link](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF?show_file_info=Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf)

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

### Offloading Memory to CPU

In this experiment I consider the effect on the throughput when the model cannot fit on the GPU.

We use the model Qwen3.6-35B-A3B-GGUF with this quantification variant: IQ4_NL from unsloth [Link](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-IQ4_NL.gguf). The model is about 18GB without any KV cache.

### Offloading 16

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

Prompt 1: Output 486 tokens 8.2s 59.49 t/s

Prompt 2: Output 702 tokens 5.1s 59.75 t/s

Prompt 3: Output 784 tokens 13s  59.56 t/s

VRAM used: 12.5 GB

### Offloading 12

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf 
--ctx-size 32768 
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  
--n-cpu-moe 12
```

Prompt 1: Output 653 tokens 9.9s 66.24 t/s

Prompt 2: Output 1090 tokens 16s 65.77 t/s

Prompt 3: Output 924 tokens 14s  65.60 t/s

VRAM used: 14 GB

### Offloading 20

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf 
--ctx-size 32768 
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  
--n-cpu-moe 20
```

Prompt 1: Output 487 tokens 8.9s 54.69 t/s

Prompt 2: Output 874 tokens 15s 55.02 t/s

Prompt 3: Output 696 tokens 12s 55.13 t/s

VRAM used: 11.2 GB

### Offloading 24

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf 
--ctx-size 32768 
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  
--n-cpu-moe 24
```

Prompt 1: Output 754 tokens 15s 49.84 t/s

Prompt 2: Output 963 tokens 19s 50.48 t/s

Prompt 3: Output 1031 tokens 20s 50.80 t/s

VRAM used: 9.7 GB

#### Observations

The prefill throughput is impacted so there is additional latency at this level.

All previous tests there is no KV cache quantification tuning done.

Offloading 16 case: we reach about 59 tokens per sec. This is not super fast, but the user experience is still very good.

Offloading 12: the throughput is a bit higher with 65 tokens per sec, but the VRAM used is already 14GB, so there is not a lot of margin for a small increase in throughput.

Offloading 20: the throughput is in the 55 tokens per sec. The user experience is still good and there is only 11.2 GB of VRAM used, it is an interesting case.

Offloading 24: the throughput goes down to about 50 tokens per sec. The user experience is ok and the VRAM used is the 9.7 GB, it leaves a lot of head room to experiment.

I would say the limiting factor in such cases are the PCI Xpress bus and the CPU/RAM Memory Bandwidth.

### Offloading Memory to CPU/MTP

I took the MTP variant of the previous model [Link](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-IQ4_NL.gguf)

#### Offloading 16 - 1 Way

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

Prompt 1: Output 689 tokens 10s 62.70 t/s

Prompt 2: Output 966 tokens 14s 64.96 t/s

Prompt 3: Output 762 tokens 11s  64.94 t/s

VRAM used: 13.3 GB

#### Offloading 16 - 2 Ways

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

Prompt 1: Output 446 tokens 6.7s 66.10 t/s

Prompt 2: Output 908 tokens 13s 65.77 t/s

Prompt 3: Output 781 tokens 12s  64.38 t/s

VRAM used: 13.3 GB

Small improvement in throughput with additional memory usage.

#### Offloading 16 - 2 Ways - KV 4 bits

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

Prompt 1: Output 514 tokens 8.4s 61.27 t/s

Prompt 2: Output 933 tokens 13s 68.50 t/s

Prompt 3: Output 1044 tokens 15s  65.90 t/s

VRAM used: 12.9 GB

Similar throughput larger context.

#### Observations

MTP in this case improved the throughput by 10% with about 800/900 MB of additional VRAM. From a user experience it is noticeable, but not a big deal.

#### Additional measurements

##### Tests done after Suspend


| CPU MoE |    VRAM |      MTP 2 Way + KV Q4 |
| ------: | ------: | ---------------: |
|      14 | 13.5 GB | **55–57 tok/s** | 
|      16 | 12.8 GB | **52–57 tok/s** | 
|      18 | 12.1 GB | **52–54 tok/s** |


| CPU MoE |    VRAM |   No MTP + KV Q4 |
| ------: | ------: | ---------------: |
|      12 | 13.5 GB | **53–54 tok/s** |
|      14 | 12.8 GB | **52–53 tok/s** |
|      16 | 12.0 GB |   **\~50 tok/s** |
|      18 | 11.3 GB | **48–49 tok/s** |


##### Tests done after fresh Reboot

**Important:** Windows suspend/resume can affect inference performance on this system. Benchmark runs should be performed after a fresh reboot to ensure reproducible GPU/PCIe performance.


| CPU MoE |    VRAM |  MTP 2 Way + KV Q4 |
| ------: | ------: | ---------------:   |
|     14  | 13.5 GB | **65–67 tok/s**    |
|     16  | 12.8 GB | **59–67 tok/s**    |
|     18  | 12.1 GB | **58–61 tok/s**    |


| CPU MoE |    VRAM |  No MTP + KV Q4  |
| ------: | ------: | ---------------: |
|      12 | 13.5 GB | **65–66 tok/s**  |
|      14 | 12.7 GB | **62–63 tok/s**  |
|      16 | 12.1 GB | **59–60 tok/s**  |
|      18 | 11.3 GB | **57–59 tok/s**  |

##### Observations

For CPU-offloaded Qwen3.6-35B-A3B IQ4\_NL, MTP is not a game changer. It provides a modest throughput improvement, but its main cost is additional VRAM rather than a dramatic increase in generation speed.

### Offloading with Context Variants

#### 24KTok Context

```bash
llama-server.exe
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
--ctx-size 32768
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00
--repeat-penalty 1.00 --presence-penalty 1.5
--chat-template-kwargs "{\"enable_thinking\":false}"
-fa on -np 1  --n-cpu-moe 16
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 24608 tokens 28s 859.18 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 2,155 tokens 40s 52.67 t/s

#### 48KTok Context

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
 --ctx-size 65536
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  --n-cpu-moe 16 
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 49184 tokens 58s 843.86 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 136 tokens 2.9s 47.46 t/s

### 96KTok Context

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
 --ctx-size 131072
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  --n-cpu-moe 16 
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 97825 tokens 2min 2s 799.61 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 97 tokens 2.4s 39.72 t/s

#### 200KTok Context

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
 --ctx-size 262144
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  --n-cpu-moe 16 
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 195105 tokens 4min 34s 711.45 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 7,230 tokens 4min 8s 29.06 t/s

VRAM Used: 14.2 GB at the end of the inference.

#### 200KTok Context - 15 Layers

```bash
llama-server.exe 
-m ..\Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
 --ctx-size 262144
--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.00 
--repeat-penalty 1.00 --presence-penalty 1.5  
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1  --n-cpu-moe 15 
--cache-type-k q4_0 --cache-type-v q4_0
```

Prefill: 195105 tokens 4min 28s 727.78 tokens/s

Prompt: Count the number of repetitions in this file

Inference: 204 tokens 7.0s 29.19 t/s

VRAM Used: 14.3 GB at the end of the inference.

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

## Qwythos-9B-Claude-Mythos-5-1M

This is a variant of Qwen 3.5 post trained on an uncensored model.

Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M [Link](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF?show_file_info=Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf)

```bash
llama-server.exe 
-m ..\Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_Mgguf 
--ctx-size 32768 
--temp 0.6 --top-p 0.95 --top-k 20 
--repeat-penalty 1.05
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1 
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompt 1: Output 132 tokens 1.6 s 83.36 t/s

Prompt 2: Output 280 tokens 3.4 s 83.12 t/s

Prompt 3: Output 280 tokens 3.4 s  81.48 t/s

VRAM used:  7.5 GB

### Variant MTP 2

```bash
llama-server.exe 
-m ..\Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_Mgguf 
--ctx-size 32768 
--temp 0.6 --top-p 0.95 --top-k 20 
--repeat-penalty 1.05
--chat-template-kwargs "{\"enable_thinking\":false}" 
-fa on -np 1 
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompt 1: Output 182 tokens 1.6 s 75.30 t/s

Prompt 2: Output 473 tokens 5.6 s 83.79 t/s

Prompt 3: Output 319 tokens 4.0 s  80.51 t/s

VRAM used:  7.6 GB

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
- Add Qwythos-9B-Claude-Mythos-5-1M

03/08/2026

- Additional experiments with offloading to check throughput effects on MOE

04/08/2026

- Additional experiments with offloading with MOE and MTP