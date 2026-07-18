# Practical Local LLM Performance on a MacBook Air M4 (July 2026)

Tests conducted in July 2026.

This report is not intended to compare model quality or reasoning ability. The goal is to evaluate which GGUF models are practically usable on a MacBook Air M4 with 24 GB of unified memory, focusing on inference speed, memory usage, and overall user experience.

As a reference point, I currently use Gemma 4 26B A4B with a 32K-token context window. During extended sessions, generation typically starts at around 30 tok/s before gradually decreasing toward 20 tok/s due to the combined effects of context growth and thermal throttling.

Update 18/7/2026: I use also Qwen 3 Coder 30B A3B Instruct, it provides shorter and faster answers. Sessions starts around 40 tok/s and gradually decreasing also.

Because models generate different numbers of tokens, the **later** prompts are sometimes run on a hotter machine than others, and longer generations can induce more thermal throttling.

This is a **practical inference benchmark**, not a synthetic one. The objective is to measure the experience of using these models interactively on a fanless laptop under realistic conditions, including long generations and thermal throttling.

This report focuses on generation throughput. Prompt processing (prefill) was not measured separately.

Note that Apple Silicon uses unified memory shared between CPU and GPU, so the reported memory figures represent total system memory usage rather than dedicated GPU VRAM usage.

*Important*: all results represent sustained fanless operation. A MacBook Air may initially outperform these numbers but will gradually reduce clocks during long generations.

## Setup

- MacBook Air M4 24 GB
- macOS Tahoe 26.5.2
- llama.cpp b8833 (later tested with llama-b9940 - no regression)
- prism llama.cpp b9581 (for Prism Bonsai 27B)

## Executive Summary

After testing several recent GGUF models on a MacBook Air M4 (24 GB), I found that:

* **Gemma 4 26B A4B** offers one of the best overall balance of responsiveness, memory usage, and perceived capability, delivering around 30 tok/s despite its large size.
* **Qwen 3.5 9B** provides the best dense-model balance between speed and capability.
* **Qwen 3 14B** is noticeably slower because it is a dense model: all 14B parameters participate in every token generation, unlike MoE models where only a subset of parameters are activated.
* **Qwen 3.6 27B** exceeds the practical memory limits of a 24 GB Air and swaps heavily without KV cache quantification reduction.
* **LFM 8B A1B** demonstrates how efficient MoE models can be, sustaining more than 60 tok/s.
* **Prism Bonsai 27B** fits comfortably within memory limits, but it is too slow to be usable with about 10 to 6 tok/s.
* **Qwen 3 Coder 30B A3B Instruct** was one of the fastest large models tested, combining high throughput (~37 tok/s) with strong coding-oriented behavior and low verbosity.
* Quantized KV cache saves 2–3 GB of memory but slightly reduced throughput in these tests.

Reported memory usage refers to total system memory consumption, including background applications such as VS Code, Terminal, Google Chrome (with a few tabs), and Safari.


| Model                         | Avg tok/s   | Memory       | Practical on 24 GB? |
| ----------------------------- | ----------- | ------------ | ------------------- |
| Gemma 4 26B A4B               | \~30        | 22 GB        | ⭐⭐⭐⭐⭐          |
| Qwen 3.5 9B                   | \~15        | Moderate     | ⭐⭐⭐⭐☆          |
| Qwen 3 14B                    | \~10        | High         | ⭐⭐⭐☆☆          |
| Gemma 4 12B                   | \~11        | Moderate     | ⭐⭐⭐☆☆          |
| LFM 8B A1B                    | \~62        | Moderate     | ⭐⭐⭐⭐☆          |
| Qwen 3.6 27B                  | \~5         | 22 GB        | ⭐⭐☆☆☆          |
| Prism Bonsai 27B              | \~6 to ~10 | 17GB to 19GB | ⭐⭐⭐☆☆          |
| Qwen 3 Coder 30B A3B Instruct | \~37        | 21GB         | ⭐⭐⭐⭐⭐          |

# LLM Models

All those models were sourced from Unsloth.ai, except for a few.

- Gemma 4 26B A4B
  - gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf [Link](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf)
- Qwen 3 14B
  - Qwen3-14B-Q4_K_M.gguf [Link](https://huggingface.co/unsloth/Qwen3-14B-GGUF?show_file_info=Qwen3-14B-Q4_K_M.gguf)
- Qwen 3.6 27B
  - Qwen3.6-27B-UD-IQ2_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF?show_file_info=Qwen3.6-27B-UD-IQ2_XXS.gguf)
- Qwen 3.5 9B
  - Qwen3.5-9B-Q4_K_M.gguf [Link](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q4_K_M.gguf)
- Gemma 4 12B
  - gemma-4-12b-it-Q4_K_M.gguf [Link](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF?show_file_info=gemma-4-12b-it-Q4_K_M.gguf)
- LFM 2.5 230 MB
  - LFM2.5-230M-UD-Q8_K_XL.gguf [Link](https://huggingface.co/unsloth/LFM2.5-230M-GGUF?show_file_info=LFM2.5-230M-UD-Q8_K_XL.gguf)
- LFM 2.5 1.2B
  - LFM2.5-1.2B-Instruct-Q4_K_M.gguf [Link](https://huggingface.co/unsloth/LFM2.5-1.2B-Instruct-GGUF/blob/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf)
- LFM 2.5 8B A1B
  - LFM2.5-8B-A1B-UD-Q4\_K\_M.gguf [Link](https://huggingface.co/unsloth/LFM2.5-8B-A1B-GGUF?show_file_info=LFM2.5-8B-A1B-UD-Q4_K_M.gguf)
- Prism Bonsai 27B
  - Ternary-Bonsai-27B-Q2_0.gguf [Link](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/blob/main/Ternary-Bonsai-27B-Q2_0.gguf)
  - Bonsai-27B-Q1_0.gguf [Link](https://huggingface.co/prism-ml/Bonsai-27B-gguf/blob/main/Bonsai-27B-Q1_0.gguf)
- Qwen 3 Coder 30B A3B Instruct
  - Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf [Link](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/blob/main/Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf)

## Prompts Used

- Prompt 1: I think that the 42 answer is also a sarcastic way to what question matters and the importance of a good question. What do you think
- Prompt 2: Can you enumerate such questions
- Prompt 3: I would say the usage of a llm is a bit like asking such kind of question

**Methodology:** Each model was given the same three prompts. Models were tested under similar interactive settings, with model-specific parameters adjusted when required for correct chat behavior. The reported token count is the number of output tokens actually generated by the model before it stopped naturally. Since different models produce different response lengths—especially reasoning models—the generation time is not directly comparable across models. The primary metric of interest is sustained generation throughput (tokens/second), while the output token counts illustrate how verbose each model is.

# Models

## Gemma 4 26B A4B

**Note:** Gemma 4 26B A4B is a Mixture-of-Experts (MoE) model. Although it contains approximately 26 billion parameters in total, only about 4 billion are active during each token generation, which explains its much higher throughput compared with similarly sized dense models.

### Default KV Cache, FA Off

````bash
./llama-server \
    -m ../gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```
````

Prompt 1: Output 1 284 tokens 40s 31.44 t/s

Prompt 2: Output 1 600 tokens 52s 30.20 t/s

Prompt 3: Output 1 419 tokens 47s 29.59 t/s

Memory usage: 21.7 GB

### 4-bit KV Cache, FA On

```bash
./llama-server \
    -m ../gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
     -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0
```

Prompt 1: Output 1 423 tokens 48s 29.39 t/s

Prompt 2: Output 1 779 tokens 1min 4s 27.57 t/s

Prompt 3: Output 1 527 tokens 55s 27.32 t/s

Memory usage: 19.2 GB

### Observations

Using a 4-bit KV cache reduced throughput in these tests.

Reduced memory usage by approximately 2 GB.

### Default KV Cache, FA Off, Reasoning Off

````bash
./llama-server \
    -m ../gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --reasoning off
```
````

Prompt 1: Output 518 tokens 15s 32.46 t/s

Prompt 2: Output 769 tokens 25s 30.63 t/s

Prompt 3: Output 690 tokens 22s 30.14 t/s

Memory usage: 21.7 GB

## 17/07/2026 : 4-bit KV Cache, FA On, Reasoning Off, Parallel Seq 1

```bash
./llama-server \
    -m ../gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
     -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --reasoning off \
    -np 1
```

It is a small additional modification to reduce a bit the memory pressure by reducing the number of parallel sequences to 1 (sometimes it is 4 by default, so better to force it).

Now the prompts are run in a small script using curl, and chained directly

Prompt 1: Output 716 tokens, 31.25 tok/s

Prompt 2: Output 748 tokens, 29.34 tok/s

Prompt 3: Output 718 tokens, 28.53 tok.s

Memory Usage: 20.5 GB

### Variant fa off

No real change

Prompt 1: Output 578 tokens, 31.60 tok/s

Prompt 2: Output 887 tokens, 29.46 tok/s

Prompt 3: Output 767 tokens, 28.55 tok.s

Memory Usage: 20.0 GB

# Qwen 3 14B

### Default KV Cache, FA Off

**NB**: *It is totally normal that it is slower* than the Gemma 4, because ALL 14B parameters are active, not only 4B at a time.

```bash
./llama-server \
    -m ../Qwen3-14B-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

Prompt 1: Output 388 tokens 35s 10.96 t/s

Prompt 2: Output 1 243 tokens 2min 7s 9.76 t/s

Prompt 3: Output 848 tokens 1min 32s 9.22 t/s

Memory usage: 21.5 GB

### 4-bit KV Cache, FA On

````bash
./llama-server \
    -m ../Qwen3-14B-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
````

Prompt 1:Output  274 tokens 25s 10.74 t/s

Prompt 2: Output 889 tokens 1min 32s 9.63 t/s

Prompt 3: Output 812 tokens 1min 47s 7.57 t/s

Memory usage: 18 GB

### FA Off

Prompt 1: Output 352 tokens 33s 10.64 t/s

Prompt 2: Output 942 tokens 1min 30s 10.37 t/s

Prompt 3: Output 706 tokens 1min 8s 10.26 t/s

### Observations

Using a 4-bit KV cache reduced throughput in these tests.

Reduced memory usage by approximately 3 GB.

Thermal throttling made repeated measurements less consistent.

## Qwen 3.6 27B

### Default KV Cache, FA Off

````bash
./llama-server \
    -m ../Qwen3.6-27B-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
````

Prompt 1: Output 559 tokens 1min 29s 6.27 t/s

Prompt 2: Output 655 tokens 2min 34s 4.23 t/s

Prompt 3: Output 750 tokens 2min 13s 5.63 t/s

Memory usage: 21.5 GB

**Significant swapping occurs, more than 4 GB**

### 4-bit KV Cache, FA On

````bash
./llama-server \
    -m ../Qwen3.6-27B-UD-IQ2_XXS.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
     -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
````

Prompt 1: Output 539 tokens 1min 24s 6.36 t/s

Prompt 2: Output 1 039 tokens 4min 9s 4.16 t/s

Prompt 3: Output 889 tokens 2min 28s 5.97 t/s

Memory usage: 20 GB

No swapping

### Observations

Memory usage exceeds the available physical memory, resulting in swapping when the KV cache 4 bit is not enabled

## Qwen 3.5 9B

### Default KV Cache, FA Off

````bash
./llama-server \
    -m ../Qwen3.5-9B-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
````

Prompt 1: Output  555 tokens 35s 15.73 t/s

Prompt 2: Output 958 tokens 1min 15.76 t/s

Prompt 3: Output 808 tokens 52s 15.32 t/s

Memory usage remained well within available RAM.

### 4-bit KV Cache, FA On

````bash
./llama-server \
    -m ../Qwen3.5-9B-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
     -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
````

Prompt 1: Output 722 tokens 46s 15.58 t/s

Prompt 2: Output 983 tokens 1min 3s 15.45 t/s

Prompt 3: Output 991 tokens 1min 4s 15.31 t/s

Memory usage remained well within available RAM.

## Gemma 4 12B

### Default KV Cache, FA Off

````bash
./llama-server \
    -m ../gemma-4-12b-it-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```
````

Prompt 1: Output 1 394 tokens 1min 56s 11.94 t/s

Prompt 2: Output 1 594 tokens 2min 30s 10.57 t/s

Prompt 3: Output 1 511 tokens 2min 17s 10.97 t/s

Thermal throttling observed.

### Default KV Cache, FA Off, Disable Reasoning

````bash
./llama-server \
    -m ../gemma-4-12b-it-Q4_K_M.gguf \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --reasoning off
```
````

Prompt 1: Output 585 tokens 47s 12.21 t/s

Prompt 2: Output 726 tokens 1min 1s 11.73 t/s

Prompt 3: Output 719 tokens 1min 2s 11.57 t/s

## LFM 2.5 230M

I was curious to try a tiny model like this one. 

*NB*: Of course, it is extremely limited and It produces mostly nonsensical answers. It is not built for that.

### Default KV Cache, FA Off

````bash
./llama-server -m ../LFM2.5-230M-UD-Q8_K_XL.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --seed 3407 \
    --prio 2 \
    --temp 0.1 \
    --top-k 50 \
    --top-p 0.1 \
    --repeat-penalty 1.05 \
    --jinja
```
````

Prompt 1: Output 191 tokens 1.1s 180.05 t/s

Prompt 2: Output 378 tokens 2.2s 170.25 t/s

Prompt 3: Output 765 tokens 4.4s 175.64 t/s

It is extremely fast, although its capabilities are understandably limited.

## LFM 2.5 1.2B Instruct

I was even more curious to try a small model after testing the tiny one.

*NB*: Clearly it has limitations and this is not its main purpose.

### Default KV Cache, FA Off

````bash
./llama-server -m ../LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --seed 3407 \
    --prio 2 \
    --temp 0.1 \
    --top-k 50 \
    --top-p 0.1 \
    --repeat-penalty 1.05 \
    --jinja

```
````

Prompt 1: Output 201 tokens 2.2s 89.48 t/s

Prompt 2: Output 555 tokens 6.3s 88.56 t/s

Prompt 3: Output 186 tokens 2.2s 86.42 t/s

### Observations

The 1.2B model is around half the speed of the 230M version but remains extremely fast at ~87 t/s.

This shows the interesting trade-off of small models: adding parameters quickly reduces throughput, but can significantly improve instruction following and usefulness.

Compared with larger local assistants, these models are not competing on deep reasoning capability, but they offer near-instant responses and very low resource usage.

## LFM 2.5 8B A1B

What does this model offer?

### Default KV Cache, FA Off

````bash
./llama-server -m ../LFM2.5-8B-A1B-UD-Q4_K_M.gguf \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --temp 0.2 \
    --top-k 80 \
    --repeat-penalty 1.05 \
    --top-p 0.1 \
    --jinja
```
````

Prompt 1: Output 659 tokens 10s 65.06 t/s

Prompt 2: Output 1 193 tokens 19s 61.75 t/s

Prompt 3: Output 929 tokens 15s 61.59 t/s

### Observations

This is a reasoning-only model.

Despite being an 8B total parameter model, the MoE architecture keeps inference extremely fast on Apple Silicon.

At around 62 t/s while generating reasoning tokens, it demonstrates a very different trade-off compared with dense models.

The model is not intended to replace larger reasoning models, but it offers an interesting middle ground:

- much faster than dense 9B–14B models
- more capable than tiny edge models
- reasoning enabled by design

## Prism Bonsai 27B

I am curious about this one, it claims to retain the majority of the dense Qwen 3.6 27B while using very aggressive compression.

```bash
llama.cpp-prism-b9591-62061f9/build/bin/llama-server
 -m ../Ternary-Bonsai-27B-Q2_0.gguf \
 --ctx-size 32768 \
--temp 0.7 \
--top-p 0.95 \
--top-k 20 \
-ngl 99 \
-fa on \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
-np 1 \
--port 8080
```

Prompt 1: Output 1523 tokens 10.13 t/s - run 1

Prompt 2: Output 2341 tokens 8.04 t/s - run 1

Prompt 3: Output 2449 tokens 3.81 t/s - run 1

Prompt 1: Output 1799 tokens 9.43 t/s - run 2

Prompt 2: Output 1978 tokens 6.87 t/s - run 2

Prompt 3: Output 2242 tokens 6.32 t/s - run 2

Memory usage is slightly lower about 18.77 GB to 19.47GB at the end of the last prompt, which is pretty good.

The token per seconds is getting slightly lower at the end of run 1, for sure thermal throttling, but may be something else. No swapping, nothing like that. Retried a second time.

I would say it is not really usable, may be on a Pro or Max model thanks to the additional  memory bandwidth. It fits without any issue, there is enough room to put large context too. Still, this is really interesting because if there are MOE variants.

### Variant 1 Bit

````bash
llama.cpp-prism-b9591-62061f9/build/bin/llama-server
 -m ../Bonsai-27B-Q1_0.gguf \
 --ctx-size 32768 \
--temp 0.7 \
--top-p 0.95 \
--top-k 20 \
-ngl 99 \
-fa on \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
-np 1 \
--port 8080
````

Prompt 1: Output 1868 tokens 14.56 t/s

Prompt 2: Output 2471 tokens 11.38 t/s

Prompt 3: Output 2506 tokens 10.24 t/s

Memory usage is slightly lower about 16.5 GB to 17.5 GB at the end of the last prompt, which is pretty good.

Not yet usable but it is slight improvement. The memory usage is really good too.

## Qwen 3 Coder 30B A3B Instruct

It is a well-known model, and it does not disappoint. It is fast, it is accurate, it has low verbosity.

```bash
llama-b9940/llama-server \
-m ../Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf \
--jinja \
-ngl 99 \
--ctx-size 32768 \
--temp 0.7 \
--min-p 0.0 \
--top-p 0.80 \
--top-k 20 \
--repeat-penalty 1.05 \
-fa on \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
-np 1 \
--port 8080
```

Prompt 1: Output 272 tokens 40.48 t/s

Prompt 2: Output 341 tokens 39.26 t/s

Prompt 3: Output 324 tokens 38.02 t/s

I was happily surprised by the speed and the limited amount of tokens. I run it a few times to be sure, this is not a cherry pick and no it is very consistent.

# Sustained Performance

The MacBook Air is fanless, so sustained workloads behave differently from actively cooled systems:

* Initial throughput around 30 tok/s
* Drops to roughly 20 tok/s during long generations
* Fanless chassis eventually throttles
* Short chats remain very responsive

# Conclusions

* Gemma 4 26B A4B provided the best overall balance of responsiveness, memory usage, and perceived capability in these tests.
* Qwen 3 14B offers stronger dense-model behavior but at roughly one-third the throughput.
* Qwen 3.6 27B fits in memory with some KV cache optimization but too slow to be usable.
* Qwen 3.5 9B provides an excellent balance between capability and speed.
* LFM 8B A1B demonstrates how efficient Mixture-of-Experts models can be on Apple Silicon.
* Prism Bonsai 27B fits well in the memory but the memory bandwidth requirement makes it not useful in practice with about 10 to 6 tok/s.
* KV cache quantization reduces memory usage by 2–3 GB but decreases throughput in these tests.
* On Apple Silicon, MoE models change the usual relationship between model size and speed. A 30B MoE model can outperform smaller dense models because only a fraction of parameters are active during generation.
* A fanless 24 GB M4 MacBook Air can realistically run surprisingly large MoE models, while dense models above ~14B become bandwidth- and memory-limited.
* The fanless MacBook Air remains highly capable for local inference, although sustained workloads eventually trigger thermal throttling.

These results should be viewed as practical guidance rather than absolute rankings, since model capabilities, quantization formats, inference engines, and future llama.cpp optimizations will continue to evolve.

## ChangeLog

17/07/2026:

* Update flags to reduce number of parallel sequences to decode (-np 1)
* Add Prism Bonsai 27B

18/07/2026:

* Add Qwen 3 Coder 30B A3B Instruct
