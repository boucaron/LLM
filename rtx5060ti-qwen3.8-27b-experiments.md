# Qwen 3.8 27B Experiments

Detailed experiments for the Qwen 3.8 27B model, part of the [RTX 5060 Ti local LLM benchmark](rtx5060ti-local-llm-benchmark.md).

Test Dates: 15 to 16 Aug 2026 (Unsloth Dynamic Quant V 2.0)

Test Dates: 19 to 21 Aug 2026 (Unsloth Dynamic Quant V 3.0)

I would say this is a model that nearly all local LLM nerds have been waiting for, and it does not disappoint.

**TL;DR:** Qwen 3.8 27B dense model on a 16 GB GPU: usable at ~35–57 t/s depending on quantization (IQ2_XXS → Q3_K_M) and MTP, with context scaling to 170K–192K once the KV cache is quantized and MTP keeps active parameters low. Unsloth Dynamic Quant v3.0 (`UD-IQ3_S`) + MTP is the sweet spot: ~40–44 t/s at 32K within ~13–16 GB VRAM. Full per-variant measurements and commands below.

I used the following Unsloth variants (15 to 17 Aug 2026: **those models are no longer available**):

* Qwen3.8-27B-IQ4\_NL
* Qwen3.8-27B-Q3\_K\_M
* Qwen3.8-27B-Q3\_K\_S
* Qwen3.8-27B-UD-Q2_K_XL
* Qwen3.8-27B-UD-IQ2_XXS

19 Aug 2026: Unsloth pushed new variants using their *Dynamic Quant v3.0* (seems to preserve more the capabilities of the model) and the GGUF has the MTP separated from the model. So I am redoing a short series of experiments, because it means we can have more context for the same model.

So all the previous tests have the prefix **D2** for *Dynamic Quant V 2.0* and the new one **D3** for *Dynamic Quant V 3.0*.

I used the following variants (19 to 21 Aug 2026):

- Qwen3.8-27B-UD-IQ3_S (MTP included)
- MTP: mtp-Qwen3.8-27B-Q4_0.gguf

Thinking mode is enabled by default, and it can be disabled with `--reasoning off` (if not set, reasoning is on).

## D2 — Q3_K_M

I performed a few quick tests. I have not yet done extensive testing across different context sizes and model variants.

### No MTP

#### 32 K Context

```
llama-server
-m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 t/s

VRAM used: 14 GB

#### 64 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 65536 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 t/s

VRAM used: 14.8 GB

Prompts additional 20 K context: prefill about 720 t/s, inference 22 t/s

VRAM used @ 28.3 K : 14.8 GB ( 200 MB Offloaded)

Prompts additional 20 K context: prefill about 610 t/s, inference 19 t/s

VRAM used @ 53.5 K : 14.8 GB ( 200 MB Offloaded)

#### 96 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 98304 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 t/s

VRAM used: 15.5 GB (300 MB Offloaded) We are already on the edge

Prompts additional 40 K context: prefill about 630 t/s, inference 18 t/s

VRAM used @ 52.59 K : 15.6 GB ( 300 MB Offloaded)

Prompts additional 20 K context: prefill about 470 t/s, inference 16 t/s

VRAM used @ 77.48 K : 15.6 GB ( 300 MB Offloaded)

### MTP 1

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1
```

#### 32 K Context

Prompts with the 3 classic questions, one after another: 32 to 35 t/s

VRAM used: 14.5 GB

Prompts additional 20 K context: prefill about 700 t/s, inference 31 t/s

VRAM used @ 27.5 K : 14.8 GB

#### 64 K Context

Prompts with the 3 classic questions, one after another: 32 to 35 t/s

VRAM used: 15.5 GB (we are on the edge already, a bit of offloading)

Prompts additional 20 K context: prefill about 680 t/s, inference 30 t/s

VRAM used @ 31.6 K : 15.6 GB ( 200 to 400 MB Offloaded)

Prompts additional 20 K context: prefill about 570 t/s, inference 28 t/s

VRAM used @ 52.7 K : 15.6 GB ( 300 to 400 MB Offloaded)

### MTP 2

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

#### 32 K Context

Prompts with the 3 classic questions, one after another: 35 to 42 t/s

VRAM used: 14.7 GB

Prompts additional 20 K context: prefill about 700 t/s, inference 34 t/s

VRAM used @ 27.5 K : 15.0 GB

#### 64 K Context

```bash
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 65536 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 33 to 39 t/s

VRAM used: 15.6 GB (we are on the edge already, a bit of offloading)

Prompts additional 20 K context: prefill about 660 t/s, inference 33 t/s

VRAM used @ 31.3 K : 15.6 GB (400 to 500 MB offloading on CPU/Memory)

Prompts additional 20 K context: prefill about 550 t/s, inference 29 t/s

VRAM used @ 56.8 K : 15.6 GB (500 MB offloading on CPU/Memory)

### MTP 3

```
llama-server -m ..\Qwen3.8-27B-Q3_K_M.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 3
```

#### 32 K Context

Prompts with the 3 classic questions, one after another: 33 to 40 t/s (large swings in throughput)

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

The default thinking effort is **very high**, and it can consume a lot of tokens. Even for a relatively short question with a small context—around 1K tokens—I observed cases where the thinking and answer combined reached about **7.6K tokens**. That's substantial, so having enough context available is important.

Hopefully, a new release of llama.cpp is out, the reasoning effort can be tune easily. This is a nice feature, but the default effort may be excessive for many use cases.

In **chat/instruct mode**, latency is low and the response length is similar to **Qwen3.6 27B**, making it a good option for interactive use.

**MTP also seems to have improved compared with Qwen3.6.** The throughput gain is significant with MTP-1 and MTP-2, with **MTP-2 currently looking like the sweet spot**. MTP-3 does not provide a meaningful advantage over MTP-2 and shows larger throughput fluctuations, so I don't think it's worth spending more time testing its behavior at larger context sizes.

These results are still preliminary, but they already show that **Qwen3.8-27B is very usable in 3-bit**. The best quantization will, of course, depend on the target context size.

In these tests, **Qwen3.8-27B-Q3_K_M.gguf with MTP-2 is a very good fit for up to 64K context**. Inference starts at around **39 t/s** and drops to about **29 t/s** as the context approaches its limit. That's very usable, especially for such a dense model.

Without MTP, the model can be pushed to **96K context**, with throughput starting around **25 t/s** and dropping to about **16 t/s** near the limit. That's on the slow side, but still usable.

| Configuration | Context | Speed                | VRAM        | Verdict                              |
| ------------- | ------- | -------------------- | ----------- | ------------------------------------ |
| No MTP        | 32K     | \~25 t/s             | 14 GB       | Baseline                             |
| No MTP        | 64K     | 25 → 19 t/s         | 14.8 GB     | Good                                 |
| No MTP        | 96K     | 25 → 16 t/s         | 15.6 GB     | Usable, but slow                     |
| MTP-1         | 64K     | 32–35 → 28 t/s     | 15.6 GB     | Good                                 |
| **MTP-2**     | **64K** | **33–39 → 29 t/s** | **15.6 GB** | **Best balance**                     |
| MTP-3         | 32K     | 33–40 t/s           | 14.8 GB     | No clear benefit; more variable      |
| Chat+MTP-2    | 32K     | Similar to MTP-2     | Similar     | **Interactive use**                  |

Overall, **64K + MTP-2 looks like the best practical configuration** from these initial tests.

## D2 — Q3_K_S

The previous experiments are interesting, but the context is a bit too small. So we try another variant with less VRAM used to have a large context.

```bash
hf download hf://unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_S.gguf
```

We will start directly with 96 K context size.

### No MTP

#### 96 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_S.gguf
--ctx-size 98304 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 26 t/s

VRAM used:  14.5GB (300 MB Offloaded)

Prompts additional 40 K context: prefill about 620 t/s, inference 20 t/s

VRAM used @ 52.8 K : 14.6 GB ( 300 MB Offloaded)

Prompts additional 20 K context: prefill about 490 t/s, inference 18 t/s

VRAM used @  77.6 K : 14.6 GB ( 300 MB Offloaded)

#### 128 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_S.gguf
--ctx-size 131072 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 25 t/s

VRAM used:  15.3GB (300 MB Offloaded)

Prompts additional 80 K context: prefill about 530 t/s, inference 16 t/s

VRAM used @ 102.6 K : 15.3 GB ( 300 MB Offloaded)

### MTP 1

#### 96 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_S.gguf
--ctx-size 98304 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompts with the 3 classic questions, one after another: 35 t/s

VRAM used:  15.4GB (400 MB Offloaded)

Prompts additional 40 K context: prefill about 580 t/s, inference 27 t/s

VRAM used @ 55.5 K : 15.4 GB ( 400 MB Offloaded)

Prompts additional 20 K context: prefill about 490 t/s, inference 24 t/s

VRAM used @  80.1 K : 15.4 GB ( 300 MB Offloaded)

#### 112 K Context

This is the upper limit for this setup to achieve good throughput.

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

Prompts with the 3 classic questions, one after another: 35 t/s

VRAM used:  15.5GB (400 MB Offloaded)

Prompts additional 80 K context: prefill about 510 t/s, inference 23 t/s

VRAM used @ 101.8 K : 15.5 GB (400 MB Offloaded)

#### 128 K Context

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

Prompts with the 3 classic questions, one after another: 31 t/s

VRAM used:  15.6GB (800 MB Offloaded) => Already above the threshold

Prompts additional 80 K context: prefill about 255 t/s, inference 9 t/s

VRAM used @ 104.1 K : 15.6 GB (800 MB Offloaded)

### MTP 2

#### 96 K Context

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

Prompts with the 3 classic questions, one after another: 39 t/s

VRAM used:  15.5GB (400 MB Offloaded)

Prompts additional 40 K context: prefill about 590 t/s, inference 28 t/s

VRAM used @ 54.6 K : 15.5 GB ( 400 MB Offloaded)

Prompts additional 20 K context: prefill about 470 t/s, inference 23 t/s

VRAM used @ 79.49 K : 15.5 GB ( 400 MB Offloaded)

#### 112 K Context

I did not run on purpose the 128 K because we already know the issue.

```bash
llama-server
-m ..\Qwen3.8-27B-Q3_K_S.gguf
--ctx-size 112000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 36 t/s

VRAM used:  15.5GB (500 MB Offloaded)

Prompts additional 80 K context: prefill about 470 t/s, inference 19 t/s

VRAM used @ 101.6 K : 15.4 GB (700 MB Offloaded)

Interestingly, we have about 100 MB more offloaded at the start and about 300 MB more than the MTP 1 model. There is a slight reduction in inference all along, meaning too much communication with the CPU/MEM. We need to reduce a bit further to achieve good throughput.

#### 106 K Context

Since the last experiment already shows a bit too much offloading reducing the throughput, let's try to restore a bit.

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

Prompts with the 3 classic questions, one after another: 38 t/s

VRAM used:  15.5GB (500 MB Offloaded)

Prompts additional 80 K context: prefill about 500 t/s, inference 24 t/s

VRAM used @ 102.4 K : 15.4 GB (500 MB Offloaded)

We are still on the edge but the throughput reduction is lower when the context is nearly full.

### Observations

There is no real winner there for the MTP 2: for larger context it is better to stick to a MTP 1 that has the same throughput and allows to have a bit more context.

| Configuration | Context  | Burst t/s | \~100K t/s | VRAM        | Takeaway           |
| ------------- | -------- | --------- | ---------- | ----------- | ------------------ |
| **No MTP**    | 96K      | 26        | 18         | 14.6 GB     | Lower throughput   |
| **No MTP**    | 128K     | 25        | 16         | 15.3 GB     | Max context        |
| **MTP 1**     | 96K      | 35        | 24         | 15.4 GB     | Strong perf        |
| **MTP 1**     | **112K** | **35**    | **23**     | **15.5 GB** | **Best balance**   |
| **MTP 1**     | 128K     | 31        | 9          | 15.6 GB     | Not recommended    |
| **MTP 2**     | 96K      | 39        | 23         | 15.5 GB     | No clear advantage |
| **MTP 2**     | 112K     | 36        | 19         | 15.4 GB     | No clear advantage |
| **MTP 2**     | 106K     | 38        | 24         | 15.4 GB     | Edge case          |

**Best overall: MTP 1 with a 112K context.** It provides the best balance for this setup, reaching about **23 t/s at \~102K tokens** while keeping offloading relatively low. MTP 2 offers higher initial throughput, but its advantage disappears as the context grows.

## D2 — IQ4_NL (4-bit)

We are testing a 4-bit quantization that is really at the limit or above what the 16 GB can handle.

### No MTP

#### All layers (full offload)

```
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

Too big for the GPU.

Model exceeds VRAM; 700 MB spilled to CPU. Throughput: ~9 t/s.

#### 51 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 51
```

Throughput 8 to 9 t/s

VRAM Used 12.8 GB

#### 59 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59
```

Throughput 12 to 13 t/s

VRAM Used 14.6 GB

#### 62 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 62
```

Throughput 15 to 16 t/s

VRAM Used 15.3 GB

### MTP 1

#### 59 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59 --spec-type draft-mtp --spec-draft-n-max 1
```

Throughput 18 to 19 t/s

VRAM Used 15.2 GB

#### 62 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 62 --spec-type draft-mtp --spec-draft-n-max 1
```

Throughput 19 to 22 t/s

VRAM Used 15.6 GB

### MTP 2

#### 59 layers

```bash
llama-server -m ..\Qwen3.8-27B-IQ4_NL.gguf
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
-ngl 59 --spec-type draft-mtp --spec-draft-n-max 2
```

Throughput 20 to 22 t/s

VRAM Used 15.3 GB

### Observations

Pretty similar behaviour like the Qwen 3.6 27B in 4-bit quantization. MTP enables additional throughput, with the offloading and the MTP together we achieved around 20 t/s in burst mode.

| Configuration     | VRAM    | Throughput     | Notes                           |
| ----------------- | ------- | -------------- | ------------------------------- |
| All layers        | >16 GB  | \~9 t/s        | \~700 MB spilled to CPU         |
| 51 layers         | 12.8 GB | 8–9 t/s       | Stable, significant CPU offload |
| 59 layers         | 14.6 GB | 12–13 t/s     | Good balance                    |
| 62 layers         | 15.3 GB | 15–16 t/s     | Near VRAM limit                 |
| 59 layers + MTP 1 | 15.2 GB | 18–19 t/s     | Significant MTP gain            |
| 59 layers + MTP 2 | 15.3 GB | **20–22 t/s** | Best overall result             |
| 62 layers + MTP 1 | 15.6 GB | 19–22 t/s     | Highest VRAM usage              |

The **Qwen3.8 27B IQ4_NL** 4-bit quantization is at, or slightly beyond, the practical limit of a **16 GB GPU** at a 32K context. Fully offloading the model causes around **700 MB to spill into CPU memory**, reducing performance to roughly **9 t/s**.

Increasing GPU offloading improves throughput significantly: from **8–9 t/s at 51 layers** to **15–16 t/s at 62 layers**, while using approximately **15.3 GB of VRAM**.

**MTP provides the biggest additional performance improvement.** With 59 layers, throughput increases from 12–13 t/s to **18–19 t/s with MTP 1**, and **20–22 t/s with MTP 2**, while remaining around 15.3 GB VRAM.

Overall, the behaviour is very similar to the **Qwen3.6 27B 4-bit** tests. The combination of **aggressive GPU offloading + MTP** achieves around **20 t/s in burst mode**, but memory offloading remains close to the edge. As the context fills, throughput begins to degrade somewhat faster.

## D2 — Q2_K_XL (2-bit)

Of course, I could not resist to put more context in the VRAM.

We use the following models there from Unsloth:

- Qwen3.8-27B-UD-Q2_K_XL
- Qwen3.8-27B-UD-IQ2_XXS.gguf

There is a large gap in the size between the Q3_K_S and this one. I want to check how much more context I can put on the GPU.

### No MTP

#### 128 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 131072 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 31 t/s

VRAM used:  13.6GB (300 MB Offloaded)

Prompts additional 80 K context: prefill about 610 t/s, inference 18 t/s

VRAM used @ 102.7 K : 13.6 GB ( 300 MB Offloaded)

#### 192 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 196608 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

Prompts with the 3 classic questions, one after another: 31 t/s

VRAM used:  15.1GB (300 MB Offloaded)

Prompts additional 80 K context: prefill about 610 t/s, inference 18 t/s

VRAM used @ 102.1 K : 15.1 GB ( 300 MB Offloaded)

Prompts additional 40 K context: prefill about 410 t/s, inference 15 t/s

VRAM used @ 151.6 K : 15.1 GB ( 300 MB Offloaded)

#### 220 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 220000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

I did not run the 256 K because there is more than 1.1 GB offloaded to the CPU/MEM and it will kill the throughput. Instead, I iterated to find the threshold and maximize the context while keeping the throughput.

Prompts with the 3 classic questions, one after another: 30 t/s

VRAM used:  15.5GB (400 MB Offloaded)

Prompts additional 80 K context: prefill about 610 t/s, inference 18 t/s

VRAM used @ 102.7 K : 15.5 GB (400 MB Offloaded)

Prompts additional 80 K context: prefill about 360 t/s, inference 13 t/s

VRAM used @ 201.6 K : 15.5 GB (400 MB Offloaded)

### MTP 1

#### 128 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 131072 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompts with the 3 classic questions, one after another: 42 t/s

VRAM used:  14.2 GB (400 MB Offloaded)

Prompts additional 80 K context: prefill about 570 t/s, inference 23 t/s

VRAM used @ 102.1 K : 14.3 GB ( 500 MB Offloaded)

#### 172 K Context

There was already a bit too much offloading at 196 K, so I found a good candidate to not reduce too much the throughput.

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 172000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompts with the 3 classic questions, one after another: 39-42 t/s

VRAM used:  15.3 GB (500 MB Offloaded)

Prompts additional 80 K context: prefill about 565 t/s, inference 23 t/s

VRAM used @ 103.2 K : 15.3 GB ( 500 MB Offloaded)

Prompts additional 40 K context: prefill about 380 t/s, inference 20 t/s

VRAM used @ 152.8 K : 15.3 GB ( 500 MB Offloaded)

This is the sweet spot for this model with this GPU, we have a large context and it is still having good throughput when the context is nearly full.

### MTP 2

#### 128 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 131072 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 39-42 t/s

VRAM used:  14.3 GB (500 MB Offloaded)

Prompts additional 80 K context: prefill about 580 t/s, inference 24 t/s

VRAM used @ 100.5 K : 14.3 GB ( 500 MB Offloaded)

#### 172 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-Q2_K_XL.gguf
--ctx-size 172000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 39-42 t/s

VRAM used:  15.4 GB (500 MB Offloaded)

Prompts additional 80 K context: prefill about 575 t/s, inference 23 t/s

VRAM used @  101.1 K : 15.4 GB ( 500 MB Offloaded)

Prompts additional 40 K context: prefill about 375 t/s, inference 20 t/s

VRAM used @ 152.1  K : 15.4 GB ( 500 MB Offloaded)

This is also a sweet spot, there is no significant difference between MTP 1 and MTP 2 with such context. Recommend using MTP 1 in this context.

### Observations

| Mode      |  Context |   Generation t/s |      Prefill t/s |  VRAM GB | Offload MB | Notes                        |
| --------- | -------: | ---------------: | ---------------: | -------: | ---------: | ---------------------------- |
| No MTP    |     128K |               31 |            \~610 |      13.6 |        300 | Baseline                     |
| No MTP    |     192K |         31 → 15 |     \~610 → 410 |     15.1 |        300 | Throughput drops after \~150K |
| No MTP    |     220K |         30 → 13 |     \~610 → 360 |     15.5 |        400 | \~202K usable                |
| MTP 1     |     128K |         42 → 23 |            \~570 |     14.3 |   400–500 | Best at 128K                 |
| **MTP 1** | **172K** | **39–42 → 20** | **\~565 → 380** | **15.3** |    **500** | **Sweet spot**               |
| MTP 2     |     128K |     39–42 → 24 |            \~580 |     14.3 |        500 | Similar to MTP 1             |
| MTP 2     |     172K |     39–42 → 20 |     \~575 → 375 |     15.4 |        500 | Essentially same as MTP 1    |

#### Key Findings

* **No MTP:** 220K context is usable, reaching about **202K tokens** with \~15.5 GB VRAM and only \~400 MB offloaded. However, inference throughput falls to around **13 t/s** at very large context.
* **MTP 1:** **172K context is the sweet spot**, reaching \~153K tokens while maintaining around **20 t/s** inference and \~380 t/s prefill.
* **MTP 2:** Performance is almost identical to MTP 1 at large context sizes. At 172K, it reaches \~152K tokens with \~20 t/s inference.
* **128K context:** MTP provides a significant generation-speed improvement, reaching roughly **42 t/s** for the initial benchmark versus **31 t/s** without MTP.
* **256K was not tested** because it caused more than **1.1 GB of CPU/RAM offloading**, which would significantly hurt throughput.

**Recommendation:** For this GPU, **MTP 1 with a 172K context** is the best overall configuration. It provides a large usable context while keeping throughput high enough for practical use. MTP 2 offers no meaningful advantage over MTP 1 at these larger context sizes, so **MTP 1 is preferable**.

Overall, the Q2_K_XL model can handle **very large contexts**, but beyond roughly **150K tokens**, the main limitation becomes KV-cache pressure and the resulting throughput reduction rather than the model's nominal context limit.

## D2 — IQ2_XXS (2-bit)

This is the smallest possible one, for sure less capable than the other quantization, but we can put a large context in it.

### MTP 2

#### 192 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-IQ2_XXS.gguf
--ctx-size 196608 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 42-57 t/s

VRAM used:  14.7 GB (600 MB Offloaded)

Prompts additional 120 K context: prefill about 440 t/s, inference 21 t/s

VRAM used @ 151.3 K : 14.7 GB ( 600 MB Offloaded)

#### 220 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-IQ2_XXS.gguf
--ctx-size 220000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2
```

Prompts with the 3 classic questions, one after another: 42-46 t/s

VRAM used:  15.3 GB (600 MB Offloaded)

Prompts additional 80 K context: prefill about 507 t/s, inference 25 t/s

VRAM used @ 104 K : 15.3 GB ( 600 MB Offloaded)

Prompts additional 80 K context: prefill about 317 t/s, inference 18 t/s

VRAM used @ 202.7 K : 15.3 GB ( 600 MB Offloaded)

### MTP 1

#### 220 K Context

```bash
llama-server
-m ..\Qwen3.8-27B-UD-IQ2_XXS.gguf
--ctx-size 220000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1
```

Prompts with the 3 classic questions, one after another: 40-43 t/s

VRAM used:  15.2 GB (600 MB Offloaded)

Prompts additional 80 K context: prefill about 517 t/s, inference 25 t/s

VRAM used @ 104 K : 15.2 GB ( 600 MB Offloaded)

Prompts additional 80 K context: prefill about 324 t/s, inference 18 t/s

VRAM used @ 199 K : 15.2 GB ( 600 MB Offloaded)

### MTP tuning

Test performed running back to back 4 to 5 times the same prompt while varying `--spec-draft-n-max` from 1 to 4 and trying some increments on `--spec-draft-p-min` from 0.0 to 0.9 using various increments. Note: `--reasoning off` and a fixed seed (`--seed 12345`) are used here, so this micro-benchmark is not directly comparable to the context tests above.

```bash
llama-server
-m ..\Qwen3.8-27B-UD-IQ2_XXS.gguf --ctx-size 65536
-fa on --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers all --parallel 1
--temp 0.2 --top-p 1.0 --top-k 0 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--seed 12345 --spec-type draft-mtp --spec-draft-n-max XXXX --spec-draft-p-min YYYY --reasoning off
```

Prompt:

"You are reviewing a Python service that processes a large stream of JSON events.

The service receives events with this structure:

{
"id": "evt_123",
"timestamp": 1712345678,
"user_id": 42,
"type": "purchase",
"payload": {
"amount": 19.99,
"currency": "EUR"
}
}

The current implementation is:

import json
from collections import defaultdict

totals = defaultdict(float)

def process(lines):
for line in lines:
event = json.loads(line)

if event["type"] == "purchase":
user_id = event["user_id"]
amount = event["payload"]["amount"]
totals[user_id] += amount
return dict(totals)
The production system may process several million events per hour. Events can arrive out of order, duplicated, or malformed. Some events can have missing fields. The service runs continuously and should not keep the entire input stream in memory.

Analyze this implementation and propose a production-quality redesign.

Explain the problems with the current implementation, including correctness, floating-point accuracy, malformed input handling, duplicate events, memory usage, concurrency, and observability.

Then provide a complete Python implementation of your proposed solution. The implementation should process the input as a stream, validate events, handle malformed records without stopping the entire stream, deduplicate events using event IDs, maintain monetary values accurately, and expose useful metrics such as processed events, rejected events, duplicate events, and total processing time.

After the implementation, explain the time and space complexity.

Finally, discuss how the design should change if the service is scaled horizontally across multiple machines and events for the same user can be processed by different workers.

Be thorough and include concrete implementation details rather than only high-level recommendations.

"

#### Short summary

| Config                 | Avg t/s          | Observation                                         |
| ---------------------- | ---------------- | --------------------------------------------------- |
| `n-max=1, p-min=0`     | **41.89**        | Becomes preferable above the context-size threshold |
| `n-max=2, p-min=0`     | **46.26**        | Best practical baseline                             |
| `n-max=3, p-min=0`     | **46.91**        | Only \~1.4% faster                                  |
| `n-max=4, p-min=0`     | **45.62**        | Slower; not worthwhile                              |
| `n-max=2, p-min≥0.10` | **\~45.8–46.1** | No benefit from `p-min`                              |

**Conclusion:** `n-max=2, p-min=0` is the best short-context setting. We already know that **above a context-size threshold, `n-max=1` becomes more efficient/faster**, making it preferable for long-context/agentic workloads.

### Observations

I did not run without MTP those tests, this model keeps the throughput high even when we reach the majority of the context. It can run in 220K without any issue, there is no real gain to run above MTP with one way. There is a bit margin to put few more tokens but that is already on the limit.

The "Max context reached" column reports how far the context actually filled before offloading took over, for a test run at the given ctx-size.

| Configuration | Context (ctx-size) | Max context reached | Throughput |   Prefill | Inference |    VRAM |
| ------------- | -----------------: | ------------------: | ---------: | --------: | --------: | ------: |
| MTP 2-way     |            192K |                151K | 42–57 t/s | \~440 t/s |  \~21 t/s | 14.7 GB |
| MTP 2-way     |            220K |                104K | 42–46 t/s | \~507 t/s |  \~25 t/s | 15.3 GB |
| MTP 2-way     |            220K |                203K | 42–46 t/s | \~317 t/s |  \~18 t/s | 15.3 GB |
| MTP 1-way     |            220K |                104K | 40–43 t/s | \~517 t/s |  \~25 t/s | 15.2 GB |
| MTP 1-way     |            220K |                199K | 40–43 t/s | \~324 t/s |  \~18 t/s | 15.2 GB |

IQ2_XXS is the smallest and least capable quantization tested, but its main advantage is its ability to handle a **very large context (\~220K tokens)** while maintaining surprisingly good throughput. MTP provides a modest speed benefit, especially at shorter contexts, but **2-way MTP offers little practical advantage over 1-way MTP** at very large context sizes. VRAM usage remains around **15 GB**, with \~600 MB offloaded.

Overall, IQ2_XXS is a compelling option when **maximum context length and low VRAM usage are more important than model quality**. It can reach roughly 200K tokens of context without a dramatic collapse in inference speed.

## D3 — UD-IQ3_S (3-bit)

Test Date: 21 Aug 2026

Unsloth reports that this *Dynamic Quant V3.0* variant better preserves model capabilities.

**llama-b10472** used for those tests.

### No MTP

#### 32 K Context

```bash
llama-server
-m "..\Qwen3.8-27B-UD-IQ3_S.gguf"
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--reasoning-effort medium
```

Standard test for burst mode.

Prompt 1: Output 869 tokens, 28s, 30.09 t/s

Prompt 2: Output 1,571 tokens, 52s, 29.80 t/s

Prompt 3: Output 1,528 tokens, 52s, 29.32 t/s

VRAM used: 12.3 GB

#### 180 K Context

```bash
llama-server
-m "..\Qwen3.8-27B-UD-IQ3_S.gguf"
--ctx-size 180000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--reasoning-effort medium
```

This is the maximum context size achievable without significant offloading.

Prompt 1: Output 849 tokens, 29s, 28.34 t/s

Prompt 2: Output 1,387 tokens, 49s, 27.90 t/s

Prompt 3: Output 1,626 tokens, 58s, 27.78 t/s

VRAM used: 15.6 GB (400 MB Shared)

Additional 49K tokens

Prefill: 49180 tokens, 1min 11s, 684.88 t/s

@53.7 K : Output: 585 tokens, 28s, 20.66 t/s

Additional 24.5 K

Prefill: 24606 tokens, 47s, 519.79 t/s

@78.5 K : Output: 157 tokens, 8.5s, 18.52 t/s

### MTP 1

Skipped 32K test; focused on max context.

#### 120 K Context

```bash
llama-server
-m "..\Qwen3.8-27B-UD-IQ3_S.gguf"
--ctx-size 120000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 1 --reasoning-effort medium
```

Prompt 1: Output 852 tokens, 21s, 40.23 t/s

Prompt 2: Output 1,849 tokens, 45s, 40.62 t/s

Prompt 3: Output 1,490 tokens, 37s, 39.38 t/s

VRAM used: 15.4 GB (400 MB shared)

Additional 49 K

Prefill: 49182 tokens, 1min 11s, 684.62 t/s

@54.3 K : Output: 917 tokens, 29s, 30.63 t/s

Additional 24.5 K

Prefill: 24606 tokens, 47s, 521.74 t/s

@79.19 K: Output: 208 tokens, 7.6s, 27.25 t/s

Additional 24.5 K

Prefill: 24606 tokens, 54s, 451.47 t/s

@103.87 K: Output: 76 tokens, 3.0s, 25.67 t/s

### MTP 2

#### 32 K Context

```bash
llama-server
-m "..\Qwen3.8-27B-UD-IQ3_S.gguf"
--ctx-size 32768 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2 --reasoning-effort medium
```

Standard test for burst mode.

Prompt 1: Output 924 tokens, 20s, 44.58 t/s

Prompt 2: Output 2,262 tokens, 51s, 43.58 t/s

Prompt 3: Output 1,530 tokens, 36s, 42.22 t/s

VRAM used: 13.4 GB

#### 120 K Context

```bash
llama-server
-m "..\Qwen3.8-27B-UD-IQ3_S.gguf"
--ctx-size 120000 -fa on
--cache-type-k q4_0 --cache-type-v q4_0
--n-gpu-layers all --parallel 1
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
--spec-type draft-mtp --spec-draft-n-max 2 --reasoning-effort medium
```

Nearly max context you can put on the GPU. Test burst mode; there are a few more tokens you can put before offloading to the CPU/MEM.

Prompt 1: Output 1,153 tokens, 27s, 41.88 t/s

Prompt 2: Output 1,701 tokens, 39s, 43.15 t/s

Prompt 3: Output 1,105 tokens, 26s, 40.93 t/s

VRAM used: 15.6 GB (400 MB shared)

Additional 49 K

Prefill: 49180 tokens, 1min 10s, 696.97 t/s

@54.2 K : Output: 1,001 tokens, 33s, 30.04 t/s

Additional 24.5 K

Prefill: 24606 tokens, 46s, 532.54 t/s

@79.05 K: Output: 220 tokens, 9.0s, 24.46 t/s

Relaunch server and resume the session (prefill all tokens): 103080 tokens prefill.

@103.8 K: Output: 184 tokens, 7.5s, 24.42 t/s

### Observations

Nothing really special, the throughput is pretty similar to previous GGUFs. I don't evaluate if the model is more capable.

| Config    | Ctx  | Throughput | Prefill  | Inference       | VRAM    |
| --------- | ---- | ---------- | -------- | --------------- | ------- |
| No MTP    | 32K  | 29–30 t/s | —       | —              | 12.3 GB |
| No MTP    | 180K | 27–28 t/s | ~685 t/s | ~18.5 t/s @78K  | 15.6 GB |
| MTP 1-way | 120K | 39–40 t/s | ~685 t/s | ~25.7 t/s @104K | 15.4 GB |
| MTP 2-way | 32K  | 42–44 t/s | —       | —              | 13.4 GB |
| MTP 2-way | 120K | 40–43 t/s | ~697 t/s | ~24.4 t/s @104K | 15.6 GB |

MTP 1 and MTP 2 perform very similar with 120 K. Without MTP you can have 180 K context, but the throughput is slightly smaller. It makes sense to use the MTP from a throughput point of view, even when the context is nearly full there is still a nice throughput advantage.

## ISTA DASLab (3-bit)

This is another variant with 3-bits, with a dynamic quantification. There is no MTP head on it by default. [Link](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)

We use the following quant with and without MTP:

- Qwen3.8-27B-GSQ-RCO-IQ3_XXS
- Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp

(Note: config files are shown here instead of command lines.)

### No MTP

#### 240 K Context

```bash
[Qwen3.8-27B-GSQ-RCO-IQ3_XXS]
model = Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf
ctx-size = 240000
n-gpu-layers = all
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
no-mmproj = true
```

Prompt 1 : 1,155 tokens 38s 29.96 t/s

Prompt 2 : 1,506 tokens 50s 29.81 t/s

Prompt 3: 1,573 tokens 53s  29.53 t/s

Prefill: 97823 tokens 2min 55s 557.50 tokens/s

Prompt: 655 tokens 35s 18.21 t/s

VRAM Used: 15.6 GB (Shared 400 MB)

These are expected figures for this model. The key point is that this dynamic quant keeps a lot of capabilities, a bit better than Unsloth Dynamic Quant 3. Of course, this document is about what we can run and how fast.

Waiting for the MTP version that could be very interesting too.

The advantage is to have more context.

### MTP 2

#### 170 K Context

```bash
[Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp]
model = Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
ctx-size = 170000
n-gpu-layers = all
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
no-mmproj = true
spec-type = draft-mtp
spec-draft-n-max = 2
```

Prompt 1: 798 tokens 18s 42.93 t/s

Prompt 2: 1,023 tokens 24s 41.85 t/s

Prompt 3: 1,250 tokens 30s 40.65 t/s

3.1K Additional Context: 99076 tokens 2min 57s  558.41 tokens/s

Prompt: 638 tokens 25s 25.46 t/s

Context Used: 101.2 K

### Observations

This new quant offers a very large context without MTP with 240K and a slight increase on the context too to 170K with MTP. The Unsloth 3-bit quant with MTP was limited to 120K; with this one we have 170K, which is a large increase of more than 40%. Without MTP we move from 180K to 240K, a large increase of 33%. MTP 2 at about 100K context is still nearly 40% faster in inference.

## DFlash2 Speculative Decoding

A focused study of this model (ISTA DASLab GSQ RCO `IQ3_XXS`) with a DFlash2 draft model (`spec-type draft-dflash`, `n-max 4`, q4_0 KV cache for both main and draft) is documented in [rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md). Key results, on real coding tasks:

* **32K–64K context: ~60–65 t/s** — the fastest configuration found for this model on the 16 GB RTX 5060 Ti, at 13.3–15.1 GB VRAM.
* **96K context: ~40–45 t/s** — memory pressure starts to show (dedicated VRAM ~14.8–14.9 GB, shared GPU memory ~1.0 GB).
* **128K context: ~40 t/s** — dedicated 15.0–15.1 GB + ~1.1 GB shared.
* **162K context: ~23 t/s** — the combined dedicated + shared footprint (~16.7 GB) exceeds the 16 GB card, and offloading becomes the dominant bottleneck.

The limiting factor at large context is therefore memory capacity and offloading overhead, not the speculative decoding efficiency itself.

## Recommendations

The right configuration depends on how much context you need:

1. **Speed / medium-to-large context → DFlash2.** The fastest configuration found for this model: ~60–65 t/s at 32K–64K, ~40–45 t/s at 96K, ~40 t/s at 128K (ISTA DASLab GSQ-RCO `IQ3_XXS` + DFlash2 draft, 13.3–15.1 GB). Beyond 128K the combined dedicated + shared footprint approaches the card limit and drops to ~23 t/s at 162K (see the [DFlash2 report](rtx5060ti-qwen3.8-27b-dflash2-speculative-decoding.md)).
2. **Large context → MTP.** MTP 2 keeps the model fast well past the DFlash2 sweet spot: ISTA DASLab GSQ-RCO at 170K context, 40–43 t/s, 15.6 GB (UD-IQ3_S D3 at 120K is similar, 40–43 t/s).
3. **Largest possible context → no speculative decoding.** A draft/MTP head consumes VRAM and layers; dropping speculation gives the maximum context: ISTA DASLab GSQ-RCO at 240K, or D2 `IQ2_XXS` at 220K (up to ~200K usable, ~15 GB), at the cost of quantization quality.

Per-quantization recipes measured in this file:

- **Q3_K_M (D2):** 64K context + MTP-2, \~33–39 t/s, 15.6 GB. Best balance for 3-bit at 64K.
- **Q3_K_S (D2):** 112K context + MTP 1, \~35 → 23 t/s, 15.5 GB. Best balance for 3-bit at larger context.
- **IQ4_NL (D2):** 59 layers + MTP 2 at 32K, 20–22 t/s, 15.3 GB. At the practical limit of a 16 GB GPU.
- **Q2_K_XL (D2):** 172K context + MTP 1, \~39–42 → 20 t/s, 15.3 GB. Sweet spot for very large context.
- **IQ2_XXS (D2):** 220K context + MTP 2 (or MTP 1), up to \~200K usable, \~15 GB. Maximum context, at the cost of quality.
- **UD-IQ3_S (D3):** 120K context + MTP 2, 40–43 t/s, 15.6 GB.

The new **IQ3_S D3** variant is particularly interesting for daily coding and agentic use. With **MTP-2 and a 120K context**, I can reach around **50 t/s in real-world coding sessions**, while keeping the throughput remarkably consistent.

I previously used **Q3_K_M + MTP-2** extensively at 32K context, typically getting around **40–46 t/s**. Being able to move to a much larger context without sacrificing interactive usability is, for me, one of the biggest improvements.

It also means **less context engineering is required**: fewer situations where I need to carefully manage the context, fork a session, or `merge` it back together. Of course, for long-running agentic sessions, forking and merging is still a good habit, but having 120K available gives much more breathing room.

**01/09/2026:** This non-uniform quantized model in 3-bits with MTP is my new recommendation for the Qwen 3.8 27B. I used the Unsloth **IQ3_S D3** variant for my daily coding and agentic use since it was out, great quant too. But I think this ISTA DASLab is a bit better, at least for my usage.

A larger context is possible: from 240K without MTP to 170K with MTP 2.
