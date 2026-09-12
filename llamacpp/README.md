# Llama.cpp configuration

Personal backup of my `llama-server` configuration profiles (format version 1) for the
Qwen 3.8 27B variants, as actually used on my machine.

**Heads up:** the `model` and `model-draft` paths are hard-coded absolute paths into my
local Hugging Face cache (including the snapshot hashes), so these files do **not**
work as-is on any other machine or cache. To use a profile, replace the `model` /
`model-draft` values with the paths to the corresponding GGUF files in your own cache.

The profile *settings* (context size, GPU layers, flash attention, q4_0 KV cache, MTP /
DFlash2 speculative decoding) are the ones tuned for the **RTX 5060 Ti 16 GB** benchmark
run described in the parent [README](../README.md).
