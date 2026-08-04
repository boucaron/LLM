text = """
The system processes information sequentially and maintains a representation
of the previous context. Each new observation can depend on information that
appeared much earlier in the conversation. This is useful for testing long
context behavior because the model must continue processing tokens while
retaining information from earlier parts of the sequence.

A practical local inference system needs to balance context length, memory
usage, cache precision, and generation speed. Increasing the context consumes
additional memory, while quantizing the key-value cache can reduce memory
requirements and make larger contexts possible. The exact tradeoff depends on
the model architecture, hardware bandwidth, and implementation details.

Local inference is particularly interesting on GPUs with limited VRAM.
Mixture-of-Experts models can keep a relatively small number of parameters
active for each token while still providing a much larger total parameter
count. However, when the complete model does not fit in GPU memory, some
experts may need to remain in system memory, introducing additional PCIe and
CPU memory traffic.

The purpose of this test is simply to provide a long sequence of tokens.
There is no hidden question and no important information that the model needs
to recover. The text is deliberately repetitive so that the context can be
filled quickly and reproducibly. The benchmark should focus on inference
performance rather than the quality of the generated content.

Memory bandwidth is often an important factor in autoregressive generation.
Every generated token requires the model to access a substantial amount of
model state. Sparse models reduce the amount of computation required per token,
but moving data between GPU memory and system memory can become a significant
limitation when experts are offloaded.

Flash Attention can reduce the memory overhead associated with attention and
can make large context windows more practical. Quantized KV caches provide
another way to reduce memory consumption. A four-bit cache can provide
substantial savings compared with a full-precision representation, although
the exact impact depends on the implementation and model.

The benchmark should remain stable across repeated runs. Using the same
context, the same model, the same sampling parameters, and the same hardware
configuration makes it easier to identify changes in generation throughput.
A fresh server process can also help avoid confusing persistent state with
the performance of the model itself.

This paragraph is repeated many times only to increase the number of tokens.
The semantic content is intentionally unimportant. The model should simply
process the sequence as part of its context and continue generating after the
context has been filled.
"""

# ~130 KB of text, roughly in the right range for a 32K-token English context.
target_chars = 135_000

with open("context-32k.txt", "w", encoding="utf-8") as f:
    while f.tell() < target_chars:
        f.write(text)

print("Created context-32k.txt")
print("Characters:", __import__("os").path.getsize("context-32k.txt"))


target_chars *= 2

with open("context-64k.txt", "w", encoding="utf-8") as f:
    while f.tell() < target_chars:
        f.write(text)

print("Created context-64k.txt")
print("Characters:", __import__("os").path.getsize("context-64k.txt"))


target_chars *= 2

with open("context-128k.txt", "w", encoding="utf-8") as f:
    while f.tell() < target_chars:
        f.write(text)

print("Created context-128k.txt")
print("Characters:", __import__("os").path.getsize("context-128k.txt"))


target_chars *= 2

with open("context-256k.txt", "w", encoding="utf-8") as f:
    while f.tell() < target_chars:
        f.write(text)

print("Created context-256k.txt")
print("Characters:", __import__("os").path.getsize("context-256k.txt"))

