# Few experiments

```bash
llama-server -m ".cache\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\fdd03b8bbd279c1694563650e79d85a2373d9934\Qwen3.8-27B-Q3_K_M.gguf"  --ctx-size 32768 -fa on  --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers all --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --spec-type draft-mtp --spec-draft-n-max 2 --reasoning-effort low
```

## Mini Ray Tracer
I am back in 1992 with one of the first scene rendered on a 482 DX2 with the Vivid Ray Tracer

Prompt: Provide inside a single html page a small raytracer that displays a scene with a sphere with reflexions with a checker on the ground and clouds in the back, everything should be in the single page. The raytracer is not using any api, you implement a software raytracer.

## Pagoda Scene
Prompt: make inside a single html page, using only the external Threejs library, a scene with an orbital camera centered around the pagoda of even, it is the day time, the ground is green, there are trees, it is wonderful and peaceful,  be creative and make your best effort to build something truly memorable.

