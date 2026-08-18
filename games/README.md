One shot games made with super simple prompt with Qwen 3.8 27B 

```bash
llama-server -m "unsloth--Qwen3.8-27B-GGUF\Qwen3.8-27B-Q3_K_M.gguf"  --ctx-size 32768 -fa on  --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers all --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --spec-type draft-mtp --spec-draft-n-max 2 --reasoning-effort low
```

Prompt 0 :
make a single page html implementing the pong game, but with a 90's neon style and some cool effects, make a short description, and generate everything in a single page

Prompt 1: 
make a single page html implementing the tetris game, using 90's neon colors, make it simple and slick, generate everything in a single page html

Prompt 2:
make a single page html implementing the breakout game, using 90's neon colors, make it simple and slick, generate everything in a single page html

Prompt 3:
make a single page html implementing the snake game, using 90's neon colors, make it simple and slick, generate everything in a single page html

