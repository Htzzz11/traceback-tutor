#!/usr/bin/env bash
# Training smoke run: measures speed and memory only. The data is the 5 fixtures
# repeated to 50 rows, so the adapter it produces is meaningless.
#
#   bash train/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=runs/smoke_data
ADAPTER=adapters/smoke
LOG=runs/smoke_train.log
ITERS=${ITERS:-50}

# 1. Smoke data: 5 fixtures x 10 = 50 train rows; the 5 fixtures again as valid.
uv run python - "$DATA" <<'EOF'
import sys
from dataclasses import replace
from pathlib import Path
from traceback_tutor.schema import read_jsonl
from traceback_tutor.to_chat import export

fixtures = list(read_jsonl("data/fixtures/examples.jsonl"))
train = [replace(ex, split="train") for _ in range(10) for ex in fixtures]
val = [replace(ex, split="val") for ex in fixtures]
print(export(train + val, Path(sys.argv[1])))
EOF

# 2. Train. Same as `mlx_lm.lora ...`, but prints MLX's peak memory at exit
#    (mlx-lm 0.32 tracks it but no longer prints it). Its tok/s counts only the
#    assistant tokens that get loss, not the masked prompt.
start=$(date +%s)
uv run python - -c train/lora_config.yaml \
    --data "$DATA" --adapter-path "$ADAPTER" --iters "$ITERS" \
    --steps-per-eval "$ITERS" --save-every "$ITERS" <<'EOF' 2>&1 | tee "$LOG"
import atexit, sys
import mlx.core as mx
from mlx_lm.lora import main

atexit.register(lambda: print(f"peak memory: {mx.get_peak_memory() / 1e9:.2f} GB"))
sys.argv[0] = "mlx_lm.lora"
main()
EOF
echo "wall time: $(($(date +%s) - start)) s (includes model load)" | tee -a "$LOG"

# 3. Load base + adapter and generate once, with the same messages as training.
uv run python - "$ADAPTER" <<'EOF'
import sys
from mlx_lm import generate, load
from traceback_tutor.prompt import build_messages, parse_response
from traceback_tutor.schema import read_jsonl

model, tokenizer = load("Qwen/Qwen2.5-Coder-1.5B-Instruct", adapter_path=sys.argv[1])
ex = next(read_jsonl("data/fixtures/examples.jsonl"))
prompt = tokenizer.apply_chat_template(
    build_messages(ex.code, ex.traceback), add_generation_prompt=True, tokenize=False
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
print(parse_response(text))
EOF
