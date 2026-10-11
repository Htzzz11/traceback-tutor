# Python Error Explainer

A small fine-tuned code LLM that reads a beginner's Python program and its traceback, then explains
the error in plain English and gives a hint, without handing over the fix.

**Status:** v0 working end to end (5 bug types, 500 examples). v1 in progress.

## Why
Python error messages assume you already know what a type, an index or a scope is, and general
chatbots tend to just return corrected code. This project asks how close a 1.5B model fine-tuned on
one narrow task can get to a frontier model, while staying cheap, private and restrained.

## How it works
1. **Data:** correct programs from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
   (CC-BY-4.0) are broken on purpose with `ast`-based bug injections, then run in a sandbox to
   capture real tracebacks. Splits are made by seed program, with near-duplicate programs grouped,
   so test programs are never seen in training.
2. **Teacher:** a frontier model (OpenAI gpt-6.1-sol) writes a plain-English explanation and one
   hint for each error, following a fixed [style guide](docs/style-guide.md). Automatic filters
   reject wrong lines, leaked fixes and over-long answers. Labelling 500 examples cost $0.88.
3. **Training:** LoRA (rank 8, 0.34% of parameters trainable) on Qwen2.5-Coder-1.5B-Instruct with
   mlx-lm, loss on the answer only. 200 steps took 29 minutes on a 16GB M4 MacBook, peak memory 6.4 GB.
4. **Evaluation:** base vs fine-tuned on the held-out test split, same prompt, greedy decoding,
   automatic metrics with paired bootstrap CIs and McNemar tests.

## Example
A held-out program where the bug (line 2) is not where it crashes (line 4):

```python
def cube_Sum(n):
    sum = "0"
    for i in range(1,n + 1):
        sum += (2*i)*(2*i)*(2*i)
    return sum
```
```
TypeError: can only concatenate str (not "int") to str
```

**Base model** points at the crash line, and its hint steers toward the wrong fix:
> Line 4: The error occurs because the variable `sum` is initially set as a string "0", and then it
> is being concatenated with the result of the multiplication operation, which is an integer. [...]
>
> Hint: How can you convert the integer result of the multiplication to a string before concatenating it to `sum`?

**Fine-tuned model** points at the real cause and leaves the fix to the student:
> Line 2: The quotes around `"0"` make `sum` text rather than a number. On line 4, the program tries
> to add a number to that text, which Python can't do.
>
> Hint: How do the quotes around `"0"` on line 2 make it different from the numbers on line 4?

## Results (v0)
60 held-out test programs, 5 bug types. Differences are paired (same programs), with 95% bootstrap CIs.

| Metric | Base 1.5B | Fine-tuned 1.5B | Difference [95% CI] |
| --- | --- | --- | --- |
| Follows the answer format | 68% (41/60) | **100%** (60/60) | +32 pts [+20, +43] |
| Cites the bug line | 72% (43/60) | **98%** (59/60) | +27 pts [+15, +38] |
| ...when the bug line differs from the crash line | 0/10 | **9/10** | |
| Cites the bug or crash line | 85% (51/60) | **100%** (60/60) | +15 pts [+7, +25] |
| Gives away the fix (lower is better) | 23% (14/60) | **0%** (0/60) | −23 pts [−35, −13] |
| Names the error type | 90% (54/60) | 98% (59/60) | +8 pts [0, +17], not significant |
| Words per response | 24 | 49 | |
| Seconds per response (full precision, M4) | 1.7 | 3.1 | longer answers, same 22 tokens/s |

**Limitations.** The test set is small (60). The metrics only check structure: line, format,
leaked fixes, error type. They don't measure whether an explanation is clear or even correct. In
one test response the model miscounts the characters in a string, copying a template from its
training examples. Teacher explanations are also formulaic, and the model inherits that phrasing.
v1 adds an LLM judge validated against blind hand labels, all 13 bug types, a prompting-only
baseline and a frontier-model ceiling.

## Reproduce
```bash
uv sync
uv run python -m traceback_tutor.build_dataset --out data/v0
uv run --env-file .env python -m traceback_tutor.teacher \
    --in data/v0/train.jsonl data/v0/val.jsonl data/v0/test.jsonl --out data/v0/labelled.jsonl
uv run python -m traceback_tutor.to_chat --in data/v0/labelled.jsonl --out data/v0/chat
uv run mlx_lm.lora -c train/lora_config.yaml
uv run python -m traceback_tutor.eval.run_eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --data data/v0/labelled.jsonl --split test --name v0_test_base
uv run python -m traceback_tutor.eval.run_eval --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --adapter adapters/v0 --data data/v0/labelled.jsonl --split test --name v0_test_lora
uv run python -m traceback_tutor.eval.compare runs/v0_test_base runs/v0_test_lora
```
The teacher step needs `OPENAI_API_KEY` in `.env`, and its output varies between runs.

Try it on your own file:
```bash
uv run explain buggy.py
```
