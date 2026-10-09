# Python Error Explainer

A small fine-tuned code LLM that reads a beginner's Python program and its traceback, then explains
the error in plain English and gives a hint, without handing over the fix.

**Status:** in progress (v0 thin slice).

## Why
Python error messages assume you already know what a type, an index or a scope is, and general
chatbots tend to just return corrected code. This project asks how close a 1.5B model fine-tuned on
one narrow task can get to a frontier model, while staying cheap, private and restrained.

## How it works
1. **Data:** correct programs from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
   (CC-BY-4.0) are broken on purpose with `ast`-based bug injections, then run in a sandbox to
   capture real tracebacks.
2. **Teacher:** a frontier model writes a plain-English explanation and one hint for each error.
3. **Training:** LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct with mlx-lm on a 16GB M4 MacBook.
4. **Evaluation:** held-out programs (split by seed program), base vs fine-tuned model.

## Example
_Coming with v0._

## Results
_Coming with v0._

## Quickstart
```bash
uv sync
uv run pytest -q
```
