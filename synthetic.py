"""Generate synthetic training text with a trained checkpoint.

Once the model is good enough to write coherent text, it can write
its own training data. That is the synthetic share of the data mix:
seed prompts go in, the model writes continuations, and the output
goes into the training stream alongside the web, books, code and
papers. The synthetic data is generated from an earlier checkpoint,
never from the run it trains, so the model cannot just rehearse
its own output back to itself in a loop.

This is the same idea frontier labs use for synthetic data, at a
small scale. The quality of the synthetic text is only as good as
the checkpoint that wrote it, so do not use synthetic data from a
smoke checkpoint for a real run. Generate it from a real
checkpoint, skim a few samples, and only then mix it in.

Usage:
  python synthetic.py --model out-nyx12/ckpt-5000 --out synthetic.jsonl --n-per-prompt 10
  python synthetic.py --model out-nyx12/ckpt-5000 --out synthetic.jsonl --prompts-file prompts.txt

The output is one JSON object per line, {"prompt": ..., "text": ...}.
data.py reads it back with synthetic_text_iter.
"""
import argparse
import json
import os
import random

# seed prompts that cover the domains the model should write about.
# A mix of explanations, stories, code and science, so the synthetic
# text is not all one kind of writing.
DEFAULT_SEED_PROMPTS = [
    "Explain how a jet engine works, in simple words.",
    "Once upon a time, in a small village by the sea,",
    "Write a Python function that sorts a list of numbers and explain it.",
    "The history of the printing press changed the world because",
    "Photosynthesis is the process by which plants",
    "A good way to learn a new language is to",
    "The water cycle moves water from the ocean to the sky and back. First,",
    "Write a short story about a robot who learns to paint.",
    "Newton's laws of motion say that",
    "To make bread at home, you need flour, water, yeast and salt. First,",
    "The Amazon rainforest is important because",
    "Explain the difference between a star and a planet.",
    "A database index makes queries faster by",
    "The human immune system defends the body by",
    "Write a poem about the first rain of the season.",
    "How does a vaccine work? A vaccine",
    "The invention of the telephone changed",
    "To solve a quadratic equation, first",
    "The deep sea is home to creatures that",
    "A black hole is a region of space where",
]


def synthetic_text_iter(path):
    """Yield synthetic doc texts from a JSONL file, one per line.

    Raises FileNotFoundError if the file is not there, so the caller
    can decide whether to warn and carry on or stop. Skips blank
    lines and malformed rows instead of crashing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"synthetic file not found: {path}, "
                                "generate it with synthetic.py first")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            if text:
                yield text


def count_synthetic(path):
    """How many docs are in a synthetic file, for logging."""
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def generate_synthetic(model_path, prompts, n_per_prompt=10, max_new=200,
                       out_path="synthetic.jsonl", device=None, temp=0.8,
                       top_p=0.95, seed=0, dtype="bf16"):
    """Generate synthetic text from a checkpoint and write JSONL.

    model_path is a trained checkpoint dir with weights and a
    tokenizer. Each seed prompt is sampled n_per_prompt times, and
    every sample is written as one doc. Returns the number of docs
    written. The model is loaded once, then all prompts run through
    it, so generating a thousand docs is one load, not a thousand.
    """
    import torch

    from generate import load_model
    from inference import generate

    random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    model, tok = load_model(model_path, device, dtype=torch_dtype)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            for i in range(n_per_prompt):
                text = generate(
                    model, tok, prompt, max_new=max_new,
                    temperature=temp, top_k=50, top_p=top_p,
                    device=device)
                full = (prompt + " " + text).strip()
                f.write(json.dumps({"prompt": prompt, "text": full}) + "\n")
                written += 1
                if written % 10 == 0:
                    print(f"wrote {written} docs", flush=True)
    print(f"done: {written} synthetic docs in {out_path}")
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="trained checkpoint dir to generate from")
    p.add_argument("--out", default="synthetic.jsonl")
    p.add_argument("--prompts-file", default=None,
                   help="file with one seed prompt per line")
    p.add_argument("--n-per-prompt", type=int, default=10)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    prompts = list(DEFAULT_SEED_PROMPTS)
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
    if not prompts:
        raise ValueError("no seed prompts, check --prompts-file")
    print(f"{len(prompts)} seed prompts, {args.n_per_prompt} samples each")
    generate_synthetic(args.model, prompts, n_per_prompt=args.n_per_prompt,
                       max_new=args.max_new, out_path=args.out,
                       temp=args.temp, top_p=args.top_p, seed=args.seed,
                       dtype=args.dtype)


if __name__ == "__main__":
    main()
