"""Evaluation for Nyx 1.2. Perplexity, benchmarks, and generation.

Three kinds of eval, all in one place so training can call them
without importing half the internet:

1. Perplexity on held-out web text. The number to watch while
   training. If it is going down, the model is learning.
2. Multiple-choice benchmarks in the style of HellaSwag, ARC and
   Winogrande. The model scores each option by likelihood and picks
   the best. No generation needed, so it is fast and stable.
3. Generation quality. Sample some completions, score them for
   repetition and length, and print a few so a human can read them.

Everything that needs the datasets library loads it lazily. The
perplexity path works with any list of texts, so you can eval on
local files with no downloads at all.
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F


# perplexity
# ----------

@torch.no_grad()
def eval_perplexity(model, tokenizer, texts, seq_len=2048, batch_size=1,
                    device="cpu", max_batches=None):
    """Average loss and perplexity over a list of texts.

    Texts are packed into seq_len blocks, the model scores them, and
    the loss is averaged with token weighting. Returns a dict with
    loss, perplexity, tokens and batches. Lower is better.
    """
    from data import pack_documents

    model.eval()
    blocks = pack_documents(tokenizer, texts, seq_len)
    if max_batches is not None:
        blocks = blocks[:max_batches * batch_size]
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i:i + batch_size]
        x = torch.tensor(chunk, dtype=torch.long, device=device)
        out = model(x, labels=x)
        n_tokens = x.numel()
        total_loss += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
        n_batches += 1
    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"),
                "tokens": 0, "batches": 0}
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss),
            "tokens": total_tokens, "batches": n_batches}


@torch.no_grad()
def eval_perplexity_file(model, tokenizer, path, seq_len=2048,
                         batch_size=1, device="cpu", max_batches=None):
    """Perplexity on a plain text file, one document per line or
    the whole file as one document if it has no newlines."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    lines = [l for l in content.splitlines() if l.strip()]
    texts = lines if len(lines) > 1 else [content]
    return eval_perplexity(model, tokenizer, texts, seq_len=seq_len,
                           batch_size=batch_size, device=device,
                           max_batches=max_batches)


# multiple choice benchmarks
# --------------------------

@torch.no_grad()
def score_choice(model, tokenizer, context, choice, device="cpu"):
    """Log-likelihood of a choice given a context, normalized by length.

    The model reads context + choice and we sum the log-probs of the
    choice tokens only, divided by the number of choice tokens. That
    normalization matters, otherwise short answers always win.
    Returns the average log-prob per token.
    """
    ctx_ids = tokenizer.encode(context)
    choice_ids = tokenizer.encode(choice)
    ids = ctx_ids + choice_ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out.logits[0]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    # token at position i predicts token i+1, so the choice tokens
    # start at len(ctx) - 1 in the shifted predictions
    total = 0.0
    start = len(ctx_ids) - 1
    for j, tok in enumerate(choice_ids):
        total += float(logprobs[start + j, tok].item())
    return total / max(1, len(choice_ids))


def eval_multiple_choice(model, tokenizer, items, device="cpu"):
    """Score a list of multiple-choice items.

    Each item is a dict with context, choices (a list of strings),
    and answer (the index of the right choice). Returns accuracy
    and per-item results. The harness is generic on purpose: feed
    it HellaSwag, ARC, Winogrande or your own set, same format.
    """
    correct = 0
    results = []
    for item in items:
        scores = [score_choice(model, tokenizer, item["context"], c, device=device)
                  for c in item["choices"]]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        ok = pred == item["answer"]
        correct += int(ok)
        results.append({"context": item["context"][:80],
                        "scores": [round(s, 3) for s in scores],
                        "pred": pred, "answer": item["answer"], "ok": ok})
    acc = correct / len(items) if items else 0.0
    return {"accuracy": acc, "correct": correct, "total": len(items),
            "results": results}


def load_hellaswag(tokenizer, split="validation", limit=200):
    """Load HellaSwag as multiple-choice items, if datasets is there."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split=split)
    items = []
    for row in ds:
        if len(items) >= limit:
            break
        ctx = row["ctx"].strip()
        endings = [e.strip() for e in row["endings"]]
        items.append({"context": ctx, "choices": endings,
                      "answer": int(row["label"])})
    return items


def load_arc(tokenizer, split="validation", limit=200, challenge=True):
    """Load AI2 ARC as multiple-choice items, if datasets is there."""
    from datasets import load_dataset

    config = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    items = []
    for row in ds:
        if len(items) >= limit:
            break
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if row["answerKey"] not in labels:
            continue
        items.append({"context": row["question"].strip(), "choices": texts,
                      "answer": labels.index(row["answerKey"])})
    return items


def load_winogrande(tokenizer, split="validation", limit=200):
    """Load Winogrande as multiple-choice items, if datasets is there."""
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split=split)
    items = []
    for row in ds:
        if len(items) >= limit:
            break
        items.append({"context": row["sentence"].replace("_", "____"),
                      "choices": [row["option1"], row["option2"]],
                      "answer": int(row["answer"]) - 1})
    return items


def run_benchmarks(model, tokenizer, device="cpu", limit=200,
                   which=("hellaswag", "arc", "winogrande")):
    """Run the benchmark suite and return a dict of accuracies.

    Each benchmark is wrapped in try/except, so a missing dataset
    or a download failure skips that one instead of killing the run.
    """
    loaders = {"hellaswag": load_hellaswag, "arc": load_arc,
               "winogrande": load_winogrande}
    out = {}
    for name in which:
        try:
            items = loaders[name](tokenizer, limit=limit)
            res = eval_multiple_choice(model, tokenizer, items, device=device)
            out[name] = {"accuracy": res["accuracy"],
                         "correct": res["correct"], "total": res["total"]}
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


# generation quality
# ------------------

def repetition_score(text):
    """Share of repeated n-grams in a text, 0 to 1.

    A model stuck in a loop repeats the same 4-grams over and over,
    so this number climbs toward 1. Healthy text sits low.
    """
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return 1.0 - unique / len(grams)


def generation_stats(texts):
    """Basic stats over a list of generated texts."""
    if not texts:
        return {}
    lengths = [len(t.split()) for t in texts]
    reps = [repetition_score(t) for t in texts]
    return {
        "n": len(texts),
        "mean_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "mean_repetition": sum(reps) / len(reps),
        "max_repetition": max(reps),
    }


@torch.no_grad()
def sample_completions(model, tokenizer, prompts, max_new=100,
                       temperature=0.8, top_p=0.95, device="cpu"):
    """Generate one completion per prompt, with sampling.

    Keeps it simple: temperature and top-p, no beam search, because
    sampling is what you want when you are reading the output by eye.
    Returns the list of completion strings.
    """
    model.eval()
    outs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **ids, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        outs.append(text[len(prompt):].strip())
    model.train()
    return outs


# full eval run
# -------------

def full_eval(model, tokenizer, device="cpu", limit=200, seq_len=2048,
              val_texts=None, prompts=None):
    """Run everything and return one big results dict.

    Perplexity on val texts, the benchmark suite, and a few sampled
    completions with stats. Pass val_texts and prompts in, or it
    uses small built-in defaults.
    """
    results = {}
    if val_texts:
        results["perplexity"] = eval_perplexity(
            model, tokenizer, val_texts, seq_len=seq_len, device=device)
    results["benchmarks"] = run_benchmarks(
        model, tokenizer, device=device, limit=limit)
    if prompts is None:
        prompts = ["The future of AI is", "Once upon a time",
                   "The capital of France is"]
    completions = sample_completions(model, tokenizer, prompts, device=device)
    results["generation"] = {
        "stats": generation_stats(completions),
        "samples": [{"prompt": p, "completion": c}
                    for p, c in zip(prompts, completions)],
    }
    return results


def print_results(results):
    """Print eval results in a readable way."""
    if "perplexity" in results:
        p = results["perplexity"]
        print(f"perplexity: {p['perplexity']:.2f} (loss {p['loss']:.4f}, "
              f"{p['tokens']:,} tokens)")
    for name, res in results.get("benchmarks", {}).items():
        if "error" in res:
            print(f"{name}: skipped ({res['error']})")
        else:
            print(f"{name}: {res['accuracy'] * 100:.1f}% "
                  f"({res['correct']}/{res['total']})")
    gen = results.get("generation", {})
    stats = gen.get("stats", {})
    if stats:
        print(f"generation: {stats['mean_words']:.0f} words avg, "
              f"repetition {stats['mean_repetition']:.2f}")
    for s in gen.get("samples", []):
        print(f"  prompt: {s['prompt']!r}")
        print(f"  -> {s['completion'][:200]!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--val-file", default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    print(f"loaded {args.model}", flush=True)

    val_texts = None
    if args.val_file:
        with open(args.val_file, encoding="utf-8", errors="ignore") as f:
            val_texts = [l.strip() for l in f if l.strip()]
    results = full_eval(model, tok, device=device, limit=args.limit,
                        seq_len=args.seq_len, val_texts=val_texts)
    print_results(results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
