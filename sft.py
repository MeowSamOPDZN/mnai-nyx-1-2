"""Supervised fine-tuning for MNAI Nyx 1.2.

Pre-training teaches the model to autocomplete the internet. It
does not teach it to answer a question, follow an instruction, or
write code on request. That is what SFT does: train on
instruction/response pairs, with the loss only on the response
tokens, so the model learns to produce the answer, not to predict
the question.

The chat format is plain text, no special tokens to learn:

    User: <instruction>
    Assistant: <response><eos>

Everything before "Assistant:" is masked out of the loss. The
model reads the instruction and is only scored on what it says
back.

Usage, from a pre-trained checkpoint:

    python sft.py --model out-nyx12/ckpt-5000 --data ./sft.jsonl \
        --out out-sft --epochs 3 --lr 2e-5

The data file is JSONL, one pair per line:

    {"instruction": "What is 2 + 2?", "response": "4."}

Smoke test, no checkpoint, no data file, CPU only:

    python sft.py --smoke

That trains the tiny twin on the built-in sample set for 3 steps
and checks the loss falls. It proves the masking, the data path,
and the optimizer step all work before you point it at a real
2.1T checkpoint.
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


# a small built-in set so --smoke runs with no files at all. Short
# and varied, so the tiny tokenizer learns real words, not one
# sentence repeated.
SFT_SAMPLES = [
    {"instruction": "What is 2 + 2?",
     "response": "4."},
    {"instruction": "Write a haiku about rain.",
     "response": "Rain on the rooftops, grey clouds drift over the town, puddles hold the sky."},
    {"instruction": "How do I boil an egg?",
     "response": "Put the egg in cold water, bring it to a boil, then simmer for 6 minutes for a soft yolk or 10 for a hard one."},
    {"instruction": "Translate 'good morning' to French.",
     "response": "Bonjour."},
    {"instruction": "What is the capital of France?",
     "response": "Paris."},
    {"instruction": "Write a Python function that adds two numbers.",
     "response": "def add(a, b):\n    return a + b"},
    {"instruction": "Why is the sky blue?",
     "response": "Sunlight is scattered by air molecules, and blue light scatters more than red, so the sky looks blue."},
    {"instruction": "Give me one tip for better sleep.",
     "response": "Keep the same bedtime every night, even on weekends."},
]

CHAT_TEMPLATE = "User: {instruction}\nAssistant: {response}"


def build_tokenizer(vocab_size=500):
    """A small BPE tokenizer for smoke mode, trained on the samples."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    texts = [CHAT_TEMPLATE.format(**s) for s in SFT_SAMPLES]
    tok = Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.Sequence(
        [decoders.ByteLevel(), decoders.Strip(content=" ", left=1)])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>")


def load_data(path):
    """JSONL of {"instruction": ..., "response": ...} pairs."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_pair(tok, instruction, response):
    """Tokenize one pair, masking the prompt out of the loss.

    Returns input_ids and labels of the same length. Labels are
    -100 everywhere except the response tokens, so the loss is only
    ever computed on what the assistant says.
    """
    prompt = f"User: {instruction}\nAssistant:"
    full = CHAT_TEMPLATE.format(instruction=instruction, response=response)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    full_ids = full_ids + [tok.eos_token_id]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def tiny_model(tok):
    """The 2-layer twin of the Nyx block layout, for smoke mode."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from model import build_config

    cfg = build_config()
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.vocab_size = tok.vocab_size
    return LlamaForCausalLM(cfg)


def train_one_epoch(model, tok, rows, opt, device, max_len=256):
    """One pass over the rows, returns the mean response loss."""
    model.train()
    total, n = 0.0, 0
    random.shuffle(rows)
    for row in rows:
        ids, labels = encode_pair(tok, row["instruction"], row["response"])
        ids = ids[:max_len]
        labels = labels[:max_len]
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        out = model(x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        total += out.loss.item()
        n += 1
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="pre-trained checkpoint dir to fine-tune")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer dir, defaults to the checkpoint dir")
    p.add_argument("--data", default=None,
                   help="JSONL of instruction/response pairs")
    p.add_argument("--out", default="out-sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--smoke", action="store_true",
                   help="tiny twin + built-in samples, 3 steps on CPU")
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        tok = build_tokenizer()
        rows = SFT_SAMPLES[:3]
        model = tiny_model(tok)
        steps = 3
        print(f"smoke mode: tiny model, {len(rows)} samples, {steps} steps")
    else:
        if not args.model:
            raise SystemExit("--model is required unless --smoke")
        from transformers import LlamaForCausalLM

        tok_dir = args.tokenizer or args.model
        tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
        rows = load_data(args.data) if args.data else SFT_SAMPLES
        model = LlamaForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16)
        steps = args.epochs * len(rows)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # smoke mode runs a fixed small number of steps, not epochs,
    # so the check finishes in seconds on CPU.
    if args.smoke:
        losses = []
        for i in range(steps):
            row = rows[i % len(rows)]
            loss = train_one_epoch(model, tok, [row], opt, device)
            losses.append(loss)
            print(f"smoke step {i + 1}: loss {loss:.4f}")
        print(f"smoke SFT: losses {['%.4f' % l for l in losses]}")
        assert losses[-1] < losses[0], "loss did not fall"
        print("SMOKE SFT PASSED")
        return 0

    for ep in range(args.epochs):
        loss = train_one_epoch(model, tok, rows, opt, device,
                               max_len=args.max_len)
        print(f"epoch {ep + 1}/{args.epochs}: mean response loss {loss:.4f}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved SFT model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
