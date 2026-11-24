"""Direct Preference Optimization for MNAI Nyx 1.2.

SFT teaches the model to answer. DPO teaches it to answer well:
given a prompt and two replies, one chosen and one rejected, the
model is pushed to prefer the chosen one. No reward model, no
sampling loop, no PPO — just a loss on pairs, which is why DPO is
the post-training step this repo ships instead of full RLHF. GRPO
is the follow-up if you want on-policy RL later; the README's
post-training section says where it fits.

The loss, for one pair: take the log-probability of the chosen
reply and of the rejected reply under the policy model, subtract
the same two numbers under a frozen reference model (the SFT
checkpoint you started from), and push the gap up:

    loss = -log sigmoid(beta * ((logp_c - logp_c_ref)
                                - (logp_r - logp_r_ref)))

The reference stops the policy from drifting away from the SFT
model just to game the pair. beta 0.1 is the usual size.

The chat format is the same as sft.py, plain text:

    User: <prompt>
    Assistant: <reply><eos>

The loss only ever sees the reply tokens, never the prompt.

Usage, from an SFT checkpoint:

    python dpo.py --model out-sft --data ./dpo.jsonl \
        --out out-dpo --epochs 1 --lr 5e-7

The data file is JSONL, one pair per line:

    {"prompt": "What is 2 + 2?", "chosen": "4.",
     "rejected": "I don't know, maybe 5?"}

Smoke test, no checkpoint, no data file, CPU only:

    python dpo.py --smoke

That trains the tiny twin on the built-in pairs for 3 steps and
checks the loss falls and the model prefers the chosen reply more
afterwards. It proves the pairing, the log-prob math, and the
optimizer step all work before you point it at a real checkpoint.
"""

import argparse
import copy
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


# a small built-in set so --smoke runs with no files at all. Each
# pair has a clearly better chosen reply, so the check is real: the
# model has to learn to prefer it, not just move the loss.
DPO_SAMPLES = [
    {"prompt": "What is 2 + 2?",
     "chosen": "4.",
     "rejected": "I don't know, maybe 5?"},
    {"prompt": "Write a haiku about rain.",
     "chosen": "Rain on the rooftops, grey clouds drift over the town, puddles hold the sky.",
     "rejected": "rain rain rain rain rain rain rain"},
    {"prompt": "How do I boil an egg?",
     "chosen": "Put the egg in cold water, bring it to a boil, then simmer for 6 minutes for a soft yolk or 10 for a hard one.",
     "rejected": "Just eat it raw, cooking is a waste of time."},
    {"prompt": "Translate 'good morning' to French.",
     "chosen": "Bonjour.",
     "rejected": "Good morning in French is good morning."},
    {"prompt": "What is the capital of France?",
     "chosen": "Paris.",
     "rejected": "London, I think. Or Berlin."},
    {"prompt": "Write a Python function that adds two numbers.",
     "chosen": "def add(a, b):\n    return a + b",
     "rejected": "def add(a, b):\n    return a - b"},
]

CHAT_TEMPLATE = "User: {prompt}\nAssistant: {reply}"


def build_tokenizer(vocab_size=500):
    """A small BPE tokenizer for smoke mode, trained on the samples."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    texts = []
    for s in DPO_SAMPLES:
        texts.append(CHAT_TEMPLATE.format(prompt=s["prompt"], reply=s["chosen"]))
        texts.append(CHAT_TEMPLATE.format(prompt=s["prompt"], reply=s["rejected"]))
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
    """JSONL of {"prompt": ..., "chosen": ..., "rejected": ...}."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_reply(tok, prompt, reply):
    """Tokenize one reply, masking the prompt out of the log-probs.

    Returns input_ids and a mask of the same length. The mask is 1
    only on the reply tokens, so the log-prob sum only ever counts
    what the assistant says, never the prompt it was given.
    """
    prompt_text = f"User: {prompt}\nAssistant:"
    full_text = CHAT_TEMPLATE.format(prompt=prompt, reply=reply)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    full_ids = full_ids + [tok.eos_token_id]
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return full_ids, mask


def completion_logprob(model, tok, prompt, reply, device, max_len=256):
    """Sum of log-probs of the reply tokens under this model."""
    ids, mask = encode_reply(tok, prompt, reply)
    ids = ids[:max_len]
    mask = mask[:max_len]
    x = torch.tensor([ids], device=device)
    logits = model(x).logits
    # logits at position t predict token t+1, so shift both sides
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    target = x[0, 1:]
    token_logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = torch.tensor(mask[1:], dtype=torch.float32, device=device)
    return (token_logp * mask).sum()


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


def dpo_loss(policy, ref, tok, row, beta, device, max_len=256):
    """One pair's DPO loss, plus the reward gap for logging."""
    prompt = row["prompt"]
    logp_c = completion_logprob(policy, tok, prompt, row["chosen"],
                                device, max_len)
    logp_r = completion_logprob(policy, tok, prompt, row["rejected"],
                                device, max_len)
    with torch.no_grad():
        ref_c = completion_logprob(ref, tok, prompt, row["chosen"],
                                   device, max_len)
        ref_r = completion_logprob(ref, tok, prompt, row["rejected"],
                                   device, max_len)
    gap = (logp_c - ref_c) - (logp_r - ref_r)
    loss = -F.logsigmoid(beta * gap)
    return loss, gap.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="SFT checkpoint dir to align")
    p.add_argument("--ref", default=None,
                   help="reference checkpoint dir, defaults to --model")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer dir, defaults to the checkpoint dir")
    p.add_argument("--data", default=None,
                   help="JSONL of prompt/chosen/rejected triples")
    p.add_argument("--out", default="out-dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1,
                   help="how hard to push the chosen/rejected gap, 0.1 is usual")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--smoke", action="store_true",
                   help="tiny twin + built-in pairs, 3 steps on CPU")
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        tok = build_tokenizer()
        rows = DPO_SAMPLES[:3]
        model = tiny_model(tok)
        steps = 3
        print(f"smoke mode: tiny model, {len(rows)} pairs, {steps} steps")
    else:
        if not args.model:
            raise SystemExit("--model is required unless --smoke")
        from transformers import LlamaForCausalLM

        tok_dir = args.tokenizer or args.model
        tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
        rows = load_data(args.data) if args.data else DPO_SAMPLES
        model = LlamaForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16)
        steps = args.epochs * len(rows)

    model.to(device)
    # the reference is a frozen copy of the starting model. It never
    # trains, it just anchors the policy so DPO cannot drift away
    # from the SFT model to game the pairs.
    ref = copy.deepcopy(model)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)

    # smoke mode uses a much bigger LR than the real default: 5e-7 is
    # right for a 2.1T model over an epoch, but it moves a tiny model
    # by nothing in 3 steps, so the check would measure noise, not
    # learning. The production default is untouched.
    lr = 1e-4 if args.smoke else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if args.smoke:
        # smoke trains on one fixed pair for all 3 steps, and checks
        # the loss on that pair falls and the gap grows. Cycling a
        # different pair each step would compare gaps across pairs,
        # which is meaningless on a tiny random model: one big AdamW
        # step on pair A can push pair B's gap either way, so the
        # check would measure noise, not learning.
        probe = rows[0]
        losses, gaps = [], []
        for i in range(steps):
            loss, gap = dpo_loss(model, ref, tok, probe, args.beta, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            gaps.append(gap)
            print(f"smoke step {i + 1}: loss {loss.item():.4f}, "
                  f"reward gap {gap:.4f}")
        print(f"smoke DPO: losses {['%.4f' % l for l in losses]}, "
              f"gaps {['%.4f' % g for g in gaps]}")
        assert losses[-1] < losses[0], "loss did not fall"
        assert gaps[-1] > gaps[0], "chosen/rejected gap did not grow"
        print("SMOKE DPO PASSED")
        return 0

    for ep in range(args.epochs):
        random.shuffle(rows)
        total, n = 0.0, 0
        for row in rows:
            loss, gap = dpo_loss(model, ref, tok, row, args.beta, device,
                                 max_len=args.max_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item()
            n += 1
        print(f"epoch {ep + 1}/{args.epochs}: "
              f"mean DPO loss {total / max(1, n):.4f}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved DPO model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
