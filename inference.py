"""Inference for Nyx 1.2. Fast generation with KV cache.

The transformers generate() call works fine for a quick test, but
a real serving setup wants more control: KV caching done right,
batched prompts with left padding, sampling strategies you can
pick per request, and stop strings so the model does not ramble
past the answer.

This file is the middle ground. It is not a full serving engine
like vLLM, it is a clean, readable generator you can drop into a
server or a script and actually understand. For the full 2.1T
checkpoint you still need tensor parallelism or a sharded engine,
this code assumes the model already fits on your GPUs.

A 2.1T checkpoint is about 4.4TB in bf16. On one machine that means
8x H100 with tensor parallel, or a sharded loader. The functions
here take a model object, they do not care how you sharded it.
"""
import math
import time

import torch
import torch.nn.functional as F


# sampling
# --------

def top_k_filter(logits, k):
    """Keep only the top k logits, set the rest to -inf."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    cutoff = values[:, -1].unsqueeze(-1)
    return torch.where(logits < cutoff,
                       torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set with mass p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits.float(), dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    # drop everything after the cumulative mass passes p
    mask = cumsum - probs > p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    """Penalize tokens that already appeared, the CTRL style.

    penalty 1.0 means off. Above 1.0 divides the logit of seen
    tokens, which pushes the model away from repeating itself.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    for batch in range(logits.size(0)):
        seen = set(generated_ids[batch].tolist())
        for tok in seen:
            if logits[batch, tok] > 0:
                logits[batch, tok] /= penalty
            else:
                logits[batch, tok] *= penalty
    return logits


def sample_token(logits, temperature=0.8, top_k=50, top_p=0.95,
                 generated_ids=None, repetition_penalty=1.0):
    """Sample one token from logits with the usual knobs.

    temperature 0 means greedy, pick the argmax. Otherwise apply
    the filters in order: repetition penalty, temperature, top-k,
    top-p, then sample. Returns a tensor of token ids, one per row.
    """
    if generated_ids is not None:
        logits = apply_repetition_penalty(
            logits, generated_ids, penalty=repetition_penalty)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


# stopping
# --------

def build_stop_checker(stop_strings, tokenizer):
    """Return a function that says if a text should stop.

    Stop strings are matched on the decoded text, which is simpler
    than matching token ids and works fine for chat turn ends like
    "\nUser:" or "<|end|>". Returns None if there are no stops.
    """
    if not stop_strings:
        return None

    def should_stop(text):
        for s in stop_strings:
            if s and s in text:
                return True
        return False

    return should_stop


def trim_at_stop(text, stop_strings):
    """Cut a generated text at the first stop string."""
    if not stop_strings:
        return text
    cut = len(text)
    for s in stop_strings:
        if s and s in text:
            cut = min(cut, text.index(s))
    return text[:cut]


# single generation
# -----------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=100, temperature=0.8,
             top_k=50, top_p=0.95, repetition_penalty=1.0,
             stop_strings=None, device="cpu", stream=False):
    """Generate a completion for one prompt, with KV cache.

    The model call uses past_key_values, so each step only runs the
    new token through the model instead of the whole sequence. That
    is the difference between fast and unusable at 112 layers.

    If stream is True, this is a generator that yields the text so
    far after each token, for a streaming API. Otherwise it returns
    the final string.
    """
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = ids["input_ids"]
    attention_mask = ids.get("attention_mask")
    generated = []
    past = None
    should_stop = build_stop_checker(stop_strings, tokenizer)

    def run_stream():
        nonlocal past
        cur = input_ids
        mask = attention_mask
        for _ in range(max_new):
            out = model(input_ids=cur, attention_mask=mask,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_tok = sample_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p,
                generated_ids=[generated] if generated else None,
                repetition_penalty=repetition_penalty)
            tok_id = int(next_tok[0, 0].item())
            if tok_id == tokenizer.eos_token_id:
                break
            generated.append(tok_id)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if should_stop and should_stop(text):
                break
            cur = next_tok
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(next_tok)], dim=-1)
            yield text
        return

    if stream:
        return run_stream()
    text = ""
    for text in run_stream():
        pass
    return trim_at_stop(text, stop_strings)


# batched generation
# -------------------

def left_pad_batch(seqs, pad_id):
    """Left-pad a list of token lists to the same length.

    Left padding matters for generation: the last token of every
    row lines up, so the first generated token comes from the right
    position for every prompt in the batch.
    """
    max_len = max(len(s) for s in seqs)
    padded = []
    masks = []
    for s in seqs:
        pad = max_len - len(s)
        padded.append([pad_id] * pad + list(s))
        masks.append([0] * pad + [1] * len(s))
    return (torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long))


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new=100, temperature=0.8,
                   top_k=50, top_p=0.95, repetition_penalty=1.0,
                   stop_strings=None, device="cpu"):
    """Generate completions for a batch of prompts at once.

    Prompts are left-padded and run together, each with its own KV
    cache. Rows that hit EOS early stop generating but stay in the
    batch, masked out. Returns a list of strings, one per prompt,
    in the same order.
    """
    model.eval()
    encoded = [tokenizer.encode(p) for p in prompts]
    pad_id = tokenizer.pad_token_id or 0
    input_ids, attn = left_pad_batch(encoded, pad_id)
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    batch = len(prompts)
    generated = [[] for _ in range(batch)]
    finished = [False] * batch
    past = None
    cur = input_ids
    mask = attn
    should_stop = build_stop_checker(stop_strings, tokenizer)

    for _ in range(max_new):
        if all(finished):
            break
        out = model(input_ids=cur, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tok = sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            generated_ids=generated, repetition_penalty=repetition_penalty)
        next_tok = next_tok.squeeze(-1)
        for i in range(batch):
            if finished[i]:
                continue
            tok_id = int(next_tok[i].item())
            if tok_id == tokenizer.eos_token_id:
                finished[i] = True
                continue
            generated[i].append(tok_id)
            text = tokenizer.decode(generated[i], skip_special_tokens=True)
            if should_stop and should_stop(text):
                finished[i] = True
        cur = next_tok.unsqueeze(-1)
        mask = torch.cat([mask, torch.ones(batch, 1, dtype=torch.long,
                                           device=device)], dim=-1)

    results = []
    for gen in generated:
        text = tokenizer.decode(gen, skip_special_tokens=True)
        results.append(trim_at_stop(text, stop_strings))
    return results


# chat
# ----

def format_chat(messages, tokenizer):
    """Turn a list of chat turns into one prompt string.

    messages is a list of dicts with role and content, like
    [{"role": "user", "content": "hello"}]. The format is simple
    and readable: each turn is "Role: content" on its own lines,
    ending with "Assistant:" so the model continues from there.
    If the tokenizer has a chat template, that is used instead.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for m in messages:
        role = m["role"].capitalize()
        parts.append(f"{role}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


@torch.no_grad()
def chat(model, tokenizer, messages, **kwargs):
    """One chat turn: format the history, generate a reply."""
    prompt = format_chat(messages, tokenizer)
    return generate(model, tokenizer, prompt, **kwargs)


# throughput
# ----------

def benchmark_generation(model, tokenizer, prompt="The future of AI is",
                         max_new=100, device="cpu", warmup=1, rounds=3):
    """Tokens per second for generation, averaged over rounds.

    Warms up first so the number reflects steady state, not the
    first slow run while caches fill. Returns a dict with tokens
    per second and time per token.
    """
    for _ in range(warmup):
        generate(model, tokenizer, prompt, max_new=10, device=device)
    times = []
    for _ in range(rounds):
        t0 = time.time()
        generate(model, tokenizer, prompt, max_new=max_new, device=device)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    return {
        "tokens": max_new,
        "seconds": avg,
        "tokens_per_second": max_new / avg,
        "ms_per_token": avg / max_new * 1000,
    }


def kv_cache_size(batch, seq_len, layers=112, kv_heads=8, head_dim=128,
                  dtype_bytes=2):
    """KV cache bytes for a batch, both K and V, all layers."""
    per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return batch * seq_len * per_token


def max_batch_for_cache(cache_gb, seq_len, layers=112, kv_heads=8,
                        head_dim=128, dtype_bytes=2):
    """How many sequences fit in a KV cache budget, roughly."""
    per_seq = seq_len * 2 * layers * kv_heads * head_dim * dtype_bytes
    return int(cache_gb * 1e9 // per_seq)
