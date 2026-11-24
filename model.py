"""MNAI Nyx 1.2, the 2.1T model. AIMYTOG: AI Model You'll Train On GPU.

You bring the GPU. The repo is the model, the training code, and the
tokenizer. Nothing is pre-trained here. You train it yourself, on your
own cluster, with your own data. That is the whole point.

The data is ALL of the internet. The training stream is the full
FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu.
That is the largest open web corpus there is. A 2.1T model wants
about 40T tokens, more than the whole internet holds, so it trains
on the full 15T, every website, every domain, even the random ones.

The shape
-----------
The shape is ours: 36864 hidden, 144 layers, 288 query heads with only
16 key/value heads, a 110592-wide SwiGLU feed-forward, RoPE positions,
RMSNorm, no biases, tied embeddings, 256000-token vocab, 16384 context.
It adds up to 2,183,775,031,296 parameters, about 2.1T.

Why this shape and not a copy of something else: the GQA ratio keeps
attention cheap at this scale, because the key and value caches stay
small even with a long context. The wide SwiGLU carries most of the
capacity, which is where a decoder spends its parameters anyway.
Tying the output head to the input embeddings saves about 9.4B params
for free, and it keeps the logits tied to the same space the model
reads in, which tends to train a little more stably.

It is built on the standard Llama block layout on purpose, so
llama.cpp can turn a finished checkpoint into a GGUF with its normal
converter, no custom converter needed. Anything that reads a Llama
checkpoint can read this one.

Attention
---------
Each layer does grouped-query attention. 288 query heads share 16 key
and value heads, so each KV head serves 18 query heads. The head dim
is 128 (36864 / 288), which is the usual size that keeps the softmax
well behaved and lets flash attention kernels run at full speed.

Positions come from RoPE, rotary embeddings with theta 500000, which
is the long-context setting. No learned position embeddings, no ALiBi.
RoPE extrapolates a bit past the training context if you ever need it
to, and it costs nothing at inference time.

Feed-forward
------------
SwiGLU, three matrices per layer: gate, up, down. Hidden 36864 goes
up to 110592 and back down. That is about 3.0x the hidden size, a
little narrower than the classic 4x because SwiGLU has three matrices
instead of two, so the parameter budget balances out.

Norms
-----
RMSNorm before attention and before the MLP, no biases anywhere.
One final RMSNorm before the output head. RMSNorm is cheaper than
LayerNorm and at this scale the missing mean-centering does not hurt.

Dtype
-----
Train in bf16. Weights, grads and activations all bf16, with fp32
master weights in the optimizer if your setup keeps them. bf16 has
the same exponent range as fp32, so it does not overflow the way fp16
does, and you do not need loss scaling. That is one less thing to
tune on a run that already has enough knobs.

What this file holds
--------------------
build_config and build_model are the shape. count_params and describe
tell you what you built. The helpers below that are the day-to-day
tools: validate the config before a run starts, print a parameter
breakdown so you can see where the 2.1T goes, estimate training
FLOPs, pick a weight init, turn flash attention on or off, and do
small model surgery like dropping layers or merging two checkpoints.

Any-to-any
----------
The text core above is only one third of the model. multimodal.py
adds the rest: a ViT vision encoder that turns pictures into tokens
the core can read, a DiT image decoder that turns core states back
into pictures, and an audio encoder and decoder that do the same for
sound. Text, image and audio all flow through the same core, so one
model reads and writes all three. build_any_to_any_model wires it
all together. The core dims never change, the modality modules just
scale with them.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

# the model name. Change this to rename the model, see the README
# section "Make your OWN named LLM" for the full rename steps.
MODEL_NAME = "MNAI Nyx 1.2"

# the shape, all in one place so train.py and generate.py agree
VOCAB_SIZE = 256000
HIDDEN_SIZE = 36864
NUM_LAYERS = 144
NUM_HEADS = 288
NUM_KV_HEADS = 16
INTERMEDIATE_SIZE = 110592
MAX_CTX = 16384
ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5

# head dim falls out of hidden / heads, 128 here, same as most big models
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
# how many query heads share one KV head
GQA_GROUPS = NUM_HEADS // NUM_KV_HEADS

# memory per parameter in bytes, for the hardware math in the README
BYTES_BF16 = 2
BYTES_FP32 = 4
BYTES_FP16 = 2
BYTES_INT8 = 1


def build_config(attn_implementation="sdpa"):
    """Build the Llama config for Nyx 1.

    attn_implementation picks the attention kernel. sdpa is the
    default, it uses PyTorch's scaled dot product attention and works
    everywhere. flash_attention_2 is faster on H100s when the
    flash-attn package is installed, pass it through if you have it.
    eager is the plain math fallback, slow but always correct.
    """
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_CTX,
        rope_theta=ROPE_THETA,
        rms_norm_eps=RMS_NORM_EPS,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation=attn_implementation,
    )


def build_model(dtype=torch.bfloat16, attn_implementation="sdpa",
                moe_experts=0, moe_topk=2, moe_capacity_factor=1.25):
    """Build a fresh Nyx 1.2 with random weights.

    dtype None leaves the model on the meta device, which is how you
    count parameters without allocating a terabyte of RAM. Pass
    torch.bfloat16 for a real model, torch.float32 if you want the
    extra precision on a small test.

    moe_experts 0 means the dense model, the default. Pass 8 (with
    moe_topk 2) for the MoE variant: every block's SwiGLU MLP is
    swapped for a MoE layer with that many experts, and each token
    is routed to the top 2. Same input and output shapes, so the
    rest of the model does not change. See MoELayer below.
    """
    cfg = build_config(attn_implementation=attn_implementation)
    model = LlamaForCausalLM(cfg)
    if moe_experts > 0:
        device = next(model.parameters()).device
        with torch.device(device):
            for layer in model.model.layers:
                layer.mlp = MoELayer(HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                     n_experts=moe_experts, top_k=moe_topk,
                                     capacity_factor=moe_capacity_factor)
    if dtype is not None:
        model = model.to(dtype)
    return model


class SwiGLUMLP(nn.Module):
    """One SwiGLU feed-forward, the same shape as the dense block's.

    Three matrices: gate, up, down. Used as a single expert inside
    a MoE layer, so each expert is exactly the dense MLP it replaces.
    """

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward, a drop-in for the dense MLP.

    A router picks the top k experts for each token, each expert is
    a SwiGLU MLP the same size as the dense one it replaces, and the
    outputs are mixed by the router weights. Tokens that pick the
    same expert are processed together. The aux loss is the Switch
    Transformer load-balancing loss: it pushes the router to spread
    tokens evenly so no expert starves and none gets overloaded.
    Add it to the training loss with a small weight, 0.01 is typical.
    The router z-loss (ST-MoE) penalises large router logits, which
    is what makes routing blow up into one expert; the trainer adds
    it with weight 0.001. Each expert also has a token capacity, a
    multiple of its even share, so a collapsed router cannot drown
    one expert in a whole batch.

    Input and output shapes are identical to the dense MLP, so this
    swaps straight into any Llama block. Total params grow with the
    expert count, but active params per token only grow with top k.
    """

    def __init__(self, hidden, intermediate, n_experts=8, top_k=2,
                 capacity_factor=1.25):
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate
        self.n_experts = n_experts
        self.top_k = top_k
        # capacity: no expert takes more than this multiple of its even
        # share of tokens. 1.0 is exactly even, 1.25 leaves headroom
        # for a lopsided batch without letting one expert drown.
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUMLP(hidden, intermediate) for _ in range(n_experts)])
        # set on every forward, read by moe_aux_loss / moe_z_loss
        self.aux_loss = 0.0
        self.z_loss = 0.0

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat)
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        # router z-loss, from ST-MoE: penalises large router logits,
        # which is what makes routing blow up into one expert. A
        # 0.001 weight on this in the trainer is the usual size.
        self.z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        out = torch.zeros_like(flat)
        n_tokens = flat.shape[0]
        # per-expert token cap. Tokens past the cap for an expert are
        # dropped for that expert: they still ride the block's residual
        # stream, they just get no expert output. Without a cap a
        # collapsed router can send a whole batch to one expert and
        # blow its memory and its gradients.
        capacity = max(1, math.ceil(n_tokens * self.top_k / self.n_experts
                                    * self.capacity_factor))
        for k in range(self.top_k):
            idx = topk_idx[:, k]
            w = topk_vals[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                pos = (idx == e).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                if pos.numel() > capacity:
                    pos = pos[:capacity]
                out[pos] = out[pos] + w[pos] * self.experts[e](flat[pos])
        # load balancing: f is the share of tokens each expert got,
        # P is the mean router probability for each expert. If one
        # expert takes everything, f * P is large, so the loss pushes
        # the router to spread out. Kept as a tensor so the router
        # gets gradients from it, f itself is just counts.
        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_experts).float()
            f = counts / max(1, flat_idx.numel())
        p_mean = probs.mean(dim=tuple(range(probs.dim() - 1)))
        self.aux_loss = self.n_experts * (f * p_mean).sum()
        return out.reshape(shape)


def moe_counts(model, moe_experts, moe_topk):
    """Total, active-per-token, and MoE-only params for a MoE model.

    Active params are the ones a single token actually touches:
    everything except the experts, plus the top k share of the
    experts. That is the number that decides how fast and how cheap
    the model runs, while the total is the capacity it holds.
    """
    total, _ = count_params(model)
    moe_params = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, MoELayer):
            moe_params += sum(p.numel() for p in mlp.parameters())
    non_moe = total - moe_params
    active = non_moe + moe_params * moe_topk // moe_experts
    return total, active, moe_params


def moe_aux_loss(model):
    """Sum of the load-balancing aux losses from all MoE layers.

    Each layer sets its aux_loss on forward, as a tensor so the
    router gets gradients. Returns 0.0 for a dense model, which has
    no MoE layers at all.
    """
    total = 0.0
    for layer in model.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, MoELayer):
            total = total + mlp.aux_loss
    return total


def moe_z_loss(model):
    """Sum of the router z-losses from all MoE layers.

    Same shape as moe_aux_loss: each layer sets z_loss on forward,
    the trainer adds it with a small weight (0.001 is typical). It
    keeps router logits small, which is the guard against routing
    collapse that the load-balancing loss alone does not give.
    Returns 0.0 for a dense model.
    """
    total = 0.0
    for layer in model.model.layers:
        mlp = layer.mlp
        if isinstance(mlp, MoELayer):
            total = total + mlp.z_loss
    return total


def moe_router_balance(model):
    """Per-expert token share from the last forward, all MoE layers.

    Returns a list of per-layer dicts with the token count each
    expert got and the worst imbalance (max share / even share).
    Used by validate_cluster.py to check the router is actually
    spreading tokens instead of collapsing onto one expert.
    """
    out = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if not isinstance(mlp, MoELayer):
            continue
        aux = mlp.aux_loss
        if isinstance(aux, torch.Tensor):
            aux = aux.item()
        out.append({"n_experts": mlp.n_experts, "top_k": mlp.top_k,
                    "capacity_factor": mlp.capacity_factor,
                    "aux_loss": aux,
                    "z_loss": mlp.z_loss.item()
                    if isinstance(mlp.z_loss, torch.Tensor) else mlp.z_loss})
    return out


def describe_moe(model, moe_experts, moe_topk):
    """Describe the MoE variant: total vs active params per token."""
    total, active, moe_params = moe_counts(model, moe_experts, moe_topk)
    lines = [
        f"{MODEL_NAME} (MoE, {moe_experts} experts, top-{moe_topk})",
        f"total params {total:,} ({total / 1e12:.3f}T)",
        f"active params per token {active:,} ({active / 1e12:.3f}T)",
        f"MoE params {moe_params:,} ({100.0 * moe_params / total:.1f}% of total)",
        f"a token touches {100.0 * active / total:.1f}% of the model",
    ]
    return "\n".join(lines)


def build_any_to_any_model(dtype=None, attn_implementation="sdpa"):
    """Build the full any-to-any model: Llama core plus modalities.

    The core is the same build_model above. The vision encoder, image
    decoder, audio encoder and audio decoder come from multimodal.py
    and are wired in by AnyToAnyModel. dtype None leaves everything
    on the meta device, which is how you count parameters without
    allocating terabytes of RAM. Pass torch.bfloat16 for a real model.
    """
    from multimodal import AnyToAnyModel

    core = build_model(dtype=dtype, attn_implementation=attn_implementation)
    return AnyToAnyModel(core)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model):
    total, _ = count_params(model)
    lines = [
        f"{MODEL_NAME}",
        f"vocab {VOCAB_SIZE}, hidden {HIDDEN_SIZE}, layers {NUM_LAYERS}",
        f"heads {NUM_HEADS}, kv heads {NUM_KV_HEADS}, ffn {INTERMEDIATE_SIZE}",
        f"context {MAX_CTX}, rope theta {int(ROPE_THETA)}",
        f"params {total:,} ({total / 1e9:.1f}B), tied embeddings: yes",
    ]
    return "\n".join(lines)


def describe_any_to_any(model):
    """Describe the full any-to-any model, core plus all four modules.

    model is an AnyToAnyModel from build_any_to_any_model. Prints the
    core count, each modality module count, and the new total.
    """
    from multimodal import modality_param_counts

    total, _ = count_params(model)
    parts = modality_param_counts()
    core_total = total - parts["modalities_total"]
    lines = [
        f"{MODEL_NAME} (any-to-any)",
        f"core text: {core_total:,} ({core_total / 1e12:.3f}T)",
        f"vision encoder: {parts['vision_encoder']:,}",
        f"image decoder: {parts['image_decoder']:,}",
        f"audio encoder: {parts['audio_encoder']:,}",
        f"audio decoder: {parts['audio_decoder']:,}",
        f"total params {total:,} ({total / 1e12:.3f}T)",
        "modalities: text, image, audio in and out",
    ]
    return "\n".join(lines)


def param_breakdown(model):
    """Show where the parameters live, per component.

    Returns a dict of name to param count, plus a printed table if
    you run it by hand. Useful when someone asks why the model is
    2.1T and you want a real answer instead of a shrug.
    """
    parts = {}
    for name, p in model.named_parameters():
        # top level key is enough: embed, layers.0.self_attn, and so on
        key = name.split(".")[0]
        if "layers" in name:
            # group by layer block type instead of layer number
            segs = name.split(".")
            if "self_attn" in segs:
                key = "attention"
            elif "mlp" in segs:
                key = "mlp"
            elif "norm" in segs[-1] or "input_layernorm" in segs or "post_attention_layernorm" in segs:
                key = "norms"
            else:
                key = "layers-other"
        parts[key] = parts.get(key, 0) + p.numel()
    return parts


def print_breakdown(model):
    total, _ = count_params(model)
    parts = param_breakdown(model)
    print("parameter breakdown:")
    for key in sorted(parts, key=lambda k: -parts[k]):
        n = parts[key]
        print(f"  {key:15s} {n:>15,}  ({100.0 * n / total:5.1f}%)")
    print(f"  {'total':15s} {total:>15,}")


def estimate_flops_per_token(model=None, seq_len=None):
    """Rough training FLOPs per token, the 6*N rule plus attention.

    The classic estimate is 6 FLOPs per parameter per token for a
    training step (2 forward, 4 backward). Attention adds a term that
    grows with context length, so this also adds it in when you pass
    a seq_len. Good enough for planning GPU hours, not a benchmark.
    """
    n = 2_183_775_031_296
    flops = 6 * n
    if seq_len:
        # attention FLOPs per token, roughly 12 * layers * heads * head_dim * seq
        attn = 12 * NUM_LAYERS * NUM_HEADS * HEAD_DIM * seq_len
        flops += attn
    return flops


def estimate_training_flops(tokens, seq_len=None):
    """Total training FLOPs for a run of this many tokens."""
    return estimate_flops_per_token(seq_len=seq_len) * tokens


def memory_for_params(n_params, dtype_bytes=BYTES_BF16, optim="bnb8bit"):
    """Rough memory for weights, grads and optimizer states.

    optim picks the optimizer state size: adamw keeps two fp32 states
    per param (8 bytes), bnb8bit keeps two int8 states (2 bytes),
    adafactor keeps roughly one fp32 state (4 bytes), sgd keeps one
    fp32 momentum state (4 bytes). Weights and grads are bf16 here.
    Returns a dict in bytes, so you can divide by 1e9 for GB.
    """
    optim_bytes = {"adamw": 8, "bnb8bit": 2, "adafactor": 4, "sgd": 4}
    o = optim_bytes.get(optim, 8)
    return {
        "weights": n_params * dtype_bytes,
        "grads": n_params * dtype_bytes,
        "optim": n_params * o,
        "total": n_params * (2 * dtype_bytes + o),
    }


def get_rope_theta(cfg=None):
    """The RoPE theta from a config, across transformers versions.

    Older transformers kept it as cfg.rope_theta. Newer ones nest
    it under cfg.rope_parameters or cfg.rope_scaling. This tries
    each place and falls back to the module constant.
    """
    cfg = cfg or build_config()
    if hasattr(cfg, "rope_theta"):
        try:
            return cfg.rope_theta
        except Exception:
            pass
    for key in ("rope_parameters", "rope_scaling"):
        params = getattr(cfg, key, None)
        if isinstance(params, dict) and "rope_theta" in params:
            return params["rope_theta"]
    return ROPE_THETA


def validate_config(cfg=None):
    """Check the Nyx 1.2 config makes sense before a run starts.

    Catches the dumb mistakes early: hidden not divisible by heads,
    KV heads not dividing query heads, vocab or context set to
    something silly, FFN size that wastes parameters. Returns a list
    of problems, empty means the config is fine.
    """
    cfg = cfg or build_config()
    problems = []
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        problems.append("hidden_size must divide evenly by num_attention_heads")
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
        problems.append("num_attention_heads must divide evenly by num_key_value_heads")
    if cfg.hidden_size // cfg.num_attention_heads < 32:
        problems.append("head dim is very small, attention will be weak")
    if cfg.hidden_size // cfg.num_attention_heads > 256:
        problems.append("head dim is very large, attention will be slow")
    if cfg.intermediate_size < cfg.hidden_size:
        problems.append("intermediate_size smaller than hidden_size, MLP is a bottleneck")
    if cfg.num_hidden_layers < 4:
        problems.append("very few layers for a model this wide")
    if cfg.vocab_size < 1000:
        problems.append("vocab_size looks wrong, did you mean a bigger number?")
    if cfg.max_position_embeddings < 512:
        problems.append("context is very short for a frontier model")
    if get_rope_theta(cfg) < 10000:
        problems.append("rope_theta is low, long context will suffer")
    if not cfg.tie_word_embeddings:
        problems.append("embeddings are untied, that costs ~3B extra params for no gain here")
    return problems


def check_config_or_raise(cfg=None):
    """Same as validate_config, but raises instead of returning."""
    problems = validate_config(cfg)
    if problems:
        raise ValueError("bad model config: " + "; ".join(problems))
    return True


def init_weights(model, strategy="default"):
    """Re-init weights with a chosen strategy.

    default is whatever transformers does, which is a normal init
    scaled by layer count. scaled is the GPT-style init where the
    std shrinks as the model gets deeper, which keeps the residual
    stream from blowing up early in training. xavier and kaiming are
    the classic ones, mostly here for small experiments. Use this
    before training starts, never on a checkpoint you care about.
    """
    if strategy == "default":
        model.apply(model._init_weights)
        return model
    if strategy == "scaled":
        # std shrinks with depth, keeps early loss sane on big models
        std = 0.02 / math.sqrt(2 * NUM_LAYERS)
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=std)
            elif "norm" not in name:
                torch.nn.init.zeros_(p)
        return model
    if strategy == "xavier":
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
        return model
    if strategy == "kaiming":
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        return model
    if strategy == "zeros":
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        return model
    raise ValueError(f"unknown init strategy {strategy!r}, pick default, scaled, xavier, kaiming or zeros")


def scale_init_for_depth(std=0.02):
    """The per-layer std for the scaled init, given the depth."""
    return std / math.sqrt(2 * NUM_LAYERS)


def set_attn_implementation(model, impl="sdpa"):
    """Swap the attention kernel on an already built model.

    sdpa works everywhere. flash_attention_2 needs the flash-attn
    package and a GPU that supports it, and it is the fastest on H100.
    eager is the slow fallback that always works. This just sets the
    config flag, the actual kernel is picked when the model runs.
    """
    allowed = ("sdpa", "flash_attention_2", "eager")
    if impl not in allowed:
        raise ValueError(f"attn {impl!r} not one of {allowed}")
    model.config._attn_implementation = impl
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = impl
    return model


def flash_available():
    """True if flash-attn is installed and usable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def best_attn_implementation():
    """Pick the fastest attention kernel this machine supports."""
    if flash_available() and torch.cuda.is_available():
        return "flash_attention_2"
    return "sdpa"


def num_layers(model):
    return len(model.model.layers)


def hidden_size(model):
    return model.config.hidden_size


def drop_layers(model, keep):
    """Return a copy of the model with only the first `keep` layers.

    Useful for quick ablations, or for carving a smaller model out of
    a big checkpoint to test an idea cheaply. The result is a new
    model object, the original is untouched. The config is updated so
    saving it writes the right layer count.
    """
    import copy

    total = num_layers(model)
    if keep >= total:
        return model
    if keep < 1:
        raise ValueError("keep at least 1 layer")
    new_model = copy.deepcopy(model)
    new_model.model.layers = new_model.model.layers[:keep]
    new_model.config.num_hidden_layers = keep
    return new_model


def merge_checkpoints(model_a, model_b, alpha=0.5):
    """Average the weights of two models, in place on model_a.

    alpha 0.5 is a plain average. This is model soup: averaging two
    fine-tunes of the same base often beats either one alone. Both
    models must have the same shape, and they should share an init,
    otherwise the average is garbage. Returns model_a.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    if set(sd_a) != set(sd_b):
        raise ValueError("models do not have the same keys, cannot merge")
    with torch.no_grad():
        for k in sd_a:
            sd_a[k].mul_(1 - alpha).add_(sd_b[k], alpha=alpha)
    model_a.load_state_dict(sd_a)
    return model_a


def count_flops_for_run(tokens, seq_len=16384):
    """Print a human readable FLOPs and time estimate for a run."""
    flops = estimate_training_flops(tokens, seq_len=seq_len)
    # an H100 does about 700 TFLOPs bf16 in practice with good MFU
    h100_hours = flops / (700e12 * 3600)
    lines = [
        f"tokens: {tokens:,}",
        f"training FLOPs: {flops:.3e}",
        f"rough H100 hours at 700 TFLOPs: {h100_hours:,.0f}",
        f"rough days on 256x H100: {h100_hours / 256 / 24:.1f}",
    ]
    return "\n".join(lines)


def model_summary(model):
    """One block of text describing the model, for logs and README."""
    total, trainable = count_params(model)
    mem = memory_for_params(total)
    lines = [
        "MNAI Nyx 1.2",
        f"  params: {total:,} ({total / 1e9:.1f}B)",
        f"  trainable: {trainable:,}",
        f"  hidden {HIDDEN_SIZE}, layers {NUM_LAYERS}, heads {NUM_HEADS}/{NUM_KV_HEADS}",
        f"  ffn {INTERMEDIATE_SIZE}, vocab {VOCAB_SIZE}, ctx {MAX_CTX}",
        f"  weights bf16: {mem['weights'] / 1e9:.0f} GB",
        f"  weights + grads + 8-bit Adam: {mem['total'] / 1e9:.0f} GB",
        f"  FLOPs per token: {estimate_flops_per_token():.3e}",
    ]
    return "\n".join(lines)


def save_config_only(out_dir):
    """Write just the config, no weights, so other tools can read the shape."""
    cfg = build_config()
    os.makedirs(out_dir, exist_ok=True)
    cfg.save_pretrained(out_dir)
    return os.path.join(out_dir, "config.json")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--moe-experts", type=int, default=0,
                   help="0 = dense (default), 8 = MoE with 8 experts per layer")
    p.add_argument("--moe-topk", type=int, default=2,
                   help="how many experts each token is routed to")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    # meta device: count params without allocating 1TB of RAM
    with torch.device("meta"):
        model = build_model(dtype=None)
    print(describe(model))
    print()
    print_breakdown(model)
    print()
    print("FLOPs per token:", f"{estimate_flops_per_token(seq_len=16384):.3e}")
    print()
    print(count_flops_for_run(10_000_000_000_000))
    print()
    print("this model is 2.1T params. Do not build it on one GPU.")
    print("train.py uses FSDP to shard it across a cluster.")
    print("config problems:", validate_config() or "none")
    print()
    # any-to-any: core plus vision, image decoder, audio in and out
    with torch.device("meta"):
        full = build_any_to_any_model(dtype=None)
    print(describe_any_to_any(full))
    print()
    print("any-to-any total is core plus all four modality modules.")
    if args.moe_experts > 0:
        # MoE variant: same core, every MLP swapped for a MoE layer
        print()
        with torch.device("meta"):
            moe_model = build_model(dtype=None, moe_experts=args.moe_experts,
                                    moe_topk=args.moe_topk)
        print(describe_moe(moe_model, args.moe_experts, args.moe_topk))
        print()
        print("MoE total is capacity, active per token is what a token")
        print("actually runs through. Dense stays the default.")
    # save the config next to the code so tools agree on the shape
    cfg = build_config()
    cfg.save_pretrained(os.path.join(here, "hf_config_only"))
    print("wrote hf_config_only/config.json (config only, no weights)")
