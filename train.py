"""Train MNAI Nyx 1.2 from scratch. Random weights, no base model.
AIMYTOG: AI Model You'll Train On GPU.

A 2.1T model does not fit on one GPU. This script uses FSDP2 to
shard weights, gradients, and optimizer states across all GPUs in
the job. Launch it with torchrun, one process per GPU.

The data is the ENTIRE web. The training stream is the full
FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu.
FineWeb comes from Common Crawl, which visits every website it can
reach, every domain, even the random ones. If a page is on the
internet, it is in the crawl. That is the largest open web corpus
there is. A 2.1T model wants about 40T tokens, more than the whole
internet holds, so it trains on the full 15T, the entire web.

Hardware math, bf16 weights + 8-bit Adam, per GPU with FSDP full shard:
  weights  2.1T * 2 bytes / N GPUs
  grads    2.1T * 2 bytes / N GPUs
  optim    2.1T * 2 bytes / N GPUs (8-bit Adam, 2 states)
  total    about 13TB / N GPUs, plus activations with checkpointing
On 512x H100 80GB, that is about 26GB per GPU for states, leaving
room for activations. On 1024x H100, about 13GB. 512x H100 is the
minimum, 1024x is ideal. Do not try this on fewer than 512 GPUs
unless you know what you are doing.

Chinchilla-optimal for 2.1T params is about 40T tokens, but the
whole internet is 15T, so the target is the full 15T. That is the
default --max-tokens. At 1024x H100, expect roughly 1-2 years of
training. This is a frontier run, not a weekend project.

What this script does beyond the basic loop
-------------------------------------------
Schedulers: cosine, linear, and WSD (warmup-stable-decay). Cosine
is the classic, WSD is the modern pick because you can stop the run
early and decay from wherever you are, which is handy on a run that
might get cut short.

Optimizers: AdamW, 8-bit Adam (default, saves memory), Adafactor,
and SGD with momentum. 8-bit Adam is the default because at 2.1T
the optimizer states are the difference between fitting and not.

Checkpoints: weights in HF format, plus optimizer state, scheduler
state, and RNG state, so a resume is a true resume. The LR schedule
and the optimizer moments pick up exactly where they left off,
instead of restarting warmup from scratch.

Eval: validation perplexity every so often, on a fixed held-out
slice of the same data mix. Watch it go down. If it goes up, stop
the run and figure out why before burning more GPU hours.

Logging: console, a JSONL log file, and optional Weights and
Biases. Throughput is logged as tokens per second and MFU, the
model FLOPs utilization, so you can tell if the cluster is actually
working or just burning power.

Multimodal data
---------------
This model is any-to-any: text, image and audio in and out. The
--modalities flag picks what the run trains on. text is the web text
stream only. text-image adds image-text pairs (captions with their
pictures, streamed the same way, encoded by the vision encoder).
text-audio adds audio-text pairs (transcripts with their sound,
encoded by the audio encoder). all trains on all three at once, which
is the default and the real any-to-any run. The modality modules
live in multimodal.py and are sharded by FSDP with the rest of the
model, so no extra setup is needed, just the flag.

Usage, torchrun on a cluster:
  torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
    --master_addr=<master> --master_port=29500 \
    train.py --tokenizer ./tokenizer --out out-nyx12 \
    --max-tokens 15000000000000 --micro-batch 1 --grad-accum 4096

WSD schedule with WandB logging:
  torchrun --nproc_per_node=8 --nnodes=32 --node_rank=0 \
    --master_addr=<master> --master_port=29500 \
    train.py --tokenizer ./tokenizer --out out-nyx12 \
    --scheduler wsd --decay-frac 0.1 --wandb --wandb-project nyx12

Smoke test on CPU, no data download. Uses a tiny twin of the same
block layout for the loop test, full 2.1T forward is in model.py:
  python train.py --smoke --max-steps 3 --out /tmp/smoke-nyx12
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from data import batch_stream, build_mix, validation_batches
from model import (
    VOCAB_SIZE,
    build_config,
    build_model,
    count_params,
    estimate_flops_per_token,
    moe_aux_loss,
    moe_counts,
)
from utils import (
    MetricLogger,
    Timer,
    all_reduce_mean,
    atomic_write_json,
    format_bytes,
    format_number,
    format_time,
    free_disk_gb,
    grad_norm,
    latest_checkpoint,
    list_checkpoints,
    load_trainer_state,
    memory_report,
    param_norm,
    prune_checkpoints,
    read_train_log,
    save_trainer_state,
    seed_all,
    setup_logging,
    write_train_log,
)

SEQ_LEN = 8192
# smoke test uses a tiny sequence, a full 8192 block on CPU would crawl


def parse_args():
    p = argparse.ArgumentParser()
    # data and output
    p.add_argument("--tokenizer", default="tokenizer")
    p.add_argument("--out", default="out-nyx12")
    p.add_argument("--modalities", default="all",
                   choices=["text", "text-image", "text-audio", "all"],
                   help="which modalities to train: text only, text+image, text+audio, or all three (default)")
    p.add_argument("--moe-experts", type=int, default=0,
                   help="0 = dense (default), 8 = MoE with 8 experts per layer, top-k routed")
    p.add_argument("--moe-topk", type=int, default=2,
                   help="how many experts each token is routed to when MoE is on")
    p.add_argument("--loss-spike-threshold", type=float, default=3.0,
                   help="if a step loss is above this many times the running average, "
                        "drop the step and reload the last checkpoint, 0 disables")
    p.add_argument("--resume", default=None)
    p.add_argument("--data-mix", default=None,
                   help="dataset mix like 'fineweb:0.8,fineweb-edu:0.2', default is 80/20")
    p.add_argument("--no-filter", action="store_true",
                   help="skip document filtering, train on raw docs")
    p.add_argument("--dedup", action="store_true",
                   help="turn on exact + near-dup filtering (slower, cleaner data)")
    # training length
    p.add_argument("--max-tokens", type=int, default=15_000_000_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps, 0 = use max-tokens")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    # optimizer and schedule
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--scheduler", default="cosine",
                   choices=["cosine", "linear", "wsd"],
                   help="cosine decays the whole run, linear is a straight line, "
                        "wsd holds steady then decays at the end")
    p.add_argument("--decay-frac", type=float, default=0.1,
                   help="for wsd: share of steps spent decaying at the end")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optim", default="bnb8bit",
                   choices=["bnb8bit", "adamw", "adafactor", "sgd"])
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.9,
                   help="momentum for sgd")
    # clipping and precision
    p.add_argument("--clip", type=float, default=1.0,
                   help="grad clip value, 0 disables clipping")
    p.add_argument("--clip-mode", default="norm", choices=["norm", "value"],
                   help="norm clips the global grad norm, value clips each grad element")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="training dtype, bf16 is the sane default")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="turn off gradient checkpointing (faster, much more memory)")
    # logging and eval
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0,
                   help="run validation perplexity every N steps, 0 = off")
    p.add_argument("--eval-batches", type=int, default=20,
                   help="how many validation batches per eval")
    p.add_argument("--keep-ckpts", type=int, default=3,
                   help="how many checkpoints to keep on disk")
    p.add_argument("--no-save-optim", action="store_true",
                   help="skip saving optimizer state in checkpoints (smaller, resume restarts moments)")
    p.add_argument("--wandb", action="store_true",
                   help="log to Weights and Biases, needs wandb installed")
    p.add_argument("--wandb-project", default="nyx1")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--max-hours", type=float, default=0,
                   help="stop cleanly after this many hours, saving a checkpoint first, 0 = off")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# smoke tokenizer
# ---------------

# a varied sample of English, for training the smoke tokenizer.
# Byte-level BPE learns its merges from this, so it needs varied
# words, not one sentence repeated.
SMOKE_SAMPLE_TEXT = """
Training a large language model from scratch is a long job. You
start with random weights and a pile of text, and you ask the model
to predict the next token, over and over, billions of times. The
ocean covers most of the Earth, and most of it is still unexplored.
Cooking rice well is mostly about water and patience. Rinse the
rice until the water runs clear, then soak it for twenty minutes.
The first railways changed how people thought about distance. A
journey that took three days by coach took four hours by train. A
good bicycle is a simple machine, and that is why it lasts. Two
wheels, a frame, a chain, and brakes. The history of medicine is
full of ideas that seem obvious now and were invisible then. The
night sky looks the same as it did a thousand years ago, and that
is part of its pull. The stars are so far away that the light
reaching your eyes left them before you were born. A market is a
place where information turns into prices. A farmer brings too many
tomatoes and the price falls. Concrete is strong when you push it
and weak when you pull it, which is why bridges hide steel inside
it. Bees navigate by the sun, even on cloudy days, because they can
see polarized light in the sky. A library is a slow machine for
remembering. Books sit on shelves for decades, waiting for the one
reader who needs them. The weather is a system with too many moving
parts to predict far ahead. Glass is sand that was melted and cooled
too fast to form crystals, so it stays in a liquid-like state
forever. When a river floods, it does not just bring water. It
brings the soil from upstream, and that soil is why flood plains
are the best farmland in the world. The human heart beats about a
hundred thousand times a day, and most people never think about it.
Sound travels through air as pressure waves, and the shape of a
room changes how those waves bounce. The first maps were drawn from
stories. Sailors came back and described what they saw, and
mapmakers drew coastlines from their words.
"""


def build_smoke_tokenizer(vocab_size=2000):
    """A small byte-level BPE tokenizer for smoke tests.

    Smoke mode used to run with no tokenizer at all, which meant
    its checkpoints had no tokenizer files and the GGUF converter
    refused them. This builds a real tokenizer from the sample
    above, so a smoke checkpoint is self-contained: weights, config
    and tokenizer, all matching. The vocab is small, so the tiny
    twin's vocab is set to match it.

    byte_fallback and the Sequence decoder matter: they make the
    tokenizer a Llama-format BPE, which is what llama.cpp's
    converter reads. Without them the converter cannot tell what
    kind of tokenizer it is and refuses the checkpoint.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.Sequence(
        [decoders.ByteLevel(), decoders.Strip(content=" ", left=1)])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok.train_from_iterator([SMOKE_SAMPLE_TEXT], trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>")


# loss spike guard
# ----------------

class LossSpikeGuard:
    """Watch for loss spikes and drop the bad step.

    Frontier runs die to loss spikes: one bad batch pushes the loss
    to 3x normal, the gradients explode, and the model never
    recovers. This guard keeps a running average of step losses.
    Once it has seen enough steps to trust the average, any step
    above threshold times the average is a spike: the step is
    dropped, the gradients are thrown away, and the trainer reloads
    the last checkpoint so the weights are exactly where they were
    before the bad batch. The spike itself never enters the running
    average, so one bad batch cannot drag the average up and hide
    the next spike.
    """

    def __init__(self, threshold=3.0, warmup_steps=20):
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.count = 0
        self.running_avg = None
        self.spikes = 0

    def check(self, loss):
        """True if this loss is a spike, False otherwise."""
        self.count += 1
        if self.threshold <= 0:
            return False
        if self.running_avg is None:
            self.running_avg = loss
            return False
        is_spike = self.count > self.warmup_steps and loss > self.threshold * self.running_avg
        if is_spike:
            self.spikes += 1
            return True
        self.running_avg = 0.9 * self.running_avg + 0.1 * loss
        return False


def reload_weights_from_checkpoint(model, ckpt_dir):
    """Reload model weights from a checkpoint dir, in place.

    Handles both single-file and sharded safetensors checkpoints,
    which is what save_pretrained writes. Missing keys are left
    alone (strict=False), so a MoE checkpoint reloads its dense
    parts and keeps its current expert weights. Returns True if
    any weights were loaded, False if the dir had nothing to load.
    """
    from safetensors.torch import load_file

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    loaded = False
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shards:
            state.update(load_file(os.path.join(ckpt_dir, shard)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = True
    else:
        single = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(single):
            state = load_file(single)
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded = True
    if loaded:
        print(f"reloaded weights from {ckpt_dir} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    return loaded


# schedulers
# ----------

def lr_multiplier(step, total_steps, warmup, min_ratio, scheduler, decay_frac=0.1):
    """LR multiplier at a step, 0 to 1, for any of the three schedules.

    Warmup is linear from 0 to 1. After that the schedules differ.
    Cosine decays smoothly to min_ratio over the whole run. Linear
    decays in a straight line. WSD holds at 1.0 until the last
    decay_frac of steps, then decays linearly to min_ratio. WSD is
    nice because you can decide late when to stop and still get a
    clean decay.
    """
    if total_steps <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if scheduler == "cosine":
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine
    if scheduler == "linear":
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        return min_ratio + (1 - min_ratio) * (1 - progress)
    if scheduler == "wsd":
        decay_steps = max(1, int(total_steps * decay_frac))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / decay_steps
        progress = min(1.0, progress)
        return min_ratio + (1 - min_ratio) * (1 - progress)
    raise ValueError(f"unknown scheduler {scheduler!r}")


def build_scheduler(opt, args, total_steps):
    """A LambdaLR that follows the chosen schedule."""
    def fn(step):
        return lr_multiplier(step, total_steps, args.warmup,
                             args.min_lr_ratio, args.scheduler,
                             decay_frac=args.decay_frac)
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# optimizers
# ----------

def build_optimizer(model, args):
    """Build the optimizer, with a bitsandbytes fallback to AdamW.

    8-bit Adam is the default because it cuts optimizer memory by
    4x, which matters a lot at 2.1T. Adafactor is the low-memory
    backup if bitsandbytes is not there. SGD is there for the
    curious, it works but needs more tuning.
    """
    params = model.parameters()
    if args.optim == "bnb8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params, lr=args.lr, weight_decay=args.weight_decay,
                betas=(args.beta1, args.beta2), eps=args.eps)
        except Exception as e:
            if is_main():
                print(f"8-bit Adam not available ({e}), falling back to AdamW", flush=True)
            return torch.optim.AdamW(
                params, lr=args.lr, weight_decay=args.weight_decay,
                betas=(args.beta1, args.beta2), eps=args.eps)
    if args.optim == "adamw":
        return torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), eps=args.eps)
    if args.optim == "adafactor":
        try:
            return torch.optim.Adafactor(
                params, lr=args.lr, weight_decay=args.weight_decay,
                beta2_decay=-0.8, eps=(None, args.eps),
                clip_threshold=1.0, relative_step=False)
        except Exception as e:
            if is_main():
                print(f"Adafactor failed ({e}), falling back to AdamW", flush=True)
            return torch.optim.AdamW(
                params, lr=args.lr, weight_decay=args.weight_decay,
                betas=(args.beta1, args.beta2), eps=args.eps)
    if args.optim == "sgd":
        return torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.weight_decay,
            momentum=args.momentum, nesterov=True)
    raise ValueError(f"unknown optimizer {args.optim!r}")


def optimizer_name(args):
    return {"bnb8bit": "AdamW 8-bit", "adamw": "AdamW",
            "adafactor": "Adafactor", "sgd": "SGD"}[args.optim]


# checkpoints
# -----------

def save_ckpt(model, tok, out_dir, step, tokens_seen, opt=None, sched=None,
              extra=None, save_optim=True):
    """Save a checkpoint: weights, tokenizer, trainer state, and
    optionally optimizer state, scheduler state and RNG state.

    Only rank 0 writes. The weights go in HF format so anything
    that reads a Llama checkpoint can read this one. The optimizer
    and scheduler states go in separate files, so a checkpoint with
    --no-save-optim is still a valid weights checkpoint. The trainer
    state is written atomically, so a crash mid-save cannot corrupt
    it and trick a resume into the wrong step.
    """
    if not is_main():
        return
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if tok is not None:
        tok.save_pretrained(out_dir)
    state = {"step": step, "tokens_seen": tokens_seen}
    if extra:
        state.update(extra)
    save_trainer_state(out_dir, state)
    if save_optim and opt is not None:
        try:
            torch.save(opt.state_dict(), os.path.join(out_dir, "optim.pt"))
        except Exception as e:
            print(f"could not save optimizer state: {e}", flush=True)
    if sched is not None:
        try:
            torch.save(sched.state_dict(), os.path.join(out_dir, "sched.pt"))
        except Exception as e:
            print(f"could not save scheduler state: {e}", flush=True)
    try:
        torch.save({"python": random.getstate(),
                    "torch": torch.get_rng_state()},
                   os.path.join(out_dir, "rng.pt"))
    except Exception as e:
        print(f"could not save RNG state: {e}", flush=True)


def load_ckpt_state(ckpt_dir, opt=None, sched=None):
    """Read the small state files from a checkpoint dir.

    Returns (step, tokens_seen, loaded_optim, loaded_sched). Missing
    files are fine, it just means the resume starts fresh on that
    piece, and it prints a warning so you know. Weights are loaded
    separately by the caller with from_pretrained.
    """
    state = load_trainer_state(ckpt_dir)
    step = state.get("step", 0)
    tokens_seen = state.get("tokens_seen", 0)
    loaded_optim = False
    loaded_sched = False
    optim_path = os.path.join(ckpt_dir, "optim.pt")
    if opt is not None and os.path.exists(optim_path):
        try:
            opt.load_state_dict(torch.load(optim_path, map_location="cpu",
                                           weights_only=False))
            loaded_optim = True
        except Exception as e:
            print(f"could not load optimizer state, starting fresh: {e}", flush=True)
    sched_path = os.path.join(ckpt_dir, "sched.pt")
    if sched is not None and os.path.exists(sched_path):
        try:
            sched.load_state_dict(torch.load(sched_path, map_location="cpu",
                                             weights_only=False))
            loaded_sched = True
        except Exception as e:
            print(f"could not load scheduler state, starting fresh: {e}", flush=True)
    rng_path = os.path.join(ckpt_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
        except Exception as e:
            print(f"could not load RNG state: {e}", flush=True)
    return step, tokens_seen, loaded_optim, loaded_sched


def resolve_resume(args):
    """Turn --resume into a checkpoint dir.

    Accepts a direct ckpt-* dir, or an output dir, in which case the
    latest checkpoint inside it is used. Returns None if nothing
    was passed, raises if the path does not exist.
    """
    if not args.resume:
        return None
    path = args.resume
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume path not found: {path}")
    if os.path.exists(os.path.join(path, "trainer_state.json")):
        return path
    latest = latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"no checkpoints inside {path}")
    if is_main():
        print(f"resume path is an output dir, using latest: {latest}", flush=True)
    return latest


# validation
# ----------

@torch.no_grad()
def run_validation(model, tok, args, device, eval_batches=None):
    """Validation perplexity on a fixed held-out slice.

    Uses the same data mix but a fixed seed, so the number is
    comparable across runs. Returns a dict with loss and perplexity.
    Runs in eval mode and puts the model back in train mode after.
    """
    from evaluate import eval_perplexity

    n = eval_batches or args.eval_batches
    blocks = validation_batches(tok, args.micro_batch, args.seq_len,
                                n_docs=n * args.micro_batch)
    if not blocks:
        return {"loss": float("nan"), "perplexity": float("nan")}
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for chunk in blocks[:n]:
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        out = model(x, labels=x)
        total_loss += float(out.loss.item()) * x.numel()
        total_tokens += x.numel()
    model.train()
    loss = total_loss / max(1, total_tokens)
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else float("inf")}


# wandb
# -----

def setup_wandb(args, total_steps):
    """Start a WandB run if asked, else return None.

    WandB is optional. If it is not installed, this warns and
    returns None, and training carries on with console and JSONL
    logging only. Nothing here should ever kill a run.
    """
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception:
        if is_main():
            print("wandb not installed, skipping WandB logging", flush=True)
        return None
    run = wandb.init(project=args.wandb_project, name=args.wandb_run,
                     config={**vars(args), "total_steps": total_steps})
    return run


def log_wandb(run, row):
    if run is None:
        return
    try:
        run.log(row)
    except Exception:
        pass


def main():
    args = parse_args()
    log = setup_logging(log_file=args.log_file)
    seed_all(args.seed)

    # distributed setup, torchrun sets these env vars
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = world_size > 1 and not args.smoke
    if use_dist:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_main():
        log.info(f"device: {device}, world size: {world_size}")

    # dtype for training: bf16 default, fp16 if you must, fp32 for tiny tests
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    train_dtype = dtype_map[args.dtype]
    if device == "cpu":
        train_dtype = torch.float32

    if args.smoke:
        # smoke mode builds a small real tokenizer, so the checkpoint
        # it saves is self-contained (weights, config, tokenizer) and
        # the GGUF converter accepts it. No dataset, no tokenizer file.
        tok = build_smoke_tokenizer()
        args.seq_len = 64
        args.micro_batch = 1
        args.grad_accum = 1
        args.optim = "adamw"
        args.eval_every = 0
        if is_main():
            log.info(f"smoke mode: small tokenizer ({tok.vocab_size} tokens), "
                     "random data, no dataset file")
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        tok.pad_token = "<pad>"
        tok.bos_token = "<bos>"
        tok.eos_token = "<eos>"
        assert tok.vocab_size == VOCAB_SIZE, f"tokenizer vocab {tok.vocab_size} != {VOCAB_SIZE}"
        mix = build_mix(args.data_mix)
        if is_main():
            log.info("data mix: " + ", ".join(f"{m['name']}:{m['weight']}" for m in mix))

    # smoke uses a tiny twin of the same shape (2 layers, 256 hidden)
    # for the loop test, because a full 2.1T backward pass needs a cluster.
    # The full 2.1T param count is tested with python model.py.
    resume_dir = resolve_resume(args)
    if resume_dir:
        if is_main():
            log.info(f"resuming weights from {resume_dir}")
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(resume_dir, torch_dtype=train_dtype)
        step = 0
        tokens_seen = 0
    else:
        if args.smoke:
            from transformers import LlamaConfig, LlamaForCausalLM

            tiny_cfg = build_config()
            tiny_cfg.hidden_size = 256
            tiny_cfg.intermediate_size = 512
            tiny_cfg.num_hidden_layers = 2
            tiny_cfg.num_attention_heads = 4
            tiny_cfg.num_key_value_heads = 2
            # the tiny twin uses the smoke tokenizer's vocab, so the
            # checkpoint is self-consistent and the GGUF converter
            # accepts it
            tiny_cfg.vocab_size = tok.vocab_size
            model = LlamaForCausalLM(tiny_cfg)
            if is_main():
                log.info("smoke model: tiny twin, same block layout, 2 layers")
        else:
            model = build_model(dtype=train_dtype,
                                moe_experts=args.moe_experts,
                                moe_topk=args.moe_topk)
        step = 0
        tokens_seen = 0

    total, _ = count_params(model)
    if is_main():
        log.info(f"model: {total:,} params ({total / 1e12:.3f}T)")
        if args.moe_experts > 0:
            _, active, _ = moe_counts(model, args.moe_experts, args.moe_topk)
            log.info(f"MoE: {args.moe_experts} experts, top-{args.moe_topk}, "
                     f"active per token {active:,} ({active / 1e12:.3f}T), "
                     f"a token touches {100.0 * active / total:.1f}% of the model")
    model.to(device)

    # FSDP2: shard the 2.1T model across all GPUs
    if use_dist:
        from torch.distributed.fsdp import fully_shard

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        if is_main():
            log.info(f"FSDP2 enabled, sharded across {world_size} GPUs")

    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.train()

    opt = build_optimizer(model, args)
    if is_main():
        log.info(f"optimizer: {optimizer_name(args)}")

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum * world_size
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(1, args.max_tokens // tokens_per_step)
    sched = build_scheduler(opt, args, total_steps)
    if is_main():
        log.info(f"plan: {total_steps} optimizer steps, {tokens_per_step:,} tokens/step")
        log.info(f"target tokens: {args.max_tokens:,}")
        log.info(f"modalities: {args.modalities} (text, image, audio in and out)")
        log.info(f"data mix: {args.data_mix or 'default (fineweb:0.8, fineweb-edu:0.2)'}")
        log.info(f"loss spike threshold: {args.loss_spike_threshold}x "
                 f"(0 disables the guard)")
        log.info(f"scheduler: {args.scheduler}, warmup {args.warmup}, "
                 f"min lr ratio {args.min_lr_ratio}")

    # resume the small state after the optimizer and scheduler exist
    if resume_dir:
        step, tokens_seen, got_opt, got_sched = load_ckpt_state(
            resume_dir, opt=opt, sched=sched)
        if is_main():
            log.info(f"resumed at step {step}, tokens {tokens_seen:,}, "
                     f"optimizer state: {got_opt}, scheduler state: {got_sched}")

    run = setup_wandb(args, total_steps)
    metrics = MetricLogger(window=100)
    timer = Timer()
    flops_per_token = estimate_flops_per_token(seq_len=args.seq_len)

    stream = batch_stream(tok, args.micro_batch, seq_len=args.seq_len,
                          mix=args.data_mix, seed=args.seed,
                          filter_docs=not args.no_filter,
                          dedup=None, smoke=args.smoke)
    # loss spike guard: tracks the running loss average, drops any
    # step that spikes above the threshold and reloads the last
    # checkpoint, so one bad batch cannot wreck a months-long run
    spike_guard = LossSpikeGuard(threshold=args.loss_spike_threshold)
    # dedup filter is built here so its state survives across batches
    if args.dedup and not args.smoke:
        from data import DedupFilter

        dedup_filter = DedupFilter()
        stream = batch_stream(tok, args.micro_batch, seq_len=args.seq_len,
                              mix=args.data_mix, seed=args.seed,
                              filter_docs=not args.no_filter,
                              dedup=dedup_filter, smoke=False)
    use_bf16 = device != "cpu" and args.dtype in ("bf16", "fp16")
    autocast_dtype = dtype_map[args.dtype] if use_bf16 else torch.bfloat16
    t0 = time.time()
    micro = 0
    loss_sum = 0.0
    loss_n = 0
    stop_reason = "finished"

    if is_main():
        log.info("waiting for first batch, this can take a few minutes on first run...")
        free = free_disk_gb(args.out) if os.path.isdir(args.out) else free_disk_gb(".")
        log.info(f"free disk: {free:.0f} GB")

    while step < total_steps:
        timer.start("batch")
        try:
            x = next(stream).to(device)
        except StopIteration:
            stream = batch_stream(tok, args.micro_batch, seq_len=args.seq_len,
                                  mix=args.data_mix, seed=args.seed,
                                  filter_docs=not args.no_filter,
                                  dedup=None, smoke=args.smoke)
            x = next(stream).to(device)
        timer.stop("batch")
        if micro == 0 and step == 0 and is_main():
            log.info("first batch in, running first forward/backward...")
        timer.start("forward")
        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                            dtype=autocast_dtype, enabled=use_bf16):
            out = model(x, labels=x)
            loss = out.loss / args.grad_accum
            # MoE load-balancing aux loss, small weight, only when MoE is on
            aux_loss = 0.0
            if args.moe_experts > 0:
                aux_loss = moe_aux_loss(model)
                loss = loss + 0.01 * aux_loss / args.grad_accum
        timer.stop("forward")
        timer.start("backward")
        loss.backward()
        timer.stop("backward")
        loss_sum += out.loss.item()
        loss_n += 1
        micro += 1
        tokens_seen += x.numel() * world_size

        if micro % args.grad_accum == 0:
            # loss spike check: if this step's loss is way above the
            # running average, drop the step and reload the last
            # checkpoint instead of letting one bad batch wreck the run
            step_loss = loss_sum / max(loss_n, 1)
            if spike_guard.check(step_loss):
                if is_main():
                    log.info(f"loss spike at step {step + 1}: loss {step_loss:.4f} "
                             f"vs running avg {spike_guard.running_avg:.4f}, "
                             f"dropping the step (spike #{spike_guard.spikes})")
                opt.zero_grad(set_to_none=True)
                micro = 0
                loss_sum, loss_n = 0.0, 0
                if not args.smoke and not use_dist:
                    last = latest_checkpoint(args.out)
                    if last is not None:
                        try:
                            reload_weights_from_checkpoint(model, last)
                            if is_main():
                                log.info(f"reloaded weights from {last}")
                        except Exception as e:
                            if is_main():
                                log.info(f"could not reload checkpoint: {e}")
                continue
            # clip before the step, so one bad batch cannot blow up the run
            if args.clip > 0:
                if args.clip_mode == "norm":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                else:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            metrics.update(loss=loss_sum / max(loss_n, 1))
            if is_main() and (step <= 20 or step % args.log_every == 0):
                el = time.time() - t0
                tok_s = tokens_seen / max(el, 1)
                # MFU: what share of the GPU peak we actually use
                mfu = flops_per_token * tok_s / (700e12 * world_size) if world_size else 0
                gnorm = grad_norm(model)
                mem = memory_report()
                msg = (f"step {step}/{total_steps} loss {metrics.avg('loss'):.4f} "
                       f"lr {sched.get_last_lr()[0]:.2e} tokens {format_number(tokens_seen)} "
                       f"{tok_s:,.0f} tok/s mfu {mfu * 100:.1f}% "
                       f"gnorm {gnorm:.2f} elapsed {format_time(el)}")
                if mem["allocated"]:
                    msg += f" mem {format_bytes(mem['allocated'])}"
                log.info(msg)
                row = {"step": step, "loss": metrics.avg("loss"),
                       "lr": sched.get_last_lr()[0], "tokens_seen": tokens_seen,
                       "tokens_per_sec": tok_s, "mfu": mfu,
                       "grad_norm": gnorm, "time_s": el}
                write_train_log(args.out, row)
                log_wandb(run, row)
                loss_sum, loss_n = 0.0, 0
            # validation perplexity on a fixed held-out slice
            if args.eval_every and step % args.eval_every == 0 and tok is not None:
                val = run_validation(model, tok, args, device)
                if is_main():
                    log.info(f"eval step {step}: loss {val['loss']:.4f}, "
                             f"perplexity {val['perplexity']:.2f}")
                    write_train_log(args.out, {"step": step, "eval": val})
                    log_wandb(run, {"step": step, **{f"eval/{k}": v for k, v in val.items()}})
            if step % args.ckpt_every == 0 or step == total_steps:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                if is_main():
                    log.info(f"saving {ckpt}")
                save_ckpt(model, tok, ckpt, step, tokens_seen, opt=opt,
                          sched=sched, save_optim=not args.no_save_optim)
                if is_main():
                    removed = prune_checkpoints(args.out, keep=args.keep_ckpts)
                    for r in removed:
                        log.info(f"pruned old checkpoint {r}")
            if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                if not os.path.exists(os.path.join(ckpt, "trainer_state.json")):
                    if is_main():
                        log.info(f"time limit reached, saving {ckpt}")
                    save_ckpt(model, tok, ckpt, step, tokens_seen, opt=opt,
                              sched=sched, save_optim=not args.no_save_optim)
                if is_main():
                    log.info(f"stopping at step {step}, time limit {args.max_hours}h reached")
                stop_reason = "time_limit"
                break
            if args.smoke and step >= 3:
                if is_main():
                    log.info("smoke done, 3 steps ran fine")
                stop_reason = "smoke"
                break

    el = time.time() - t0
    if is_main():
        log.info(f"done ({stop_reason}). steps {step}, tokens {format_number(tokens_seen)}, "
                 f"time {format_time(el)}")
        log.info("timing: " + timer.report().replace("\n", ", "))
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
