"""Shared helpers for Nyx 1.2 training, eval and inference.

Nothing here knows about the model shape, it is all plumbing:
logging that looks the same on every rank, seeding so runs are
reproducible, distributed helpers that do the right thing on one
GPU and on a thousand, checkpoint save and load that survives a
crash mid-write, and memory profiling so you can see where the
GPU memory actually went.

Import this from anywhere, it only needs torch and the standard
library, no datasets, no wandb, no bitsandbytes.
"""
import json
import logging
import math
import os
import random
import time

import torch
import torch.distributed as dist


# logging
# -------

def setup_logging(level="INFO", log_file=None):
    """One logger for the whole run, same format everywhere.

    Rank 0 prints to the console, other ranks stay quiet unless you
    pass a log file, in which case each rank writes its own file.
    Call once at the top of main, then use get_logger anywhere.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    rank = dist.get_rank() if dist.is_initialized() else 0
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    if log_file:
        path = log_file if rank == 0 else f"{log_file}.rank{rank}"
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers or [logging.NullHandler()],
        force=True,
    )
    return logging.getLogger("nyx12")


def get_logger(name="nyx12"):
    return logging.getLogger(name)


class MetricLogger:
    """Running averages for loss, tokens per second, and friends.

    Feed it numbers as training goes, ask it for the averages when
    you log. Keeps a short window so the numbers reflect recent
    training, not the first ten steps when everything was chaos.
    """

    def __init__(self, window=100):
        self.window = window
        self.values = {}
        self.counts = {}

    def update(self, **kwargs):
        for key, val in kwargs.items():
            hist = self.values.setdefault(key, [])
            hist.append(float(val))
            if len(hist) > self.window:
                hist.pop(0)

    def avg(self, key):
        hist = self.values.get(key, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def summary(self):
        return {k: self.avg(k) for k in self.values}

    def reset(self):
        self.values.clear()
        self.counts.clear()


class Timer:
    """Wall clock timer with a name, for timing parts of a step."""

    def __init__(self):
        self.starts = {}
        self.totals = {}
        self.counts = {}

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        if name not in self.starts:
            return 0.0
        dt = time.time() - self.starts.pop(name)
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
        return dt

    def avg(self, name):
        if not self.counts.get(name):
            return 0.0
        return self.totals[name] / self.counts[name]

    def report(self):
        lines = []
        for name in sorted(self.totals):
            lines.append(f"{name}: {self.avg(name) * 1000:.1f}ms avg, "
                         f"{self.counts[name]} calls")
        return "\n".join(lines)


# seeding
# -------

def seed_all(seed, rank=0):
    """Seed everything, with a per-rank offset so ranks differ.

    Python random, numpy if it is there, torch CPU and CUDA. The
    rank offset means each data parallel rank shuffles differently
    while the run as a whole is still reproducible from one seed.
    """
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    return seed


def random_state():
    """Snapshot the RNG states, so you can restore them on resume."""
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state):
    """Restore RNG states saved by random_state."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# distributed
# ------------

def is_dist():
    return dist.is_initialized()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    return get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src=0):
    """Send a Python object from src rank to all ranks."""
    if not dist.is_initialized():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def all_reduce_mean(value, device="cuda"):
    """Average a scalar across all ranks, returns a float."""
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / dist.get_world_size())


def all_reduce_sum(value, device="cuda"):
    """Sum a scalar across all ranks, returns a float."""
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def setup_distributed():
    """Init torch distributed from torchrun env vars.

    Returns (device, world_size, rank, local_rank). Safe to call
    when not under torchrun, it just returns a single CPU or CUDA
    device and world size 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, world_size, rank, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def device_of(model):
    """The device the first parameter of a model lives on."""
    return next(model.parameters()).device


# memory
# --------

def memory_report(device=None):
    """Current GPU memory use, in a readable dict.

    Returns zeros on CPU, which is fine, it just means there is no
    GPU to report on. All numbers are bytes, divide by 1e9 for GB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "free": 0}
    if device is None:
        device = torch.cuda.current_device()
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "free": torch.cuda.get_device_properties(device).total_memory
        - torch.cuda.memory_allocated(device),
    }


def format_bytes(n):
    """Turn a byte count into something a human can read."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_memory_report(tag="", device=None):
    """Log the GPU memory use with a label, rank 0 only."""
    if not is_main():
        return
    mem = memory_report(device)
    log = get_logger()
    log.info(f"memory [{tag}] allocated {format_bytes(mem['allocated'])}, "
             f"reserved {format_bytes(mem['reserved'])}, "
             f"max {format_bytes(mem['max_allocated'])}")


def peak_memory_gb(device=None):
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def reset_peak_memory(device=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def estimate_activation_memory(batch, seq_len, hidden, layers, dtype_bytes=2):
    """Very rough activation memory for one micro-batch.

    With gradient checkpointing this is mostly the layer inputs and
    the final logits. Without it, multiply by the layer count. Use
    it to guess a micro-batch size before you OOM, not as gospel.
    """
    per_layer_input = batch * seq_len * hidden * dtype_bytes
    logits = batch * seq_len * 256000 * dtype_bytes
    # checkpointing keeps one input per layer, plus the logits
    return per_layer_input * layers + logits


# checkpoints
# -----------

def atomic_write_json(path, data):
    """Write JSON via a temp file then rename, so a crash cannot
    leave a half-written file behind. The rename is atomic on the
    same filesystem."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def list_checkpoints(out_dir):
    """All ckpt-* dirs in an output dir, sorted by step number."""
    if not os.path.isdir(out_dir):
        return []
    found = []
    for name in os.listdir(out_dir):
        if not name.startswith("ckpt-"):
            continue
        try:
            step = int(name.split("-")[1])
        except (IndexError, ValueError):
            continue
        found.append((step, os.path.join(out_dir, name)))
    return [path for _, path in sorted(found)]


def latest_checkpoint(out_dir):
    """The newest checkpoint dir, or None if there is not one."""
    ckpts = list_checkpoints(out_dir)
    return ckpts[-1] if ckpts else None


def prune_checkpoints(out_dir, keep=3):
    """Delete old checkpoints, keep the newest `keep` ones.

    Checkpoints are huge at 2.1T, disk fills fast. Call this after
    each save. Rank 0 only, it touches the filesystem.
    """
    ckpts = list_checkpoints(out_dir)
    if len(ckpts) <= keep:
        return []
    removed = []
    for path in ckpts[:-keep]:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def save_trainer_state(out_dir, state):
    """Save the small trainer state file, atomically."""
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_json(os.path.join(out_dir, "trainer_state.json"), state)


def load_trainer_state(ckpt_dir):
    """Read the trainer state file from a checkpoint dir."""
    return read_json(os.path.join(ckpt_dir, "trainer_state.json"), default={})


def write_train_log(out_dir, row):
    """Append one JSON line to the training log."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_train_log(out_dir):
    """Read the training log back as a list of dicts."""
    path = os.path.join(out_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def free_disk_gb(path):
    """Free disk space at a path, in GB."""
    import shutil

    _, _, free = shutil.disk_usage(path)
    return free / 1e9


def check_disk_space(out_dir, need_gb=100):
    """Warn if the disk is too full to save a checkpoint."""
    free = free_disk_gb(out_dir)
    if free < need_gb:
        get_logger().warning(
            f"low disk space at {out_dir}: {free:.1f} GB free, "
            f"want at least {need_gb} GB for a checkpoint")
        return False
    return True


# numbers
# -------

def format_number(n):
    """1234567890 becomes 1.23B, for logs."""
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def format_time(seconds):
    """Seconds to a short human string like 3h 12m."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours < 48:
        return f"{hours}h {minutes}m"
    return f"{hours // 24}d {hours % 24}h"


def tokens_to_flops(tokens, n_params=496_369_029_120):
    """Training FLOPs for a token count, the 6*N rule."""
    return 6 * n_params * tokens


def eta_string(tokens_done, tokens_total, tokens_per_sec):
    """How long until the run finishes, as a readable string."""
    if tokens_per_sec <= 0:
        return "unknown"
    left = max(0, tokens_total - tokens_done)
    return format_time(left / tokens_per_sec)


def grad_norm(model):
    """Total gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item()) ** 2
    return math.sqrt(total)


def param_norm(model):
    """Total parameter norm, useful to watch for drift."""
    total = 0.0
    for p in model.parameters():
        total += float(p.data.norm(2).item()) ** 2
    return math.sqrt(total)
