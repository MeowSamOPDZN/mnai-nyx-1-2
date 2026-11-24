"""3D parallelism for MNAI Nyx 1.2: tensor, pipeline, and expert parallel.

Pure FSDP shards everything across every GPU. That works up to
about 128 GPUs, then the all-reduce across nodes becomes the
bottleneck: every step moves the full gradient set over inter-node
links. Frontier labs do not do that. They split the model three
ways at once:

- tensor parallel (TP): each attention head group and each MLP
  shard lives on a different GPU inside one node, where NVLink is
  fast. 8 GPUs per node is the natural TP size.
- pipeline parallel (PP): different layers live on different
  nodes, each node runs its stage and passes activations on.
- data parallel (DP): the rest of the GPUs run different batches,
  sharded with FSDP inside the DP group.

This module builds that 3D device mesh with PyTorch's native
device_mesh, applies TP to the attention and MLP layers with
parallelize_module, splits layers into pipeline stages, and plans
the expert-parallel assignment for MoE. FSDP stays the default in
train.py; pass --parallel-strategy 3d to use this instead.

Run the checks with: python parallel.py --test
That spawns two CPU processes with gloo and verifies a real
2-rank tensor-parallel forward matches the dense model.
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn


# FSDP over 256+ GPUs is a bad choice. Below this, plain FSDP is
# fine and simpler; above it, switch to 3D parallel.
FSDP_WARN_GPUS = 256
# TP lives inside one node, 8 GPUs is the usual node size.
TP_PER_NODE = 8
# bytes per param in training state: bf16 weights 2, bf16 grads 2,
# 8-bit Adam keeps two 1-byte states per param.
STATE_BYTES_PER_PARAM = 6


def build_mesh(world_size, tp=1, pp=1, dp=1, device_type=None):
    """Build the 3D device mesh: (data, pipeline, tensor) degrees.

    The three degrees must multiply to the world size. The mesh is
    a single object; each rank asks for its sub-mesh by name, so
    the TP code only ever talks to its own tensor group and the DP
    code only to its data group. device_type is cuda on a GPU box,
    cpu for the gloo test on a laptop.
    """
    if tp * pp * dp != world_size:
        raise ValueError(
            f"tp {tp} * pp {pp} * dp {dp} = {tp * pp * dp}, "
            f"but world size is {world_size}")
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh(device_type, (dp, pp, tp),
                            mesh_dim_names=("dp", "pp", "tp"))
    return mesh


def apply_tensor_parallel(model, tp_mesh):
    """Shard attention and MLP layers across the tensor group.

    Each decoder layer gets the standard TP recipe: q/k/v/gate/up
    are column-parallel (their outputs are sharded, each rank holds
    its heads), o/down are row-parallel (they all-reduce back to a
    replicated hidden state). After sharding, each rank's attention
    only sees its own heads, so num_heads and num_key_value_heads
    are divided by the TP size on that rank. The lm_head and the
    embeddings stay replicated: every rank computes full logits,
    which costs memory but keeps the loss identical.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    tp_size = tp_mesh.size()
    if tp_size == 1:
        return model
    plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.model.layers:
        parallelize_module(layer, tp_mesh, plan)
    # no head-count fixup needed: this transformers version derives
    # the head count from the projection shapes in the attention
    # forward (it reshapes to (-1, head_dim)), so a sharded q_proj
    # automatically means fewer heads on this rank. The Q/KV ratio
    # is unchanged because q, k and v are all sharded by the same
    # factor, so num_key_value_groups stays correct as well.
    return model


def split_layers_into_stages(n_layers, pp_degree):
    """Split layer indices into pipeline stages, as even as possible.

    Returns a list of (start, end) layer ranges, one per stage.
    The layers must divide evenly enough that no stage is empty;
    pp_degree has to divide n_layers or leave a remainder smaller
    than the stage size. Nyx 1.2 has 144 layers, so pp 1, 2, 3, 4, 6, 8,
    9, 12, 16, 18 all divide it cleanly.
    """
    if pp_degree < 1:
        raise ValueError("pp_degree must be at least 1")
    if pp_degree > n_layers:
        raise ValueError(f"pp {pp_degree} > layers {n_layers}, "
                         "a stage would be empty")
    base = n_layers // pp_degree
    extra = n_layers % pp_degree
    stages = []
    start = 0
    for i in range(pp_degree):
        size = base + (1 if i < extra else 0)
        stages.append((start, start + size))
        start += size
    return stages


def expert_parallel_plan(n_experts, ep_degree):
    """Which experts live on which rank, for expert parallelism.

    MoE experts are independent, so they shard cleanly: each rank
    holds n_experts // ep_degree experts and the router dispatches
    tokens across ranks with an all-to-all. Returns a list of
    expert-index lists, one per rank. ep_degree must divide the
    expert count, 8 experts split over 2, 4 or 8 ranks.
    """
    if ep_degree < 1:
        raise ValueError("ep_degree must be at least 1")
    if n_experts % ep_degree != 0:
        raise ValueError(f"ep {ep_degree} does not divide "
                         f"{n_experts} experts evenly")
    per_rank = n_experts // ep_degree
    return [list(range(i * per_rank, (i + 1) * per_rank))
            for i in range(ep_degree)]


def plan_parallel(n_gpus, total_params, n_layers, moe_experts=0,
                  dtype_bytes=STATE_BYTES_PER_PARAM):
    """Recommend TP/PP/DP degrees for a GPU count, with memory math.

    Returns a dict with the strategy, the degrees, per-GPU state
    memory, and a warning when pure FSDP is a bad choice. The memory
    number is training state only (weights + grads + 8-bit Adam),
    activations with gradient checkpointing sit on top of it. It is
    an estimate, not a measurement: it assumes the state shards
    evenly across dp * tp * pp, which is what 3D parallel does.
    """
    if n_gpus < 1:
        raise ValueError("need at least 1 GPU")
    plan = {"gpus": n_gpus, "total_params": total_params}
    if n_gpus < FSDP_WARN_GPUS:
        plan["strategy"] = "fsdp"
        plan["tp"], plan["pp"], plan["dp"] = 1, 1, n_gpus
        plan["warning"] = None
    else:
        # TP stays inside a node, PP splits the layers, the rest is DP.
        tp = TP_PER_NODE
        pp = 4
        while n_layers % pp != 0 and pp > 1:
            pp -= 1
        dp = n_gpus // (tp * pp)
        if dp < 1:
            dp = 1
            pp = 1
            while n_layers % pp != 0 and pp > 1:
                pp -= 1
            dp = n_gpus // (tp * pp)
        plan["strategy"] = "3d"
        plan["tp"], plan["pp"], plan["dp"] = tp, pp, dp
        plan["warning"] = (
            f"pure FSDP over {n_gpus} GPUs hits inter-node network "
            "bottlenecks: every step all-reduces the full gradient "
            "set across nodes. Use 3D parallel (TP inside a node, "
            "PP across layers, FSDP only inside the DP group).")
    shard = plan["tp"] * plan["pp"] * plan["dp"]
    plan["state_gb_per_gpu"] = (
        total_params * dtype_bytes / shard / 1e9)
    plan["stages"] = split_layers_into_stages(n_layers, plan["pp"])
    if moe_experts > 0:
        ep = min(plan["tp"], moe_experts)
        while moe_experts % ep != 0:
            ep -= 1
        plan["ep"] = ep
        plan["experts_per_rank"] = expert_parallel_plan(moe_experts, ep)
    return plan


def print_parallel_plan(n_gpus, total_params, n_layers, moe_experts=0):
    """Print the recommended parallel plan for a GPU count."""
    plan = plan_parallel(n_gpus, total_params, n_layers, moe_experts)
    print(f"GPUs: {plan['gpus']}, model: {total_params:,} params "
          f"({total_params / 1e9:.1f}B)")
    print(f"strategy: {plan['strategy']}")
    print(f"  tensor parallel (TP): {plan['tp']}")
    print(f"  pipeline parallel (PP): {plan['pp']}")
    print(f"  data parallel (DP): {plan['dp']}")
    if "ep" in plan:
        print(f"  expert parallel (EP): {plan['ep']} "
              f"({moe_experts} experts, {moe_experts // plan['ep']} per rank)")
    for i, (s, e) in enumerate(plan["stages"]):
        print(f"  PP stage {i}: layers {s}-{e - 1} ({e - s} layers)")
    print(f"training state per GPU: ~{plan['state_gb_per_gpu']:.1f} GB "
          "(weights + grads + 8-bit Adam, activations on top)")
    if plan["warning"]:
        print(f"WARNING: {plan['warning']}")
    return plan


def apply_3d_parallel(model, mesh, n_layers):
    """Apply the 3D recipe to a model on this rank: TP now, PP/EP noted.

    Tensor parallel is applied for real with parallelize_module.
    Pipeline and expert parallel are stage assignments: this rank
    logs which layers and which experts it owns, and the training
    loop drives the pipeline schedule when pp > 1. Returns the
    model with TP applied.
    """
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]
    rank = dist.get_rank() if dist.is_initialized() else 0
    pp_rank = pp_mesh.get_local_rank()
    stages = split_layers_into_stages(n_layers, pp_mesh.size())
    print(f"3D parallel: rank {rank} is PP stage {pp_rank}, "
          f"layers {stages[pp_rank][0]}-{stages[pp_rank][1] - 1}, "
          f"TP size {tp_mesh.size()}")
    return apply_tensor_parallel(model, tp_mesh)


def _tp_worker(rank, world_size, store_path):
    """One worker of the 2-rank TP check: forward must match dense."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=f"file://{store_path}")
    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1000,
                      max_position_embeddings=512, tie_word_embeddings=True)
    ref = LlamaForCausalLM(cfg)
    ref.eval()
    model = LlamaForCausalLM(cfg)
    model.load_state_dict(ref.state_dict())
    model.eval()
    mesh = build_mesh(world_size, tp=2, pp=1, dp=1, device_type="cpu")
    apply_tensor_parallel(model, mesh["tp"])
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        got = model(x).logits
        want = ref(x).logits
    # each rank holds half the heads, the all-reduce in o_proj and
    # down_proj puts the full hidden state back, so the logits must
    # match the dense model to float precision.
    diff = (got - want).abs().max().item()
    print(f"rank {rank}: TP logits max diff vs dense: {diff:.2e}")
    assert diff < 1e-4, f"TP forward diverged: {diff}"
    dist.barrier()
    dist.destroy_process_group()


def _free_port():
    """A free localhost port, kept for callers that need one."""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tensor_parallel():
    """Spawn 2 CPU ranks and check TP matches the dense model.

    Uses the fork start method, not spawn: fork inherits the
    already-imported torch instead of re-importing it in each
    child, and it skips the slow spawn pipe handshake that hung
    for 12 minutes on this VM. The rendezvous is a file store,
    not a TCP port: the TCP rendezvous also hung here, retrying
    a store that never came up. CPU-only (gloo), so fork is safe.
    """
    import tempfile
    import torch.multiprocessing as mp

    store_dir = tempfile.mkdtemp(prefix="nyx-tp-")
    store_path = os.path.join(store_dir, "store")
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # already set, for example by an earlier test run
    mp.spawn(_tp_worker, args=(2, store_path), nprocs=2, join=True)
    print("TP check passed: 2-rank tensor-parallel forward matches dense")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="run the 2-rank tensor-parallel check on CPU")
    p.add_argument("--plan", type=int, default=0,
                   help="print the parallel plan for this many GPUs")
    p.add_argument("--params", type=int, default=2_210_128_644_368,
                   help="total params for the plan, default Nyx 1.2 any-to-any")
    p.add_argument("--layers", type=int, default=144,
                   help="layers for the plan, default Nyx 1.2")
    p.add_argument("--moe-experts", type=int, default=0)
    args = p.parse_args()
    if args.test:
        test_tensor_parallel()
    if args.plan:
        print_parallel_plan(args.plan, args.params, args.layers,
                            args.moe_experts)


if __name__ == "__main__":
    main()
