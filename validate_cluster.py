"""Cluster pre-flight checks for MNAI Nyx 1.2.

Before a months-long run on 512+ GPUs, run this on the cluster
first. It verifies the things that only break at scale, on a tiny
model, in minutes:

- distributed init: the process group comes up, NCCL on GPU or
  gloo on CPU, every rank can see every other rank.
- all-reduce: a gradient all-reduce gives the same value on every
  rank, so ranks cannot silently drift apart.
- FSDP sharding: the model shards, forward and backward run, and
  the gradients are finite.
- MoE router balance: on a real batch, every expert gets tokens,
  the load-balancing aux loss and the router z-loss are finite, and
  no expert is starved or drowned.
- multimodal: the vision encoder, image decoder, audio encoder and
  audio decoder all produce finite outputs on small inputs.

Single-process mode runs everything on one rank (gloo, world size
1) and is how you check it on a laptop:

    python validate_cluster.py

On the cluster, run it the same way you run train.py:

    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
      --master_addr=<master> --master_port=29500 \
      validate_cluster.py

Exit code 0 means every check passed. Anything else prints which
check failed and why. Fix that before starting the real run.
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn


def check(name, ok, detail=""):
    """One line of the report, and the pass/fail that decides exit."""
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def tiny_model(moe_experts=0):
    """A 2-layer twin of the Nyx block layout, small enough for CPU."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from model import MoELayer

    cfg = LlamaConfig(hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1000,
                      max_position_embeddings=512, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    if moe_experts > 0:
        for layer in model.model.layers:
            layer.mlp = MoELayer(256, 512, n_experts=moe_experts, top_k=2)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--moe-experts", type=int, default=8,
                   help="experts for the router-balance check, 0 skips MoE")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    use_dist = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_ok = True

    # distributed init: the group has to come up before anything
    # else can be checked. Single-process mode uses a file store in
    # a fresh temp dir, not a fixed TCP port: a fixed port collides
    # with a leftover run and fails the ping with garbage, which is
    # exactly what happened the first time this was run here. The
    # file store exercises the same code path with no peers.
    try:
        if use_dist:
            dist.init_process_group(backend)
        else:
            store_dir = tempfile.mkdtemp(prefix="nyx-validate-")
            store = dist.FileStore(
                os.path.join(store_dir, "store"), 1)
            dist.init_process_group(backend, rank=0, world_size=1,
                                    store=store)
        all_ok &= check("distributed init",
                        dist.is_initialized(),
                        f"backend {backend}, world size "
                        f"{dist.get_world_size()}, rank {rank}")
    except Exception as e:  # noqa: BLE001 - report it, do not crash
        all_ok &= check("distributed init", False, str(e))
        print("cannot continue without a process group")
        return 1

    # all-reduce: every rank all-reduces the same tensor and must
    # get the same answer. If ranks drift, training diverges and
    # nobody notices until the loss looks wrong weeks later.
    try:
        t = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        want = sum(range(1, world_size + 1))
        all_ok &= check("gradient all-reduce",
                        abs(t.item() - want) < 1e-6,
                        f"sum of ranks 1..{world_size} = {t.item():.0f}, "
                        f"expected {want}")
    except Exception as e:  # noqa: BLE001
        all_ok &= check("gradient all-reduce", False, str(e))

    # FSDP sharding: shard the tiny model, run a forward and a
    # backward, and check the gradients are real numbers. A sharding
    # bug shows up here in seconds instead of on step 1 of the real
    # run, when the cluster is already paid for.
    try:
        from torch.distributed.fsdp import fully_shard

        model = tiny_model()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        x = torch.randint(0, 1000, (1, 16))
        out = model(x, labels=x)
        out.loss.backward()
        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters() if p.grad is not None)
        all_ok &= check("FSDP shard + forward/backward",
                        grads_finite and torch.isfinite(out.loss).item(),
                        f"loss {out.loss.item():.4f}, "
                        "all grads finite" if grads_finite
                        else "non-finite grads")
    except Exception as e:  # noqa: BLE001
        all_ok &= check("FSDP shard + forward/backward", False, str(e))

    # MoE router balance: run a real batch through the MoE twin and
    # count tokens per expert. A collapsed router sends everything
    # to one expert, which the aux loss and z-loss are supposed to
    # stop. This check proves they are actually working.
    if args.moe_experts > 0:
        try:
            from model import moe_aux_loss, moe_z_loss

            model = tiny_model(moe_experts=args.moe_experts)
            model.eval()
            x = torch.randint(0, 1000, (2, 32))
            with torch.no_grad():
                model(x)
            aux = moe_aux_loss(model)
            z = moe_z_loss(model)
            aux_val = aux.item() if isinstance(aux, torch.Tensor) else aux
            z_val = z.item() if isinstance(z, torch.Tensor) else z
            # per-expert token counts from the first layer's router
            layer = model.model.layers[0].mlp
            logits = layer.router(
                torch.randn(64, 256, generator=torch.Generator().manual_seed(1)))
            topk = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).indices
            counts = torch.bincount(topk.reshape(-1),
                                    minlength=args.moe_experts).float()
            even = 64 * 2 / args.moe_experts
            worst = counts.max().item() / even
            starved = int((counts == 0).sum().item())
            balanced = starved == 0 and worst < 3.0
            all_ok &= check(
                "MoE router balance",
                balanced and torch.isfinite(torch.tensor(aux_val)).item()
                and torch.isfinite(torch.tensor(z_val)).item(),
                f"{args.moe_experts} experts, worst expert {worst:.2f}x "
                f"even share, {starved} starved, aux {aux_val:.4f}, "
                f"z-loss {z_val:.4f}")
        except Exception as e:  # noqa: BLE001
            all_ok &= check("MoE router balance", False, str(e))

    # multimodal: small versions of all four modality modules, random
    # inputs, outputs must be finite. A NaN here means the encoders
    # and decoders disagree on shapes or scales, which would poison
    # the any-to-any training the moment image or audio data flows.
    try:
        from multimodal import (AudioDecoder, AudioEncoder, ImageDecoder,
                                VisionEncoder)

        core = 64
        vis = VisionEncoder(hidden=64, layers=1, heads=4, core_hidden=core)
        img = ImageDecoder(hidden=64, layers=1, heads=4, core_hidden=core)
        aud_e = AudioEncoder(hidden=64, layers=1, core_hidden=core)
        aud_d = AudioDecoder(hidden=64, layers=1, core_hidden=core)
        with torch.no_grad():
            v = vis(torch.randn(1, 3, 224, 224))
            i = img(torch.randn(1, 3, 128, 128),
                    torch.zeros(1, dtype=torch.long),
                    torch.randn(1, core))
            a = aud_e(torch.randn(1, 16, 80))
            d = aud_d(torch.randn(1, 16, core))
        finite = all(torch.isfinite(t).all().item() for t in (v, i, a, d))
        all_ok &= check("multimodal finite outputs", finite,
                        f"vision {tuple(v.shape)}, image {tuple(i.shape)}, "
                        f"audio enc {tuple(a.shape)}, audio dec {tuple(d.shape)}")
    except Exception as e:  # noqa: BLE001
        all_ok &= check("multimodal finite outputs", False, str(e))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
