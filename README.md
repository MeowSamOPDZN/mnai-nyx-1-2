# MNAI Nyx 1.2

**AIMYTOG: AI Model You'll Train On GPU.**

**ANY-to-ANY: text, image, and audio in, text, image, and audio out. One model reads and writes all three.**

A 2.1T parameter frontier any-to-any model, from scratch. You bring the GPU. The repo is the model, the training code, and the tokenizer. Nothing is pre-trained here. You train it yourself, on your own cluster, with your own data. That is the whole point.

> MNAI Nyx family is a real legit AI, That YOU have to train.

The data is the ENTIRE web. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. FineWeb comes from Common Crawl, which visits every website it can reach, every domain, even the random ones. If a page is on the internet, it is in the crawl. That is the largest open web corpus there is.

## The model

MNAI Nyx 1.2. From scratch, no base model, no distillation.

| | |
|---|---|
| Params | **2,210,128,644,368 (2.210T)** — core text 2.184T + vision 6.29B + image decoder 18.86B + audio 1.20B |
| Architecture | Llama decoder, GQA, SwiGLU, RoPE, RMSNorm + ViT vision encoder + DiT image decoder + audio encoder/decoder |
| Modalities | text, image, audio in and out |
| Hidden | 36864 |
| Layers | 144 |
| Heads | 288 query, 16 key/value |
| FFN | 110592 |
| Vocab | 256000 |
| Context | 16384 |
| Embeddings | tied |

Standard Llama block layout, so llama.cpp can turn a finished checkpoint into a GGUF with its normal converter.

## The data

The entire web. The training stream is the full FineWeb crawl (15T tokens) plus FineWeb-Edu, interleaved 80/20. A 2.1T model wants about 40T tokens, more than the whole internet holds, so it trains on the full 15T, every website, every domain, even the random ones.

## Hardware

A 2.1T model does not fit on one GPU. Training needs a cluster with FSDP.

Per GPU, bf16 weights + 8-bit Adam, FSDP full shard:

| GPUs | States per GPU |
|---|---|
| 512x H100 80GB | ~26GB |
| 1024x H100 80GB | ~13GB |

512x H100 is the minimum, 1024x is ideal. Do not try this on fewer than 512 GPUs unless you know what you are doing. Activations with gradient checkpointing need the rest of the memory.

A full run is the entire internet, 15T tokens. At 1024x H100, expect roughly 1-2 years. This is a frontier run, not a weekend project.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train the tokenizer on your own text
python train_tokenizer.py --data ./data/*.txt --out ./tokenizer

# 2. Train, torchrun, one process per GPU
torchrun --nproc_per_node=8 --nnodes=128 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-nyx12 \
  --max-tokens 15000000000000 --micro-batch 1 --grad-accum 4096

# 3. Resume after a stop
torchrun --nproc_per_node=8 --nnodes=128 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-nyx12 \
  --resume out-nyx12/ckpt-500 --max-tokens 15000000000000

# 4. Generate (needs a sharded inference engine for the full 2.1T,
#    vLLM or TensorRT-LLM, this script is for a small test checkpoint)
python generate.py --model out-nyx12/ckpt-1000 --prompt "The future of AI is"
```

## Training options

```bash
# WSD schedule with WandB logging and validation perplexity
torchrun --nproc_per_node=8 --nnodes=128 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-nyx12 \
  --scheduler wsd --decay-frac 0.1 --wandb --wandb-project nyx12 \
  --eval-every 1000 --eval-batches 20

# Adafactor optimizer, fp16, custom data mix
torchrun --nproc_per_node=8 --nnodes=128 --node_rank=0 \
  --master_addr=<master> --master_port=29500 \
  train.py --tokenizer ./tokenizer --out out-nyx12 \
  --optim adafactor --dtype fp16 --data-mix "fineweb:0.7,fineweb-edu:0.3"
```

Schedulers: `cosine` (classic), `linear`, `wsd` (warmup-stable-decay,
holds steady then decays at the end, good when you might stop early).
Optimizers: `bnb8bit` (default), `adamw`, `adafactor`, `sgd`.

## Evaluate a checkpoint

```bash
# perplexity, HellaSwag, ARC, Winogrande, and sample generations
python evaluate.py --model out-nyx12/ckpt-1000 --limit 200

# perplexity on your own text file
python evaluate.py --model out-nyx12/ckpt-1000 --val-file ./val.txt
```

## Smoke test

No GPU, no data download. Uses a tiny twin of the same block layout.

```bash
python train.py --smoke --max-steps 3 --out /tmp/smoke-nyx12
python model.py
```

## Make your OWN named LLM

This repo is a template. Rename the model to yours in 2 minutes, 4 small changes. Example: rename it to **MyCoolModel**.

**Step 1 — model.py, line 95.** Open `model.py`, find this line:
```python
MODEL_NAME = "MNAI Nyx 1.2"
```
Change it to:
```python
MODEL_NAME = "MyCoolModel"
```
That is the only code change. The name shows up in the model description and logs from here on.

**Step 2 — README.md, line 1.** Open `README.md`, find this line:
```markdown
# MNAI Nyx 1.2
```
Change it to:
```markdown
# MyCoolModel
```

**Step 3 — train.py, line 129.** Open `train.py`, find this line:
```python
p.add_argument("--out", default="out-nyx12")
```
Change it to:
```python
p.add_argument("--out", default="out-mycoolmodel")
```
That is just the default output folder name for your checkpoints. You can also leave it and pass `--out` yourself every run.

**Step 4 — config.json.** Nothing to change. There is no name field in `config.json`, the shape is the same whatever you call the model.

That is it. Train it, and the model that comes out is MyCoolModel, trained by you, on your data, on your GPU.

## Files

| File | What it is |
|---|---|
| `model.py` | The 2.1T architecture (MNAI Nyx 1.2), param count, FLOPs, init, surgery helpers |
| `multimodal.py` | Any-to-any modules: ViT vision encoder, DiT image decoder, audio encoder/decoder |
| `train.py` | FSDP2 distributed training, schedulers, optimizers, WandB, eval, full checkpoints |
| `data.py` | Data loading, filtering, dedup, tokenization, dataset mixing |
| `evaluate.py` | Perplexity, HellaSwag / ARC / Winogrande benchmarks, generation quality |
| `utils.py` | Logging, seeding, distributed helpers, checkpoints, memory profiling |
| `inference.py` | Generation with KV cache, batched generation, sampling, chat |
| `train_tokenizer.py` | Byte-level BPE tokenizer, 256000 tokens |
| `generate.py` | Text generation from a checkpoint, CLI wrapper around inference.py |
| `config.json` | Model config, HF format |
| `requirements.txt` | Dependencies |

## License

MIT. Train it, use it, sell it. It is yours once you train it.

## FAQ

**Is MNAI Nyx a real AI?**
MNAI Nyx family is a real legit AI, That YOU have to train.

**What is MNAI Nyx 1.2?**
MNAI Nyx 1.2 is a 2.1T parameter frontier large language model, built from scratch by MeowSamOPDZN Studio. It is the second model in the Nyx family, after Nyx 1 (500B).

**What is AIMYTOG?**
AIMYTOG stands for AI Model You'll Train On GPU. It is a new technology from MeowSamOPDZN Studio: instead of downloading a pre-trained model, you train the model yourself on your own GPU cluster. The repo gives you the model, the training code, and the tokenizer. You bring the GPU.

**Can I train MNAI Nyx 1.2 myself?**
Yes. That is the whole point. The repo is public. You need a GPU cluster (512x H100 minimum, 1024x recommended), the training code uses FSDP to shard the 2.1T model across all your GPUs.

**What data does MNAI Nyx 1.2 train on?**
The entire web. The full FineWeb crawl, 15 trillion tokens of open web text, plus FineWeb-Edu. FineWeb comes from Common Crawl, which visits every website it can reach, every domain, even the random ones.

**Is MNAI Nyx 1.2 open source?**
Yes. The code is MIT licensed and public on GitHub. The model is yours once you train it.

**How big is MNAI Nyx 1.2?**
2.210 trillion parameters. Core text is 2.184T (36864 hidden, 144 layers, 288 attention heads, 256000 token vocabulary, 16384 context), plus the any-to-any modules: vision encoder 6.29B, image decoder 18.86B, audio encoder and decoder 1.20B.

**Is MNAI Nyx 1.2 any-to-any?**
Yes. Text, image, and audio in, text, image, and audio out. One model reads and writes all three. The vision encoder, DiT image decoder, and audio encoder/decoder live in multimodal.py.

## Credits

Built by MeowSamOPDZN Studio. The 2.1T shape is ours, the Llama block layout is standard so tooling just works.
