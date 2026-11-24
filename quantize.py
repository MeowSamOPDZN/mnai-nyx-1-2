"""Turn a trained Nyx 1.2 checkpoint into two quantized GGUFs.

A trained 2.1T checkpoint is about 4.4TB in bf16. Nobody runs that
on a laptop. Quantization shrinks it: Q4_K_M is about 0.55 bytes
per parameter, Q8_0 is about 1 byte per parameter. This script
produces both, so you can pick the small one for a weak machine or
the accurate one for a strong one.

It uses llama.cpp, the same converter Nano 2 uses: convert the HF
checkpoint to F16 GGUF, then quantize that to Q4_K_M and Q8_0 with
llama-quantize. If llama.cpp is not next to the repo it is cloned
from GitHub, and llama-quantize is built if it is not there yet.
That is the only path that produces correct K-quants.

If llama.cpp truly cannot be obtained, the script falls back to a
minimal GGUF writer built on the gguf Python package, which writes
Q8_0 and Q4_K_M directly from the HF weights. The llama.cpp path
is preferred and is the one that is tested.

The Q files are produced from a trained checkpoint once someone
trains the model. The script itself is ready now: point it at any
HF checkpoint, even a tiny smoke checkpoint, and it produces both
files. Expected sizes for the full 2.184T text core:
  Q4_K_M  about 1,201 GB (0.55 bytes per param)
  Q8_0    about 2,184 GB (1 byte per param)
The any-to-any modality modules stay in PyTorch, only the text
core becomes a GGUF, because llama.cpp converts Llama models.

Usage:
  python quantize.py --ckpt out-nyx12/ckpt-5000 --out out-gguf
  python quantize.py --ckpt out-nyx12/ckpt-5000 --out out-gguf --llama-cpp ./llama.cpp
  python quantize.py --ckpt /tmp/smoke-nyx12/ckpt-3 --out /tmp/smoke-gguf --llama-cpp ~/workspace/mnai-nano/llama.cpp
"""
import argparse
import os
import subprocess
import sys

# bytes per parameter for each quant type, for the size estimates
BYTES_PER_PARAM = {"Q4_K_M": 0.55, "Q8_0": 1.0}


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_llama_cpp(explicit=None):
    """Find a llama.cpp checkout, or clone one next to the repo.

    Checks the explicit path first, then ./llama.cpp, then the
    Nano workspace copy this machine usually has. Clones from
    GitHub as a last resort. Returns the path, or None if nothing
    worked, in which case the gguf fallback is used.
    """
    candidates = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    candidates.append(os.path.join(os.getcwd(), "llama.cpp"))
    candidates.append(os.path.expanduser("~/workspace/mnai-nano/llama.cpp"))
    for path in candidates:
        converter = os.path.join(path, "convert_hf_to_gguf.py")
        if os.path.isdir(path) and os.path.exists(converter):
            return path
    print("llama.cpp not found, cloning it...", flush=True)
    try:
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git",
             "llama.cpp"])
        return os.path.join(os.getcwd(), "llama.cpp")
    except Exception as e:
        print(f"could not clone llama.cpp ({e}), using the gguf fallback", flush=True)
        return None


def ensure_gguf_package(llama_cpp):
    """Make sure the gguf Python package is importable.

    The converter needs it. llama.cpp ships a requirements file,
    install it quietly if the import fails.
    """
    try:
        import gguf  # noqa: F401
        return True
    except ImportError:
        req = os.path.join(llama_cpp, "requirements.txt")
        print("installing llama.cpp requirements (gguf package)...", flush=True)
        try:
            run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
            import gguf  # noqa: F401
            return True
        except Exception as e:
            print(f"could not install gguf package: {e}", flush=True)
            return False


def find_quantize_binary(llama_cpp):
    """Find or build the llama-quantize binary."""
    for path in (os.path.join(llama_cpp, "build", "bin", "llama-quantize"),
                 os.path.join(llama_cpp, "llama-quantize")):
        if os.path.exists(path):
            return path
    print("building llama-quantize...", flush=True)
    run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=llama_cpp)
    run(["cmake", "--build", "build", "-j", "--target", "llama-quantize"],
        cwd=llama_cpp)
    return os.path.join(llama_cpp, "build", "bin", "llama-quantize")


def convert_with_llama_cpp(ckpt, out_dir, llama_cpp, quants=("Q4_K_M", "Q8_0")):
    """Full llama.cpp path: HF -> F16 GGUF -> one file per quant."""
    os.makedirs(out_dir, exist_ok=True)
    converter = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    f16_path = os.path.join(out_dir, "model.f16.tmp.gguf")
    run([sys.executable, converter, ckpt, "--outfile", f16_path,
         "--outtype", "f16"])
    quant = find_quantize_binary(llama_cpp)
    made = {}
    for q in quants:
        out_path = os.path.join(out_dir, f"model-{q.lower().replace('_', '-')}.gguf")
        run([quant, f16_path, out_path, q])
        made[q] = out_path
    os.remove(f16_path)
    return made


def quantize_fallback(ckpt, out_dir, quants=("Q4_K_M", "Q8_0")):
    """Minimal GGUF writer fallback, via the gguf Python package.

    Loads the HF weights with safetensors, writes a GGUF per quant
    type with gguf.GGUFWriter and its K-quant routines. Only used
    when llama.cpp cannot be obtained at all. The llama.cpp path
    above is preferred and is the tested one.
    """
    try:
        import gguf
        from safetensors.torch import load_file
    except ImportError:
        print("gguf package not available, installing it...", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "gguf"])
        import gguf
        from safetensors.torch import load_file
    import json as _json

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        cfg = _json.load(f)
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    state = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = _json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(os.path.join(ckpt, shard)))
    else:
        state = load_file(os.path.join(ckpt, "model.safetensors"))
    made = {}
    for q in quants:
        out_path = os.path.join(out_dir, f"model-{q.lower().replace('_', '-')}.gguf")
        writer = gguf.GGUFWriter(out_path, "llama")
        writer.add_architecture()
        writer.add_name("MNAI Nyx 1.2")
        writer.add_context_length(cfg.get("max_position_embeddings", 16384))
        writer.add_embedding_length(cfg["hidden_size"])
        writer.add_block_count(cfg["num_hidden_layers"])
        writer.add_feed_forward_length(cfg["intermediate_size"])
        writer.add_head_count(cfg["num_attention_heads"])
        writer.add_head_count_kv(cfg["num_key_value_heads"])
        writer.add_rope_freq_base(cfg.get("rope_theta", 500000.0))
        writer.add_rms_norm_eps(cfg.get("rms_norm_eps", 1e-5))
        writer.add_vocab_size(cfg["vocab_size"])
        for name, tensor in state.items():
            tensor = tensor.float().numpy()
            if q == "Q8_0":
                data, qtype = gguf.quants.quantize_q8_0(tensor), gguf.GGMLQuantizationType.Q8_0
            else:
                data, qtype = gguf.quants.quantize_q4_k(tensor), gguf.GGMLQuantizationType.Q4_K
            writer.add_tensor(name, data, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        made[q] = out_path
    return made


def estimate_size(n_params, quant):
    """Expected GGUF size in bytes for a model of this many params."""
    return int(n_params * BYTES_PER_PARAM[quant])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="trained HF checkpoint dir (weights + config.json)")
    p.add_argument("--out", required=True,
                   help="output dir for the two GGUF files")
    p.add_argument("--llama-cpp", default=None,
                   help="llama.cpp checkout path, cloned if missing")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail instead of using the gguf fallback writer")
    args = p.parse_args()

    if not os.path.isdir(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    if not os.path.exists(os.path.join(args.ckpt, "config.json")):
        raise FileNotFoundError(f"no config.json in {args.ckpt}, "
                                "is this a HF checkpoint dir?")
    quants = ("Q4_K_M", "Q8_0")
    llama_cpp = find_llama_cpp(args.llama_cpp)
    made = None
    if llama_cpp is not None and ensure_gguf_package(llama_cpp):
        made = convert_with_llama_cpp(args.ckpt, args.out, llama_cpp, quants)
    elif not args.no_fallback:
        print("falling back to the minimal gguf writer", flush=True)
        made = quantize_fallback(args.ckpt, args.out, quants)
    else:
        raise RuntimeError("llama.cpp unavailable and --no-fallback was passed")
    print()
    for q in quants:
        path = made[q]
        size = os.path.getsize(path)
        print(f"{q:8s} {path} ({size / 1e6:.1f} MB)")
    print("done, both quantized models written")


if __name__ == "__main__":
    main()
