"""Train the Nyx 1.2 BPE tokenizer. 152064 tokens, byte-level BPE.

Usage:
  python train_tokenizer.py --data ./data/train.txt --out ./tokenizer
  python train_tokenizer.py --data ./data/*.txt --vocab-size 256000
  python train_tokenizer.py --data ./data/*.txt --out ./tokenizer --show

The tokenizer is trained on your own text. Give it a large, clean
corpus and it will learn the merges. A good tokenizer makes training
faster and the model smarter, because fewer tokens means more text
per context window.

Byte-level BPE means it can tokenize anything, even text it has
never seen, because every byte is in the vocabulary. No unknown
tokens in practice, which is what you want when the model trains
on all of the internet.

What the options do
-------------------
--min-frequency sets how rare a merge has to be before it is kept.
2 is the default, higher values give a cleaner vocab with fewer
junk merges, lower values keep more rare words whole.

--show prints sample tokenizations after training, so you can see
with your own eyes whether common words are single tokens and
whether the tokenizer is wasting tokens on spaces.

--test takes a few strings and tokenizes them with an existing
tokenizer, without training. Handy when you want to check what a
tokenizer you already have does with a sentence.
"""
import argparse
import glob
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 256000
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

SAMPLE_TEXTS = [
    "The future of AI is",
    "Hello, how are you today?",
    "def train(model, data):",
    "The quick brown fox jumps over the lazy dog.",
    "MNAI Nyx 1.2 is a 500B parameter language model.",
]


def collect_files(patterns):
    """Expand globs into a sorted, deduplicated file list."""
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return files


def train_tokenizer(files, out_dir, vocab_size=VOCAB_SIZE, min_frequency=2,
                    show_progress=True):
    """Train a byte-level BPE tokenizer on the given files."""
    print(f"training on {len(files)} files", flush=True)
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    tok.train(files, trainer)
    os.makedirs(out_dir, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast.save_pretrained(out_dir)
    print(f"saved tokenizer to {out_dir}", flush=True)
    print(f"vocab size: {len(fast)}", flush=True)
    return fast


def vocab_stats(tokenizer):
    """Basic stats about a trained tokenizer."""
    vocab = tokenizer.get_vocab()
    # how many tokens are plain ASCII words vs subwords vs bytes
    ascii_tokens = sum(1 for t in vocab if all(ord(c) < 128 for c in t))
    return {
        "vocab_size": len(vocab),
        "ascii_tokens": ascii_tokens,
        "bos": tokenizer.bos_token,
        "eos": tokenizer.eos_token,
        "pad": tokenizer.pad_token,
        "unk": tokenizer.unk_token,
    }


def show_samples(tokenizer, texts=None):
    """Print tokenizations for a few sample sentences."""
    texts = texts or SAMPLE_TEXTS
    print("sample tokenizations:")
    for text in texts:
        ids = tokenizer.encode(text)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {text!r}")
        print(f"    {len(ids)} tokens: {pieces}")


def tokens_per_char(tokenizer, texts):
    """Average characters per token, a rough efficiency number.

    English is about 4 chars per token for a good BPE. Much lower
    means the tokenizer is splitting words too much, much higher
    usually means it memorized whole phrases.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def check_coverage(tokenizer, files, limit=100000):
    """What share of the training text is covered by whole words.

    Reads the first files up to a char limit, tokenizes, and reports
    the average token length. Long tokens mean the vocab fits the
    data well. Short tokens mean the data has a lot the tokenizer
    has not seen.
    """
    chars = 0
    texts = []
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            chunk = f.read(20000)
        texts.append(chunk)
        chars += len(chunk)
        if chars >= limit:
            break
    avg_len = tokens_per_char(tokenizer, texts)
    print(f"coverage check on {len(texts)} files, {chars:,} chars")
    print(f"average chars per token: {avg_len:.2f}")
    return avg_len


def test_tokenizer(path, texts):
    """Load an existing tokenizer and show what it does with texts."""
    tok = PreTrainedTokenizerFast.from_pretrained(path)
    print(f"loaded {path}, vocab {len(tok)}")
    show_samples(tok, texts)
    print(f"chars per token: {tokens_per_char(tok, texts):.2f}")
    return tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=None,
                   help="text files to train on, globs allowed")
    p.add_argument("--out", default="./tokenizer")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--min-frequency", type=int, default=2,
                   help="a merge must appear this many times to be kept")
    p.add_argument("--show", action="store_true",
                   help="print sample tokenizations after training")
    p.add_argument("--coverage", action="store_true",
                   help="check how well the vocab fits the training text")
    p.add_argument("--stats", default=None,
                   help="write vocab stats as JSON to this file")
    p.add_argument("--test", default=None,
                   help="load a tokenizer from this path and test it, no training")
    p.add_argument("--test-texts", nargs="*", default=None,
                   help="texts to tokenize with --test")
    args = p.parse_args()

    if args.test:
        test_tokenizer(args.test, args.test_texts or SAMPLE_TEXTS)
        return

    if not args.data:
        raise SystemExit("--data is required unless you pass --test")
    files = collect_files(args.data)
    if not files:
        raise SystemExit("no input files found")
    tok = train_tokenizer(files, args.out, vocab_size=args.vocab_size,
                          min_frequency=args.min_frequency)
    stats = vocab_stats(tok)
    print(f"vocab stats: {json.dumps(stats)}")
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"wrote {args.stats}")
    if args.show:
        show_samples(tok)
    if args.coverage:
        check_coverage(tok, files)


if __name__ == "__main__":
    main()
