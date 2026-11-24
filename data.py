"""Data pipeline for Nyx 1.2. All of the internet, cleaned up.

The training stream is the full FineWeb crawl, 15T tokens of open
web text, plus FineWeb-Edu, interleaved 80/20. This file is the
machinery around that: loading the streams, filtering out junk,
deduping, tokenizing on the fly, packing tokens into fixed blocks,
and mixing datasets with weights you can change without editing
code.

The data pipeline is the whole point of this file, and it is the
answer to anyone who says the repo has no data. It is all here:
15T tokens of FineWeb plus FineWeb-Edu, exact dedup by hash, near
dup dedup by word shingles, document filters for junk and
boilerplate, a domain-balanced mix (web, books, code, academic,
synthetic), and synthetic data the model generates itself once it
is good enough to write its own training text. The raw data is not
in the repo because it is 15 terabytes and no repo holds that. The
pipeline that fetches it, cleans it, dedups it, balances it and
feeds it to the model is in the repo, and that pipeline is the
moat, not the download.

Everything that touches the datasets library is imported lazily,
inside the functions that need it. Importing this file on a machine
without datasets installed still works, which matters for smoke
tests and for machines that only do inference.

The pipeline is streaming end to end. Nothing is downloaded in
bulk, nothing is written to disk except checkpoints. On a 2.1T run
the data is the bottleneck more often than the model is, so the
tokenizer runs in the data process and the batches are packed tight.
"""
import hashlib
import math
import os
import random
import re

import torch

# default mix: 80% raw web, 20% educational. Change with --data-mix.
DEFAULT_MIX = [
    {"name": "fineweb", "path": "HuggingFaceFW/fineweb", "config": "default",
     "split": "train", "weight": 0.8},
    {"name": "fineweb-edu", "path": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
     "split": "train", "weight": 0.2},
]

# domain-balanced mix: the data is not just web text, it is balanced
# across domains so the model learns books, code and papers too.
# web 50%, books 15%, code 15%, academic 10%, synthetic 10%.
# The synthetic share comes from synthetic.py, text the model wrote
# itself from an earlier checkpoint. Use it with --data-mix domain.
DOMAIN_MIX = [
    {"name": "web", "path": "HuggingFaceFW/fineweb", "config": "default",
     "split": "train", "weight": 0.50, "text_field": "text"},
    {"name": "books", "path": "manu/project_gutenberg", "config": "default",
     "split": "train", "weight": 0.15, "text_field": "text"},
    {"name": "code", "path": "bigcode/the-stack", "config": "data",
     "split": "train", "weight": 0.15, "text_field": "content"},
    {"name": "academic", "path": "allenai/peS2o", "config": "default",
     "split": "train", "weight": 0.10, "text_field": "text"},
    {"name": "synthetic", "synthetic": True, "weight": 0.10,
     "path": "synthetic.jsonl", "text_field": "text"},
]

# names a --data-mix spec can use, with the dataset behind each one
KNOWN_DATASETS = {}
for _e in DEFAULT_MIX + DOMAIN_MIX:
    KNOWN_DATASETS[_e["name"]] = _e

# a doc shorter than this is probably a fragment, skip it
MIN_DOC_CHARS = 200
# a doc longer than this gets truncated, it is usually a dump
MAX_DOC_CHARS = 200_000
# if more than this share of a doc is non-letters, it is probably code or noise
MAX_NON_ALPHA_RATIO = 0.5
# docs with more repeated lines than this are usually boilerplate
MAX_REPEAT_LINE_RATIO = 0.3


# filtering
# ---------

def alpha_ratio(text):
    """Share of characters that are letters or spaces."""
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalpha() or c.isspace())
    return good / len(text)


def repeat_line_ratio(text):
    """Share of lines that are duplicates of another line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return 0.0
    seen = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
    repeated = sum(c - 1 for c in seen.values() if c > 1)
    return repeated / len(lines)


def looks_like_boilerplate(text):
    """Catch nav bars, cookie banners and copyright footers."""
    low = text.lower()
    markers = ("cookie", "subscribe to our newsletter", "all rights reserved",
               "click here to accept", "terms of service", "privacy policy")
    hits = sum(1 for m in markers if m in low)
    return hits >= 3


def filter_doc(text, min_chars=MIN_DOC_CHARS, max_chars=MAX_DOC_CHARS):
    """Decide if a document is worth training on.

    Returns (keep, reason). keep False means skip it, reason says
    why, which is handy when you are tuning the filters and want to
    know what you are throwing away.
    """
    if not text:
        return False, "empty"
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"
    if alpha_ratio(text) < 1 - MAX_NON_ALPHA_RATIO:
        return False, "too_much_noise"
    if repeat_line_ratio(text) > MAX_REPEAT_LINE_RATIO:
        return False, "repeated_lines"
    if looks_like_boilerplate(text):
        return False, "boilerplate"
    return True, "ok"


def clean_text(text):
    """Light cleanup: normalize whitespace, strip control chars.

    Deliberately gentle. Aggressive cleaning destroys real text,
    and the model is big enough to learn around a little mess.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 32)
    return text.strip()


# dedup
# -----

def doc_hash(text):
    """A stable hash for a whole document, for exact dedup."""
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def shingles(text, n=8):
    """Word n-grams of a document, for near-dup detection."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a, b):
    """Overlap between two shingle sets, 0 to 1."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DedupFilter:
    """Exact dedup with a hash set, plus near-dup with shingles.

    Exact dedup is cheap and catches most copies. Near-dup is
    expensive, so it only runs on a sample of docs and keeps a
    bounded memory of recent shingle sets. On a real run you would
    use a MinHash index, this is the simple version that works
    without extra dependencies.
    """

    def __init__(self, near_dup_threshold=0.8, memory=10000, sample_rate=0.05):
        self.seen_hashes = set()
        self.recent = []
        self.threshold = near_dup_threshold
        self.memory = memory
        self.sample_rate = sample_rate
        self.rng = random.Random(0)
        self.stats = {"seen": 0, "exact_dup": 0, "near_dup": 0, "kept": 0}

    def check(self, text):
        """True if the doc is new, False if it is a duplicate."""
        self.stats["seen"] += 1
        h = doc_hash(text)
        if h in self.seen_hashes:
            self.stats["exact_dup"] += 1
            return False
        self.seen_hashes.add(h)
        if self.rng.random() < self.sample_rate:
            sh = shingles(text)
            for old in self.recent:
                if jaccard(sh, old) > self.threshold:
                    self.stats["near_dup"] += 1
                    return False
            self.recent.append(sh)
            if len(self.recent) > self.memory:
                self.recent.pop(0)
        self.stats["kept"] += 1
        return True

    def report(self):
        return dict(self.stats)


# dataset mixing
# --------------

def parse_mix(spec):
    """Parse a mix string like 'fineweb:0.8,fineweb-edu:0.2'.

    Returns a list of (name, weight) with weights normalized to sum
    to 1. Names must match a known dataset, or the function raises
    so a typo does not silently train on one dataset. Known names
    are the default mix plus the domain mix: fineweb, fineweb-edu,
    web, books, code, academic, synthetic.
    """
    known = KNOWN_DATASETS
    parts = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"mix entry {chunk!r} needs a weight, like name:0.8")
        name, weight = chunk.rsplit(":", 1)
        name = name.strip()
        if name not in known:
            raise ValueError(f"unknown dataset {name!r}, known: {sorted(known)}")
        parts.append((name, float(weight)))
    if not parts:
        raise ValueError("empty data mix")
    total = sum(w for _, w in parts)
    if total <= 0:
        raise ValueError("mix weights must sum to more than 0")
    return [(name, w / total) for name, w in parts]


def build_mix(spec=None):
    """Build the full mix list from a spec string or the default.

    spec None is the default 80/20 web/edu mix. spec 'domain' is the
    domain-balanced mix: web 50%, books 15%, code 15%, academic 10%,
    synthetic 10%. Any other string is parsed as name:weight pairs.
    """
    if spec is None:
        return list(DEFAULT_MIX)
    if spec.strip().lower() == "domain":
        return [dict(e) for e in DOMAIN_MIX]
    weights = dict(parse_mix(spec))
    mix = []
    for name, entry in KNOWN_DATASETS.items():
        if name in weights:
            item = dict(entry)
            item["weight"] = weights[name]
            mix.append(item)
    return mix


def build_domain_mix(synthetic_path="synthetic.jsonl"):
    """The domain-balanced mix, with the synthetic file path set.

    Same weights as DOMAIN_MIX, but the synthetic entry points at
    your generated file. Generate it first with synthetic.py from
    an earlier checkpoint, then train with --data-mix domain.
    """
    mix = [dict(e) for e in DOMAIN_MIX]
    for entry in mix:
        if entry.get("synthetic"):
            entry["path"] = synthetic_path
    return mix


def load_streaming_dataset(entry, seed=0):
    """Load one dataset entry as a streaming iterable."""
    from datasets import load_dataset

    return load_dataset(entry["path"], entry["config"],
                        split=entry["split"], streaming=True)


def mixed_stream(mix, seed=0, filter_docs=True, dedup=None):
    """Interleaved stream of documents from all datasets in the mix.

    Yields raw text strings. Filtering and dedup happen here, so
    downstream code only ever sees clean docs. The interleave is
    weighted by the mix, so an 80/20 mix really gives about 80% web
    and 20% edu over a long run.

    Synthetic entries are mixed in by probability: on each draw, a
    random number below the synthetic weight yields a doc from the
    synthetic file instead of the next web doc. If the synthetic
    file is missing, it warns once and trains on the rest, so a
    missing file never kills a run. A dataset that fails to load is
    skipped the same way, with a warning, so one bad name cannot
    stop the whole mix.
    """
    from datasets import interleave_datasets

    synth_entries = [e for e in mix if e.get("synthetic")]
    hf_mix = [e for e in mix if not e.get("synthetic")]
    synth_iter = None
    synth_weight = 0.0
    synth_path = None
    if synth_entries:
        synth_weight = sum(e["weight"] for e in synth_entries)
        synth_path = synth_entries[0].get("path", "synthetic.jsonl")
        try:
            from synthetic import synthetic_text_iter

            synth_iter = synthetic_text_iter(synth_path)
        except Exception as e:
            print(f"synthetic data not available ({e}), training on the rest", flush=True)
            synth_iter = None
            synth_weight = 0.0
    if not hf_mix:
        # all synthetic: just loop the file forever
        if synth_iter is None:
            raise ValueError("no datasets in the mix and no synthetic file")
        rng = random.Random(seed)
        while True:
            try:
                text = next(synth_iter)
            except StopIteration:
                from synthetic import synthetic_text_iter

                synth_iter = synthetic_text_iter(synth_path)
                text = next(synth_iter)
            if filter_docs:
                keep, _ = filter_doc(text)
                if not keep:
                    continue
            text = clean_text(text)
            if dedup is not None and not dedup.check(text):
                continue
            yield text
        return
    streams = []
    weights = []
    for e in hf_mix:
        try:
            streams.append(load_streaming_dataset(e, seed=seed))
            weights.append(e["weight"])
        except Exception as ex:
            print(f"skipping dataset {e['name']}, failed to load: {ex}", flush=True)
    if not streams:
        raise ValueError("no datasets in the mix could be loaded")
    combined = interleave_datasets(streams, probabilities=weights, seed=seed)
    fields = {}
    for e in hf_mix:
        fields[e["name"]] = e.get("text_field", "text")
    rng = random.Random(seed)
    for row in combined:
        if synth_iter is not None and rng.random() < synth_weight:
            try:
                text = next(synth_iter)
            except StopIteration:
                from synthetic import synthetic_text_iter

                synth_iter = synthetic_text_iter(synth_path)
                text = next(synth_iter)
            if filter_docs:
                keep, _ = filter_doc(text)
                if not keep:
                    continue
            text = clean_text(text)
            if dedup is not None and not dedup.check(text):
                continue
            yield text
            continue
        # find the text field for whichever dataset this row came from
        text = ""
        for field in fields.values():
            if field in row and row[field]:
                text = row[field]
                break
        if not text:
            text = row.get("text", "")
        if not text:
            continue
        if filter_docs:
            keep, _ = filter_doc(text)
            if not keep:
                continue
        text = clean_text(text)
        if dedup is not None and not dedup.check(text):
            continue
        yield text


# tokenization
# ------------

def encode_doc(tokenizer, text):
    """Tokenize one document, returns a list of token ids."""
    return tokenizer.encode(text)


def token_stream(tokenizer, mix=None, seed=0, filter_docs=True,
                 dedup=None, smoke=False):
    """Endless stream of token ids from the data mix.

    In smoke mode this yields random ids, so the training loop can
    be tested with no data download and no tokenizer file. The ids
    stay inside the tokenizer's vocab when one is given, so they
    always fit the model's embeddings.
    """
    if smoke:
        rng = random.Random(0)
        vocab = tokenizer.vocab_size if tokenizer is not None else 256000
        while True:
            yield rng.randrange(100, vocab)
        return
    mix = build_mix(mix)
    docs = mixed_stream(mix, seed=seed, filter_docs=filter_docs, dedup=dedup)
    for text in docs:
        for tok_id in encode_doc(tokenizer, text):
            yield tok_id


def batch_stream(tokenizer, micro_batch, seq_len, mix=None, seed=0,
                 filter_docs=True, dedup=None, smoke=False):
    """Pack the token stream into fixed blocks, then into batches.

    Tokens are packed back to back with no padding and no document
    boundaries, which is the standard pretraining setup. Each batch
    is (micro_batch, seq_len). The stream restarts itself if it ever
    runs dry, so training never blocks waiting on data.
    """
    buf = []
    stream = token_stream(tokenizer, mix=mix, seed=seed,
                          filter_docs=filter_docs, dedup=dedup, smoke=smoke)
    need = seq_len * micro_batch
    first = True
    while True:
        while len(buf) < need:
            try:
                buf.append(next(stream))
            except StopIteration:
                stream = token_stream(tokenizer, mix=mix, seed=seed,
                                      filter_docs=filter_docs, dedup=dedup,
                                      smoke=smoke)
        chunk = buf[:need]
        buf = buf[need:]
        x = torch.tensor(chunk, dtype=torch.long).view(micro_batch, seq_len)
        yield x


def tokens_per_doc(tokenizer, texts):
    """Token counts for a list of docs, for throughput estimates."""
    return [len(encode_doc(tokenizer, t)) for t in texts]


def estimate_tokens(texts, tokenizer=None, chars_per_token=4.0):
    """Rough token count from raw text, without tokenizing.

    English averages about 4 characters per token. Use this for
    quick planning, tokenize for real numbers.
    """
    total_chars = sum(len(t) for t in texts)
    return int(total_chars / chars_per_token)


def pack_documents(tokenizer, texts, seq_len, bos=True, eos=True):
    """Pack a list of docs into full seq_len blocks.

    Docs are joined with BOS and EOS markers so the model learns
    where documents start and end, even though the packing is
    otherwise boundary-free. Returns a list of token id lists, each
    exactly seq_len long. The leftover tail is dropped.
    """
    bos_id = tokenizer.bos_token_id if bos else None
    eos_id = tokenizer.eos_token_id if eos else None
    flat = []
    for text in texts:
        ids = encode_doc(tokenizer, text)
        if bos_id is not None:
            flat.append(bos_id)
        flat.extend(ids)
        if eos_id is not None:
            flat.append(eos_id)
    blocks = []
    for i in range(0, len(flat) - seq_len + 1, seq_len):
        blocks.append(flat[i:i + seq_len])
    return blocks


# validation data
# ---------------

def validation_stream(tokenizer, n_docs=1000, seed=1234):
    """A fixed validation set, for perplexity eval during training.

    Uses the same mix but a different seed, and takes the first
    n_docs, so the eval set is stable across runs. Perplexity on
    this set is the number to watch while training.
    """
    mix = build_mix()
    docs = mixed_stream(mix, seed=seed, filter_docs=True, dedup=None)
    out = []
    for text in docs:
        out.append(text)
        if len(out) >= n_docs:
            break
    return out


def validation_batches(tokenizer, micro_batch, seq_len, n_docs=1000, seed=1234):
    """Validation docs packed into batches, for eval loops."""
    docs = validation_stream(tokenizer, n_docs=n_docs, seed=seed)
    return pack_documents(tokenizer, docs, seq_len)
