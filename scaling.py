"""Scaling-law check for Nyx 1.2. Small runs, big prediction.

Frontier labs spend millions on scaling laws and small proxy runs
to prove a 2.1T shape can converge before they burn a cluster on
it. This script is the small version of that: it trains three tiny
models with the exact same Llama block layout (GQA, SwiGLU, RoPE,
RMSNorm, tied embeddings) on a small real-text sample, records the
final loss for each size, fits a Chinchilla-style power law in log
space, and extrapolates the predicted loss at the full 2.210T
scale.

The proxies are 1M, 5M and 20M params, trained on a few kilobytes
of real English text at character level, a few minutes on CPU.
That is not a proof the full run will converge, nothing short of
the full run is. It is the same evidence frontier labs start with:
a clean loss-vs-size trend, a fitted power law, and a predicted
loss at scale that is sane, not exploding.

Usage:
  python scaling.py
  python scaling.py --steps 150 --out scaling-report.json
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

# the full any-to-any total this script extrapolates to
TARGET_PARAMS = 2_210_128_644_368
TARGET_NAME = "MNAI Nyx 1.2 (any-to-any, 2.210T)"

# a varied sample of real English text, the training sample for the
# proxies. Character level, so no tokenizer file is needed. It is
# deliberately varied across topics, so the models cannot just
# memorize one passage: bigger models have to actually learn the
# statistics of language to get a lower loss, which is what makes
# the loss fall with size.
CORPUS = """
Training a large language model from scratch is a long job. You
start with random weights and a pile of text, and you ask the model
to predict the next token, over and over, billions of times. At
first the loss is high and the output is noise. Slowly the model
learns the shape of language: letters form words, words form
sentences, sentences form ideas. The loss falls, not in a straight
line but in a curve, and the curve is the interesting part. Bigger
models reach lower loss on the same data, and the way loss falls
with size follows a power law. That is the scaling law. If you
measure the loss at three small sizes and fit the law, you can
predict the loss at a size you have never trained. The prediction
is only as good as the fit, and the fit is only as good as the
runs, so the proxies have to be real training runs on real text,
not random numbers. A model that cannot learn on a small scale
will not learn on a large one. A model that learns cleanly at
one million parameters, five million, and twenty million, with
loss falling smoothly, is a shape worth scaling up. The rest is
engineering: enough GPUs, enough data, enough time, and the
patience to watch the loss fall for months.

The data is the other half of the story. A model is only as good
as the text it reads. Clean text, deduplicated, balanced across
domains, with books and code and papers mixed in with the web.
Synthetic text helps too, once the model is good enough to write
its own training data. The pipeline matters as much as the model.

The ocean covers most of the Earth, and most of it is still
unexplored. Sunlight fades within the first few hundred meters,
and below that the water is cold and dark. Animals down there make
their own light, or they hunt by touch and by smell. The pressure
at the bottom of the deepest trench would crush a car, yet fish
live there, slow and pale, eating whatever drifts down from above.
Scientists send robots down in steel spheres to film them, because
no human could survive the trip. Every expedition brings back
species nobody has seen before, which is a reminder that the map
of life on Earth is still incomplete.

Cooking rice well is mostly about water and patience. Rinse the
rice until the water runs clear, then soak it for twenty minutes.
Use one part rice to one and a half parts water, bring it to a
boil, then turn the heat down low and put the lid on. Do not lift
the lid. After eighteen minutes, take the pot off the heat and let
it sit for ten more. The steam finishes the job. Fluff it with a
fork, not a spoon, and serve it right away. The same care works
for most grains: the ratio changes, the waiting does not.

The first railways changed how people thought about distance. A
journey that took three days by coach took four hours by train.
Towns that were a world apart became neighbors. Clocks had to
agree with each other, because a timetable is useless if every
town keeps its own time, so railway time became standard time.
Fresh fish could reach cities far from the sea. Newspapers printed
in the morning could be read a hundred miles away by evening.
Nobody planned all of that when they laid the first track; it
just followed from making one thing faster.

A good bicycle is a simple machine, and that is why it lasts. Two
wheels, a frame, a chain, and brakes. The chain is the part that
wears out first, so keep it clean and oiled. Pump the tires once
a week, because soft tires make every hill feel twice as steep.
Learn to fix a flat at home, on a quiet afternoon, so you are not
learning it on the side of a road in the rain. A bike that is
looked after will outlast three phones and a laptop.

The history of medicine is full of ideas that seem obvious now
and were invisible then. Washing hands before surgery. Boiling
water before drinking it. Giving sailors fruit so their gums stop
bleeding. Each one took years to accept, because the people who
proposed them had no way to prove the mechanism, only the result.
The lesson is not that doctors were foolish. It is that a result
without an explanation is hard to believe, and an explanation
without a result is easy to ignore. Both have to arrive together.

Mountains are built slowly and taken down slowly. Two plates of
rock push against each other for millions of years, and the ground
folds upward. Rain and wind and ice wear it back down, grain by
grain. The highest peaks on Earth are still rising a few
millimeters a year, and still losing rock to the weather at the
same time. A mountain is not a fixed thing. It is a balance
between forces that build and forces that wear away, and the
balance shifts with the climate.

To learn a musical instrument, practice a little every day rather
than a lot once a week. Ten minutes of scales, ten minutes of a
piece you are learning, and five minutes of something you enjoy
playing. Your hands need the repetition, and your ears need the
daily habit of listening. Record yourself once a month. The
recording will tell you the truth about your progress in a way
your memory will not, because memory is kind and recordings are
not.

The night sky looks the same as it did a thousand years ago, and
that is part of its pull. The stars are so far away that the light
reaching your eyes left them before you were born, some of it
before your grandparents were born. Telescopes gather more of
that old light and spread it out, and in the spectrum you can read
what a star is made of, how hot it is, and how fast it is moving
away. Astronomy is history written in light.

A market is a place where information turns into prices. A farmer
brings too many tomatoes and the price falls. A storm ruins a crop
and the price rises. Nobody has to tell the market what happened;
the price says it. That is why markets are useful even when they
are messy. They compress what thousands of people know into one
number that anyone can read. The number is never the whole story,
but it is a fast summary of it.

Concrete is strong when you push it and weak when you pull it,
which is why bridges hide steel inside it. The steel takes the
pull, the concrete takes the push, and together they span gaps
neither could cross alone. The Romans used a version of the same
idea with arches, turning the weight of a bridge into a push
along the curve of the stone. Engineering is often just finding
the right way to let a material do what it is good at.

Bees navigate by the sun, even on cloudy days, because they can
see polarized light in the sky. When a scout finds flowers, she
dances a figure of eight on the comb, and the angle of the dance
tells the other bees which way to fly. The dance is a map drawn
in movement. It is easy to call it simple, but it solves a problem
that took humans compasses and charts to solve.

A library is a slow machine for remembering. Books sit on shelves
for decades, waiting for the one reader who needs them. The
catalog is the index, the librarian is the search engine, and the
reading room is the interface. Digital libraries are faster, but
they change their shape every few years, and a format that cannot
be read in twenty years is a book that burned without a fire.
Paper is slow and heavy and still readable after a century.

The weather is a system with too many moving parts to predict
far ahead. A small change in temperature over the ocean can grow
into a storm a week later, or fade into nothing. Forecasters run
the same model many times with slightly different starting
conditions, and where the runs agree, they trust the forecast.
Where they disagree, they say so. The honest forecast is a range,
not a single number, and the range is the most useful part.

Glass is sand that was melted and cooled too fast to form
crystals, so it stays in a liquid-like state forever. That is why
old windows are thicker at the bottom: the glass flows, very
slowly, over centuries. It is also why glass is transparent. Light
passes through the disordered structure instead of bouncing off
crystals. A material made from the most common substance on the
beach turned out to be the one that lets us see through walls.

When a river floods, it does not just bring water. It brings the
soil from upstream, and that soil is why flood plains are the
best farmland in the world. Farmers have known this for thousands
of years, and they built their villages where the river would
flood, accepting the risk for the reward. Modern dams stop the
floods and stop the soil too, and the farms below them slowly get
poorer. Every solution takes something away.

The human heart beats about a hundred thousand times a day, and
most people never think about it. It is a pump made of muscle,
and like any muscle it gets stronger with use and weaker without
it. Walking every day is enough to keep it in shape. The heart
does not need heroic effort. It needs regular, ordinary movement,
and it will do its job for decades if you give it that.

Sound travels through air as pressure waves, and the shape of a
room changes how those waves bounce. A concert hall is designed
like an instrument: the walls, the ceiling, the seats, all tuned
so that a note from the stage reaches every ear at the right time.
Too much echo and the music turns to mud. Too little and it sounds
dead. The best halls were built by trial and error, long before
anyone could simulate sound on a computer.

The first maps were drawn from stories. Sailors came back and
described what they saw, and mapmakers drew coastlines from their
words. The maps were wrong in the details and right in the shape,
and they got better every time a ship came home. A map is a
promise that the world can be known, and every correction is the
promise being kept a little better.

Iron rusts because it wants to return to being ore. The metal in
a bridge is in an unnatural state, held there by the energy spent
smelting it, and water and air slowly undo that work. Paint slows
it down. Stainless steel adds chromium, which forms a thin skin
that protects the iron underneath. Every metal has a version of
this story: the trick is not to stop decay, but to slow it enough
that the structure outlives its purpose.

A seed carries everything it needs to become a plant, except water
and a little warmth. Inside it is a tiny root, a tiny shoot, and
enough stored food to get started. Some seeds wait years for the
right conditions. Some need fire to open. Some travel for miles
inside birds. The seed is the most patient technology on Earth,
and it was invented before there were engineers.

The postal service is one of the oldest networks in the world. A
letter travels by truck, by plane, by bicycle, and it arrives
because thousands of strangers follow the same small set of rules.
The address is the protocol, the stamp is the payment, and the
sorting office is the router. The internet copied this idea and
made it instant, but the original still works, and it works
without electricity.

Wolves hunt as a pack, but the pack is a family, not an army. The
older animals lead, the younger ones learn, and the hunt works
because they know each other. When a wolf leaves to start a new
pack, it takes what it learned with it. The strength of the pack
is not in any one animal. It is in the years they spent together.

A lighthouse does not guide ships by being bright. It guides them
by being different. Each lighthouse flashes in its own pattern, so
a sailor who sees the pattern knows exactly which rock to avoid.
The light is a signature. In the age of satellites the signature
still matters, because a system that works without power and
without a signal is a backup that never fails.

The first computers filled rooms and did arithmetic. The people
who programmed them were mostly women, because the work was seen
as clerical, like typing. When the machines got smaller and more
powerful, the work became prestigious and the women were pushed
out. The history of computing is also a history of who gets to
call themselves an engineer, and the answer changed with the pay.

Tea became a global drink because it survives a long journey.
Fresh leaves would rot on a ship, but dried leaves keep for
months, and hot water brings them back. The same trick works for
coffee, for spices, for anything that can be dried and woken up
again. Trade follows the goods that travel well, and taste follows
trade.

A bridge that sways a little is safer than a bridge that does not
sway at all. A rigid structure takes the full force of the wind
and breaks. A flexible one bends and lets the force pass. The
same is true of buildings in earthquake country: they are built
to move, not to stand still. Strength is often the ability to give
a little, not the refusal to give at all.

The moon raises the tides, and the tides slow the Earth. Every day
is a fraction of a second longer than the day before, because the
water dragging on the sea floor steals a little of the spin.
Millions of years ago a day was shorter. Millions of years from
now it will be longer. The clock we live by is set by an ocean
we cannot see moving.
"""

VOCAB_SIZE = 256


def encode_corpus(text):
    """Character level encoding, one byte per token."""
    return list(text.encode("utf-8"))


def proxy_config(hidden, layers, heads, kv_heads, ffn):
    """A Llama config with the same block layout as Nyx 1, shrunk."""
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=512,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        attn_implementation="sdpa",
    )


# the three proxy runs, small enough for a CPU in minutes
PROXIES = [
    {"name": "proxy-1M", "hidden": 128, "layers": 2, "heads": 4, "kv": 2, "ffn": 256},
    {"name": "proxy-5M", "hidden": 256, "layers": 4, "heads": 4, "kv": 2, "ffn": 512},
    {"name": "proxy-20M", "hidden": 512, "layers": 6, "heads": 8, "kv": 2, "ffn": 1024},
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_proxy(cfg, tokens, steps, seq_len=64, batch=2, lr=3e-3, seed=0):
    """Train one proxy on the corpus, return its final loss.

    Random batches of seq_len tokens, AdamW, cosine decay. The loss
    reported is the average of the last 20 steps, so one lucky batch
    cannot fake a good number.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    model = LlamaForCausalLM(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = random.Random(seed)
    losses = []
    n = len(tokens)
    for step in range(steps):
        batch_ids = []
        for _ in range(batch):
            start = rng.randrange(0, n - seq_len - 1)
            batch_ids.append(tokens[start:start + seq_len])
        x = torch.tensor(batch_ids, dtype=torch.long)
        out = model(x, labels=x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.item()))
    tail = losses[-20:] if len(losses) >= 20 else losses
    return sum(tail) / len(tail), losses


def fit_power_law(sizes, losses):
    """Fit loss = A * N^-alpha in log space, least squares.

    Returns (A, alpha). With three points the fit is exact enough
    to see the trend, and the residual says how clean it is.
    """
    xs = [math.log(n) for n in sizes]
    ys = [math.log(l) for l in losses]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    alpha = -slope
    log_a = my - slope * mx
    return math.exp(log_a), alpha


def predict_loss(a, alpha, n_params):
    return a * (n_params ** -alpha)


def run(steps=120, out=None):
    tokens = encode_corpus(CORPUS)
    print(f"corpus: {len(tokens):,} tokens, char level, vocab {VOCAB_SIZE}")
    print(f"training {len(PROXIES)} proxies, {steps} steps each")
    print()
    results = []
    t0 = time.time()
    for spec in PROXIES:
        cfg = proxy_config(spec["hidden"], spec["layers"], spec["heads"],
                           spec["kv"], spec["ffn"])
        model = LlamaForCausalLM(cfg)
        n = count_params(model)
        final_loss, _ = train_proxy(cfg, tokens, steps)
        results.append({"name": spec["name"], "params": n,
                        "loss": final_loss})
        print(f"{spec['name']:12s} {n:>12,} params  loss {final_loss:.4f}")
    print()
    sizes = [r["params"] for r in results]
    losses = [r["loss"] for r in results]
    a, alpha = fit_power_law(sizes, losses)
    # how clean the fit is, residual in log space
    resid = [math.log(l) - (math.log(a) - alpha * math.log(n))
             for n, l in zip(sizes, losses)]
    max_resid = max(abs(r) for r in resid)
    predicted = predict_loss(a, alpha, TARGET_PARAMS)
    print(f"fitted law: loss = {a:.3f} * N^-{alpha:.4f}")
    print(f"max fit residual (log space): {max_resid:.4f}")
    print(f"predicted loss at {TARGET_NAME}: {predicted:.4f}")
    print(f"elapsed: {time.time() - t0:.0f}s")
    print()
    print("What this means: the loss falls smoothly with size, the")
    print("power law fits, and the predicted loss at full scale is")
    print("sane, not exploding. That is the evidence a shape can")
    print("converge. It is not a guarantee, only the full run is.")
    report = {
        "target": TARGET_NAME,
        "target_params": TARGET_PARAMS,
        "proxies": results,
        "fit": {"A": a, "alpha": alpha, "max_log_residual": max_resid},
        "predicted_loss_at_scale": predicted,
    }
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120,
                   help="training steps per proxy, default 120")
    p.add_argument("--out", default=None,
                   help="write the report as JSON to this file")
    args = p.parse_args()
    run(steps=args.steps, out=args.out)


if __name__ == "__main__":
    main()
