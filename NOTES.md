# How to use a fly brain

This is the document the project exists for. The game is the excuse.

If you read one thing, read [the five stages](#the-five-stages). Everything in
[`../archive/`](../archive) — 85 projects, Doom and Minecraft and desktop pets and stock
traders — is a rearrangement of those five stages, and only one of them is the connectome.

---

## What a connectome actually is

A connectome is a **wiring diagram**. For MaleCNS v1.0 it is literally three tables:

| File | Size | What's in it |
|---|---:|---|
| `body-annotations-…feather` | 14 MB | 211,577 rows: `bodyId`, `type` (LC4, DNp01…), `superclass`, `somaLocation` |
| `connectome-weights-…feather` | 1,051 MB | 151,856,684 rows of `(body_pre, body_post, weight)` |
| `body-neurotransmitters-…feather` | 43 MB | per-cell predicted neurotransmitter |

That is all. 151.9 million directed pairs, 311.8 million synaptic contacts. `weight` is a
**count of synaptic contacts** between two cells — not a strength, not a sign, not a delay.

Note what is **not** there:

- no neuron dynamics — nothing says what a cell computes
- no synaptic sign — "this cell is predicted to release GABA" is not "this synapse inhibits"
- no delays, no plasticity rules, no receptor types
- no sensory encoding — nothing maps the world onto cells
- no motor decoding — nothing maps cells onto actions

Everything in that list has to be **invented by you**. That is the single most important
fact about this whole field of toy projects, and it is why the five stages below matter more
than any particular game.

---

## The five stages

Every connectome game is this pipeline. Stage 2 is measured. The rest is engineering.

```
   world state                                                     action
       |                                                              ^
       v                                                              |
  [1] ENCODER  ->  [2] GRAPH  ->  [3] DYNAMICS  ->  [4] READOUT  ------+
   assumed          MEASURED       assumed           assumed & trained

                        [5] CONTROLS: how you find out whether stage 2 mattered
```

### 1. Encoder — assumed

Turn the world into external drive on named cells. In this project:

```
18 channels: mine[0..8], theirs[0..8]
channel c  ->  one visual-projection cell  ->  u = 2*(feature - 0.5)
```

`src/flybrain/encode.ts`. Assigning board square 4 to cell type LC12 has **no biological
meaning whatsoever**. LC12 responds to small moving objects in a real fly; it has never seen
a game board. Fly Dino maps obstacle distance to LC4; fly-chess maps 64 squares × 12 piece
channels onto sensory neurons via Poisson stimulation. All of these are arbitrary. The
connectome contributes nothing at this stage.

### 2. Graph — **measured**

The one real thing. Our extraction (`scripts/build_circuit.py`) is deterministic, uses
anatomy alone, and is fixed before any training:

1. **18 inputs** — for each named visual-projection type, the cell with the most contacts
   onto descending neurons.
2. **16 outputs** — the descending cells those inputs contact most.
3. **64 bridges** — strongest two-hop intermediates, ranked by
   `min(contacts in from inputs, contacts out to outputs)`.
4. **Every** measured edge among those 98 cells is kept — single-contact and recurrent
   included. Nothing is synthesized.

Result: **98 cells, 1,494 directed edges, 27,762 synaptic contacts.**

Two things about that number. First, it is **0.001%** of the measured connectome — we
filtered 151.9M pairs down to 19.9M (both ends visible cells with somata), then to 1,494.
Every project in the archive is doing something in this range. The "fly brain" in a
fly-brain game is a hand-picked splinter, and **the selection rule is the real design
decision** — far more consequential than anything downstream.

Second, we didn't choose the famous cells; they chose themselves. Asking only "which
descending cells do visual projection neurons contact most" returns **DNp01 — the Giant
Fiber, the escape-reflex command neuron** — plus DNp03, DNp04, DNp05, DNp09, DNp11, DNp35,
DNp103, pIP1. That is a real anatomical fact: the visual system's highest-contact path to
the motor system is the escape circuit. A fly is mostly wired to notice something coming
and leave.

### 3. Dynamics — assumed

How activity flows along the measured edges. `src/flybrain/dynamics.ts`:

```
sign:  acetylcholine -> +1,  GABA/glutamate -> -1,  unclear -> 0
W[j][i] = c[j][i] * s[j] / sum_k( c[k][i] * |s[k]| )        # normalize per postsynaptic cell
h <- 0.3*h + 0.7*tanh( u + 1.4 * (W^T h) )                  # 3 synchronous iterations
```

Four numbers — leak 0.3, gain 1.4, 3 iterations, readout gain 4 — plus a sign policy.
**None of them come from the data.** `h` is dimensionless: not a firing rate, not a membrane
voltage. Anyone showing you millivolts either ran a real LIF model or is overclaiming.

There *is* a more principled option, and fly-chess takes it: a genuine leaky
integrate-and-fire model with Shiu et al.'s parameters (−52 mV rest, −45 mV threshold,
τ_m 20 ms), validated against Brian2 to 2.6e-13 mV. It is more honest physiology and much
slower. It also produced, in their own reported test, **10,068 spikes and zero spikes in
descending neurons** — their readout runs off sub-threshold membrane offsets. Worth
remembering: "the brain ran" and "the brain drove the output" are different claims.

### 4. Readout — assumed, and the only part that learns

16 descending activities → 16 tanh hidden → 9 square scores. **425 parameters.**
Illegal squares are masked to −∞ before the argmax, so an illegal move is impossible by
construction rather than something training has to learn.

Trained with the **cross-entropy method**: sample 64 candidates from a diagonal Gaussian,
keep the best 8, move the distribution towards them, repeat. No gradients, ~30 lines.

**Nothing biological changes during training.** The graph, the signs, the dynamics and the
encoder are frozen. Whatever skill appears was learned by a 425-parameter perceptron reading
a fly circuit — there is no plasticity anywhere in this model, and no fly learned anything.

### 5. Controls — how you find out whether any of it mattered

This is what separates an experiment from a demo, and it is where most of the archive is
thin. See [`docs/experiment.md`](docs/experiment.md) for our table. The two questions are
**different**:

- **trained vs. silenced** → does circuit activity carry the decision?
- **trained vs. rewired at equal budget** → does *this measured topology* matter, or would
  any graph with the same degree distribution do?

A project that reports only the first and implies the second is overclaiming. Ours reports
both, whichever way they came out.

---

## Two things measurement changed my mind about

**Raw synapse counts badly predict functional influence.** Direct contacts from our input
cells onto descending cells span **247:1** (LC4 has 247, LC10a has 1). I expected the quiet
channels to be mute. Measuring properly — flip one channel on an otherwise empty board, see
how far the 16 descending activities move — the spread collapses to **21:1**, and the
ordering scrambles: **LC9 is the most influential channel despite only 74 direct contacts,
while LC4 with 247 ranks fourth.** Per-postsynaptic normalization divides out contact
volume, and 64 bridge cells give every channel an indirect route. Both of those are *our
assumptions*. The connectome constrained the answer; the modelling choices decided it.

**A plausible-looking result is the dangerous kind.** My first version of that measurement
was wrong: driving "one channel" still drives all 18, because an empty square encodes as
u = −1, not 0. Every channel scored a uniform 0.44–0.56 and I nearly wrote down "the
channels are balanced." It looked exactly like a finding. Only the second version showed the
21:1 spread. The failure mode here isn't code that crashes — it's code that returns a
number you were happy to believe.

---

## Reading the archive

`../archive/` has 85 repos. Sorted by which stage they vary:

| Stage varied | Projects | What to steal |
|---|---|---|
| **1. Encoder** | `cobanov__flyjump`, `tolatolatop__fly-chess`, `nftechie__stonkfly` | how to turn game state into cell drive; flyjump's 8-channel table is the clearest |
| **2. Graph** | `philshiu__Drosophila_brain_model`, `YijieYin__connectome_data_prep`, `dhakalnirajan__axonweave` | selection and data prep; axonweave exposes MaleCNS as a torch `nn.Module` |
| **3. Dynamics** | `tolatolatop__fly-chess`, `eonsystemspbc__fly-brain`, `eonfathom__FastFly` | real LIF instead of our tanh; Brian2/CUDA/WASM backends |
| **4. Readout** | `liuzihe02__fly-craftax` (PPO), `cobanov__flyjump` (CEM), `nftechie__flm` | how to train an action decoder on circuit activity |
| **5. Controls** | `eganeganegan__flydoom`, `cobanov__flyjump`, `5p00kyy__neuroterrarium` | rewired/shuffled baselines, negative results reported honestly |
| **Body** | `TuragaLab__flybody`, `NeLy-EPFL__flygym` | MuJoCo physics, if you want real locomotion instead of a game |
| **Viewers** | `cobanov__fly-connectome-template` (this project's base), `murthylab__codex` | 3D anatomy without writing WebGL |

Start with `cobanov__flyjump/docs/experiment.md`. It is the best-documented protocol in the
list and this project follows it closely enough that differences are informative.

## If you want to build your own

1. **Pick your dataset first, and know which one it is.** MaleCNS is an adult *male* CNS;
   FlyWire/FAFB is an adult *female* brain. They are different animals and not interchangeable.
2. **Write the selection rule before you look at results**, and make it deterministic
   (ties broken by ID). Otherwise you will drift into choosing a circuit that works.
3. **Check reachability before training.** `build_circuit.py` fails loudly if any output
   cell can't be reached from an input through non-zero-sign edges. A dead circuit cannot
   be rescued by a readout, and you want to know in seconds, not after a training run.
4. **Look at the activity before you trust the score.** No saturation, no dead cells, and it
   should *change* when the input changes. Ours: mean |h| 0.30, max 0.92, 0 saturated,
   0 dead, mean |Δh| 0.17 between two boards.
5. **Build the silenced control at the same time as the real one.** If you build it later
   you will be tempted not to.
6. **Say what is measured and what is assumed, on the page.** It costs one paragraph.

## Reproducing this project

```sh
../data/fetch_malecns.sh                                  # ~1.1 GB, one time
cd flytictactoe
uv run --with pyarrow --with numpy python scripts/build_circuit.py ../data/malecns
npm ci && npm test
node --experimental-strip-types scripts/train.mjs 20260914 250 circuit
node --experimental-strip-types scripts/benchmark.mjs
npm run dev
```

`scripts/diagnose.mjs` prints the activity health check and per-channel influence table.
