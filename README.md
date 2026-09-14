# Fly Tic-Tac-Toe

**A 98-cell measured *Drosophila* MaleCNS circuit picks the moves in a game of tic-tac-toe.**
Play it, watch the cells light up in the real 3D soma atlas, and train the controller from
random weights in your own browser.

**▶ [Play it](https://pikabell.github.io/fly-tictactoe/)**

## What is real here, and what is not

This matters more than the game, so it goes first.

**Measured** — the wiring, and only the wiring. 98 cells from
[MaleCNS v1.0](https://male-cns.janelia.org/) with every directed edge between them:
1,494 edges, 27,762 synaptic contacts, selected by anatomy alone before any training.
Single-contact and recurrent edges are kept. No edge is synthesized.

**Assumed** — everything else:

- **the encoder.** Board square → cell type is arbitrary. LC12 responds to small moving
  objects in a real fly; it has never seen a game board.
- **the sign policy.** Predicted acetylcholine → +1, GABA/glutamate → −1, unclear → 0. That
  is a modelling choice applied to a *prediction*, not a measurement of synaptic sign.
- **the dynamics.** `h ← 0.3·h + 0.7·tanh(u + 1.4·Wᵀh)`, three iterations. `h` is
  dimensionless — not a firing rate, not a membrane voltage.
- **the readout.** 16 descending activities → 16 hidden → 9 square scores, 425 parameters,
  trained to imitate exact minimax while never seeing the board.

**Trained** — those 425 readout parameters, and nothing else. The graph, signs, dynamics and
encoder are frozen. There is no plasticity anywhere in this model and no fly learned anything.

So the defensible claim is *"a small trainable readout learned to play using activity from a
measured fly circuit, and it depends on that activity."* Not *"a fly brain plays
tic-tac-toe."* Whether the **measured topology** was the reason it works is a separate
question, and the control table below answers it directly.

## The control table

Held-out seeds (2,100,001+), disjoint from training and validation, weights frozen first.

<!-- BENCHMARK_TABLE -->
| Controller | vs random (W/D/L) | score | losses vs perfect | illegal | what it answers |
|---|---|---:|---:|---:|---|
| Circuit + imitation readout | 759/189/52 | 0.854 | 11/200 | 0 | the shipped result |
| Circuit + CEM readout | 800/88/112 | 0.844 | 88/200 | 0 | outcome-only reward |
| Same readout, circuit SILENCED | 600/89/311 | 0.644 | 153/200 | 0 | does the circuit carry the signal? |
| Circuit + untrained readout | 518/125/357 | 0.581 | 159/200 | 0 | is training doing the work? |
| REWIRED + imitation (equal budget) | 778/143/79 | 0.850 | 19/200 | 0 | is THIS topology special? |
| REWIRED + CEM (equal budget) | 815/58/127 | 0.844 | 62/200 | 0 | same, for the CEM readout |
| Raw board -> net (457 par), imitation | 828/153/19 | 0.904 | 6/200 | 0 | cost of the bottleneck |
| Raw board -> net (457 par), CEM | 800/122/78 | 0.861 | 42/200 | 0 | same, outcome-only reward |
| Random legal move | 417/128/455 | 0.481 | 185/200 | 0 | the floor |

1000 games vs random and 200 vs perfect per controller. Score counts a draw as 0.5, because a draw is the correct result in tic-tac-toe. Checkpoint: seed 20260914, 250 generations.

1000 games vs random and 200 vs perfect per controller. Score counts a draw as 0.5, because a draw is the correct result in tic-tac-toe. Checkpoint: seed 20260914, 250 generations.

1000 games vs random and 200 vs perfect per controller. Score counts a draw as 0.5, because a draw is the correct result in tic-tac-toe. Checkpoint: seed 20260914, 250 generations.

Read it as two separate questions:

- **trained vs. silenced** — does circuit activity carry the decision?
- **trained vs. rewired at equal budget** — does *this* topology matter, or would any graph
  with the same degree distribution do?

<!-- BENCHMARK_READING -->
**Does the training signal matter?** Enormously, and this was the fix for "it always loses". CEM optimises only the final game result — one bit of feedback after up to nine decisions — and the controller it produced blocked an immediate threat just **36%** of the time, against **33%** for random play. Training the same 425 parameters to imitate exact minimax instead, still reading **only** the 16 descending activities and never the board, takes blocking to **56%** and cuts losses against perfect play from **88/200 to 11/200**. The connectome's role is unchanged; only the quality of the learning signal changed.
**Does the circuit carry the decision?** Yes. Silencing it drops the score from **0.854 to 0.644** (−0.209) with the same 425 weights, and losses against perfect play rise from 11/200 to 153/200. A silenced controller is literally board-blind — it produces identical scores for every position, which the test suite asserts, so what it retains is one fixed square preference that happens to be a passable opening. Training matters too: an untrained readout on the live circuit scores 0.581.
**Does *this* measured topology matter?** **No — not measurably.** A degree-preserving rewired graph, retrained at an identical budget, scores 0.850 against the real circuit's 0.854 (the real circuit is slightly ahead on losses to perfect play, 11 vs 19 of 200 — one seed, so not a finding). That is the honest result and it is worth stating plainly: the readout learned to use a recurrent network with the fly's degree distribution and contact-count statistics, and shuffling which cell connects to which cost it nothing. Anyone claiming fly wiring is *good at* a task needs exactly this control, and it is the one most often missing.
**What does the bottleneck cost?** Feeding the raw 18-channel board into a network of comparable size (457 parameters) scores 0.904. Routing the board through 98 fly cells and reading only 16 descending ones costs about 0.051. The circuit is a constraint on the task, not an advantage — which is what you would expect, and is fine, as long as nobody says otherwise.
**Nobody reaches the real bar.** Tic-tac-toe is solved, so a genuinely good controller should *never* lose to perfect play. Against exact minimax the shipped controller still loses **11/200** games, and the unconstrained direct-board control, trained identically, loses 6/200. A few-hundred-parameter feed-forward readout with no search is not a strong tic-tac-toe player, and beating a random opponent ~76% of the time should not be mistaken for one.
**The floor:** random legal play scores 0.481. The silenced controller scores 0.644, *above* random against a random opponent — a board-blind network still plays one fixed square order, and a fixed order beats guessing. Against perfect play that illusion collapses: 153/200 losses. Beating a weak opponent is not evidence of seeing the board. **Illegal moves: 0 across every condition** — masking makes them impossible by construction, not by training.

Full protocol and limitations: [`docs/experiment.md`](docs/experiment.md).
How any of this works, and how to build your own: [`NOTES.md`](NOTES.md).

## Run it

Node 22.18+:

```sh
npm ci
npm run dev            # play
npm test               # 23 tests
```

Rebuild the circuit from the official release (~1.1 GB download, one time):

```sh
../data/fetch_malecns.sh
uv run --with pyarrow --with numpy python scripts/build_circuit.py ../data/malecns
```

`build_circuit.py` verifies the SHA-256 of all three source tables and fails if any output
cell is unreachable from an input, rather than emitting a circuit that cannot work.

Reproduce the training and the table:

```sh
npm run diagnose                              # activity health + per-channel influence

npm run imitate -- 20260914 20 circuit        # the shipped controller
npm run imitate -- 20260914 20 rewired        # topology control
npm run imitate -- 20260914 20 direct         # bottleneck control

npm run train   -- 20260914 250 circuit       # the CEM comparison
npm run train   -- 20260914 250 rewired
npm run train   -- 20260914 250 direct

npm run benchmark && npm run report           # control table -> README and docs
npm run tactics                               # takes-win / blocks / optimal-move rates
```

`npm run report` regenerates the table above from `public/benchmarks/benchmark.json`, so the
published numbers are never typed by hand.

## Things worth trying in the UI

- **Silence circuit** — descending activity is pinned to zero and play collapses. This is the
  control that shows the circuit is carrying the decision rather than decorating a readout.
- **Reset state each move** — activity normally persists across moves within a game, so the
  circuit carries a trace of how the game developed. Turn it off and see whether it mattered.
- **Train in this browser** — real training in a Web Worker, using the same modules as the
  visible game, about a minute. Switch the method between *imitate minimax* and
  *cross-entropy (outcome only)* and watch the same circuit and the same 425 parameters
  produce very different play. That gap is the most instructive thing here.

## Credits and licences

Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by
[Mert Cobanov](https://github.com/cobanov), under the
[Cobanov Template Attribution License 1.0](LICENSE) — an attribution-required,
source-available licence, **not** OSI-approved. If you fork this, that credit has to stay in
both your UI and your README.

Protocol and dynamics follow
[Fly Dino](https://github.com/cobanov/flyjump)'s published
[experiment protocol](https://github.com/cobanov/flyjump/blob/main/docs/experiment.md),
widened from 8 input channels to 18. Found via
[awesome-fly](https://github.com/cobanov/awesome-fly).

**MaleCNS v1.0** connectome data: **CC BY 4.0**, FlyEM / HHMI Janelia, University of
Cambridge, MRC Laboratory of Molecular Biology, and Google Research —
[data and publication](https://male-cns.janelia.org/download/). **Flybody** mesh:
Apache-2.0, [TuragaLab](https://github.com/TuragaLab/flybody). Raw connectome tables are not
redistributed here; `scripts/build_circuit.py` downloads and verifies them.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
