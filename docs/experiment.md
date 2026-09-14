# Fly Tic-Tac-Toe: protocol and evidence

Recorded 2026-09-14. Environment ID: `flytictactoe-malecns-v1`.

## Question and scope

Can a small trainable readout learn tic-tac-toe when its only inputs are the computed
activities of a measured MaleCNS circuit? Does removing that activity destroy the behaviour?
And — separately — does the *measured topology* matter, or would any graph with the same
degree distribution work as well?

This experiment demonstrates numerical learning and a real causal computation path. It does
**not** model a whole brain, measured electrophysiology, muscles, learning in a living
animal, or plasticity of biological synapses.

Tic-tac-toe was chosen because it is **solved**: 5,478 reachable positions, and perfect play
always draws. So "never loses to perfect play" is an exact, checkable bar rather than an
impression, and there is no room for a demo to look better than it is.

## Measured circuit and selection

Data: [MaleCNS v1.0](https://male-cns.janelia.org/download/), flat connectome, minimum
confidence 0.5. Creators: FlyEM / HHMI Janelia and collaborators; CC BY 4.0.

| Source table | SHA-256 |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| `body-neurotransmitters-male-cns-v1.0.feather` | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |

The annotations hash matches the `sourceSha256` in the template's own atlas manifest, so the
circuit and the 3D points on screen come from the same file.

Selection is deterministic, uses **anatomy alone**, and is fixed before training. Ties are
broken by body ID, so re-running produces a byte-identical `circuit.json`.

1. For each of 18 visual-projection types (`LC4, LC6, LC9, LC11, LC12, LC13, LC15, LC16,
   LC17, LC18, LC20a, LC21, LC22, LC25, LPLC1, LPLC2, LPLC4, LC10a`), retain the cell with
   the largest total direct contact count onto descending neurons with soma coordinates.
2. Take the union of the top two descending targets per input cell; fill to 16 by total
   contact rank.
3. Add the 64 strongest two-hop bridge cells, ranked by the minimum of summed input from
   selected inputs and summed output to selected outputs.
4. Retain **every** measured directed edge among the selected cells, including single-contact
   and recurrent edges. Do not synthesize edges.

Result: **98 cells, 1,494 directed edges, 27,762 synaptic contacts** — 18 input, 64 bridge,
16 output. Transmitters: 64 acetylcholine, 26 GABA, 7 glutamate, 1 unclear. Density 0.157,
mean degree 15.2, 25.5% single-contact edges, no self-edges.

Only edges among cells that are *visible in the atlas* are considered, so every selected cell
can be drawn and passes the template's replay validator. Of 151,856,684 directed pairs in the
release, 19,899,310 have both ends in that set; we keep 1,494 of them — about **0.001%** of
the measured connectome. This selection and its boundary truncation are strong inductive
biases, not a representative sample of the CNS.

Note on naming: MaleCNS v1.0 has no type `LC20` (Fly Dino's list uses an older vintage); it
is split into `LC20a` and `LC20b`. `LC20a` is used.

## Observation encoder

18 engineered channels — `mine[0..8]` then `theirs[0..8]`, each 1.0 when that square holds
that mark. The board is canonicalized to the side to move, so no turn-parity channel is
needed. A driven cell receives `u = 2·(feature − 0.5)`; every other cell receives zero.

Assignment of board channel to cell type is an **arbitrary fixed engineering encoder with no
claimed biological interpretation.** No rule-based policy supplies targets or actions during
training.

## Dynamics

Presynaptic sign from the predicted consensus neurotransmitter: acetylcholine `+1`,
GABA/glutamate `−1`, unclear/modulatory `0` (such a cell keeps its anatomical edges but
contributes no modelled drive). Incoming signed weights are normalized by total absolute
signed contact count at each postsynaptic cell:

```
W[j][i] = c[j][i] · s[j] / Σ_k ( c[k][i] · |s[k]| )
h ← 0.3·h + 0.7·tanh( u + 1.4·(Wᵀh) )
```

Three synchronous iterations per move. `h` resets to zero at the start of each game and
persists across moves within a game. JavaScript Float64 throughout, in both the visible game
and headless rollouts — one code path, asserted by test.

The normalized signed activity is **dimensionless**; it is neither firing rate nor membrane
voltage. The sign mapping, gain, leak and iteration count are simplified assumptions, not
calibrated physiology.

### Health check before training

Measured on an untrained readout (`npm run diagnose`): mean |h| 0.30, max 0.92, **0 saturated
cells, 0 dead cells**; mean |Δh| 0.17 between two different boards. 1,471 of 1,494 edges
carry drive (23 have a sign-0 presynaptic cell). Fly Dino's constants transferred unchanged.

### Per-channel influence

Direct contacts from input cells onto output cells span **247:1** (LC4 247 … LC10a 1). But
measured marginal influence — flip one channel on an otherwise empty board, measure
displacement of the 16 descending activities — spans only **21:1**, and reorders: **LC9 is
most influential (0.263) despite 74 direct contacts, while LC4 with 247 ranks fourth.**
Per-postsynaptic normalization and the 64 bridge cells account for this. Both are modelling
assumptions, not measurements.

## Readout and learning

`4·h[output cells]` → 16 inputs → 16 tanh hidden → 9 linear scores, with biases:
**425 parameters**. Illegal squares are masked to `−∞` before the argmax, so illegal moves are
impossible by construction. No path carries board information to the readout except through
the circuit — asserted by test (a silenced controller produces identical scores for every
board).

**Only the 425 readout parameters change** under either method below. The graph, signs,
dynamics and encoder are fixed; there is no synaptic plasticity anywhere in this model.
Validation is the fixed seed set 1,100,001–1,100,024 for both; training seeds start at 1.

### Method A — cross-entropy method (the comparison)

Diagonal Gaussian CEM: score-based policy search, not DQN, PPO, NEAT or backpropagation.
64 candidates, 8 elites, 250 generations, seed `20260914`; mean/sigma ← 0.3·old + 0.7·elite,
sigma floor 0.07; candidate 0 preserves the champion. Fitness is the mean score over 96 games
per candidate (12 seeds × 4 pool opponents × both colours); a draw counts 0.5 because a draw
is the correct tic-tac-toe result. Opponent pool: random, ε-greedy perfect at ε = 0.5 and 0.2,
and exact minimax. The champion is replaced only on strictly improved validation score.

An early run used 24 games per candidate and produced a bouncing validation curve and a
champion that froze at generation 35 — CEM was ranking noise rather than skill. 96 games per
candidate fixed it. Recorded because the failure looked like convergence.

### Method B — imitation of exact minimax (**shipped**)

CEM's only feedback is the final game result: one bit after up to nine decisions. Measured
over all 4,520 reachable positions, the CEM controller took an available immediate win 46.9%
of the time and blocked an immediate threat 36.0% — against 42.4% and 33.2% for random play.
It had learned an opening, not tactics.

Because tic-tac-toe is solved, every position can be labelled with the moves minimax calls
optimal. The readout is trained to match, by Adam on a masked softmax cross-entropy over
legal moves (lr 0.01, batch 64, L2 1e-5). Training data is collected by **playing**, with
circuit state persisting across moves exactly as in the real game, and re-collected under the
current policy each round (DAgger, 20 rounds, ε decaying 0.9 → 0.1) so training states match
the states the controller actually visits. The best-by-validation round is kept, not the last.

**The readout still sees only the 16 descending activities and never the board.** The
connectome's role is identical under both methods; only the learning signal differs. Result:
blocking 36.0% → 56.1%, and losses against perfect play 88/200 → 11/200.

The direct-board control is trained by the same method with the same shape, so the
bottleneck comparison in the table below is like-for-like.

A NaN bug in the first version of this trainer is worth recording: a local `batch` shadowed
`config.batch`, so `1 / batch` divided by an *array*. Every gradient became NaN, the
parameter vector went NaN, and because `raw[m] > bestScore` is false for NaN the controller
silently degenerated to "always play the first legal square" — which scores 66% optimal-move
accuracy and reads as mediocre learning rather than as a broken run. `assertFinite()` now
guards every round.

## Held-out evaluation

Test seeds **2,100,001–2,100,125**, absent from training and validation. Architecture and
weights frozen before this test.

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

<!-- BENCHMARK_READING -->
**Does the training signal matter?** Enormously, and this was the fix for "it always loses". CEM optimises only the final game result — one bit of feedback after up to nine decisions — and the controller it produced blocked an immediate threat just **36%** of the time, against **33%** for random play. Training the same 425 parameters to imitate exact minimax instead, still reading **only** the 16 descending activities and never the board, takes blocking to **56%** and cuts losses against perfect play from **88/200 to 11/200**. The connectome's role is unchanged; only the quality of the learning signal changed.
**Does the circuit carry the decision?** Yes. Silencing it drops the score from **0.854 to 0.644** (−0.209) with the same 425 weights, and losses against perfect play rise from 11/200 to 153/200. A silenced controller is literally board-blind — it produces identical scores for every position, which the test suite asserts, so what it retains is one fixed square preference that happens to be a passable opening. Training matters too: an untrained readout on the live circuit scores 0.581.
**Does *this* measured topology matter?** **No — not measurably.** A degree-preserving rewired graph, retrained at an identical budget, scores 0.850 against the real circuit's 0.854 (the real circuit is slightly ahead on losses to perfect play, 11 vs 19 of 200 — one seed, so not a finding). That is the honest result and it is worth stating plainly: the readout learned to use a recurrent network with the fly's degree distribution and contact-count statistics, and shuffling which cell connects to which cost it nothing. Anyone claiming fly wiring is *good at* a task needs exactly this control, and it is the one most often missing.
**What does the bottleneck cost?** Feeding the raw 18-channel board into a network of comparable size (457 parameters) scores 0.904. Routing the board through 98 fly cells and reading only 16 descending ones costs about 0.051. The circuit is a constraint on the task, not an advantage — which is what you would expect, and is fine, as long as nobody says otherwise.
**Nobody reaches the real bar.** Tic-tac-toe is solved, so a genuinely good controller should *never* lose to perfect play. Against exact minimax the shipped controller still loses **11/200** games, and the unconstrained direct-board control, trained identically, loses 6/200. A few-hundred-parameter feed-forward readout with no search is not a strong tic-tac-toe player, and beating a random opponent ~76% of the time should not be mistaken for one.
**The floor:** random legal play scores 0.481. The silenced controller scores 0.644, *above* random against a random opponent — a board-blind network still plays one fixed square order, and a fixed order beats guessing. Against perfect play that illusion collapses: 153/200 losses. Beating a weak opponent is not evidence of seeing the board. **Illegal moves: 0 across every condition** — masking makes them impossible by construction, not by training.

## Limitations

- 98 cells of ~124,000 visible somata. Not a brain, not a whole-circuit simulation.
- The encoder, sign policy, dynamics and readout are all assumed. Only the wiring is measured.
- Neurotransmitter values are **predictions**, and predicted transmitter is not synaptic sign.
- `h` is dimensionless. No spikes, no membrane voltages, no delays, no plasticity.
- One training seed per condition. Three replicas would be the minimum for any claim about
  the difference between conditions, and are not included here. The trained-vs-rewired gap
  (0.854 vs 0.850) is well inside what a second seed could move.
- The Flybody mesh on screen is display only; nothing in this project drives a body.
- Nothing here shows that fly connectivity is *suited* to tic-tac-toe, and the rewired
  control is reported specifically so that reading is not available.
