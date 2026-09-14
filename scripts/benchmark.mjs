/**
 * The control table. Weights frozen, held-out seeds disjoint from training and validation.
 *
 * Two questions this answers, and they are different:
 *   1. does circuit activity carry the decision?   -> trained vs SILENCED
 *   2. does THIS measured topology matter?          -> trained vs REWIRED (equal budget)
 * A project that reports only (1) and implies (2) is overclaiming.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCircuit, rewire } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { N_PARAMS, randomReadout } from '../src/flybrain/readout.ts';
import { perfectOpponent, randomOpponent, rng } from '../src/game/opponents.ts';
import { emptyTally, playGame } from '../src/training/evaluate.ts';
import { DIRECT_PARAMS, DirectController, RandomController } from '../src/training/baselines.ts';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => (existsSync(`${here}/${p}`) ? JSON.parse(readFileSync(`${here}/${p}`, 'utf8')) : null);

const doc = read('../public/data/circuit.json');
const circuit = buildCircuit(doc);
const rewired = buildCircuit(rewire(doc, 7777));

const champion = read('../public/checkpoints/champion.json');
const rewiredCk = read('../public/checkpoints/champion-rewired.json');
const directCk = read('../public/checkpoints/champion-direct.json');
const imiCk = read('../public/checkpoints/champion-imitation.json');
const imiRwCk = read('../public/checkpoints/champion-imitation-rewired.json');
const imiDirCk = read('../public/checkpoints/champion-imitation-direct.json');
if (!champion) { console.error('no champion.json — run scripts/train.mjs first'); process.exit(1); }

const theta = Float64Array.from(champion.theta);
const untrained = randomReadout(1);

// Held out: 2100001+ never appears in training (1..) or validation (1100001..).
const TEST = Array.from({ length: 125 }, (_, i) => 2100001 + i);   // 125 seeds x 8 games = 1000

function run(make, opponent, seeds) {
  const tally = emptyTally();
  for (const seed of seeds) {
    const random = rng(seed);
    const ctl = make(random);
    for (let r = 0; r < 4; r++) { playGame(ctl, 'X', opponent, random, tally); playGame(ctl, 'O', opponent, random, tally); }
  }
  return tally;
}

const shipped = imiCk ? Float64Array.from(imiCk.theta) : theta;
const conditions = [];
if (imiCk) conditions.push({ name: 'Circuit + imitation readout', make: () => new Controller(circuit, shipped), note: 'the shipped result' });
conditions.push(
  { name: 'Circuit + CEM readout', make: () => new Controller(circuit, theta), note: 'outcome-only reward' },
  { name: 'Same readout, circuit SILENCED', make: () => new Controller(circuit, shipped, true), note: 'does the circuit carry the signal?' },
  { name: 'Circuit + untrained readout', make: () => new Controller(circuit, untrained), note: 'is training doing the work?' },
);
if (imiRwCk) conditions.push({ name: 'REWIRED + imitation (equal budget)', make: () => new Controller(rewired, Float64Array.from(imiRwCk.theta)), note: 'is THIS topology special?' });
if (rewiredCk) conditions.push({ name: 'REWIRED + CEM (equal budget)', make: () => new Controller(rewired, Float64Array.from(rewiredCk.theta)), note: 'same, for the CEM readout' });
// Trained the SAME way as the shipped controller, or the bottleneck comparison is unfair.
if (imiDirCk) conditions.push({ name: `Raw board -> net (${DIRECT_PARAMS} par), imitation`, make: () => new DirectController(Float64Array.from(imiDirCk.theta)), note: 'cost of the bottleneck' });
if (directCk) conditions.push({ name: `Raw board -> net (${DIRECT_PARAMS} par), CEM`, make: () => new DirectController(Float64Array.from(directCk.theta)), note: 'same, outcome-only reward' });
conditions.push({ name: 'Random legal move', make: (random) => new RandomController(random), note: 'the floor' });

const rows = [];
for (const c of conditions) {
  const vsRandom = run(c.make, randomOpponent, TEST);
  const vsPerfect = run(c.make, perfectOpponent, TEST.slice(0, 25));
  rows.push({
    condition: c.name, note: c.note,
    vsRandom: { win: vsRandom.win, draw: vsRandom.draw, loss: vsRandom.loss, games: vsRandom.games, winRate: vsRandom.win / vsRandom.games, score: (vsRandom.win + 0.5 * vsRandom.draw) / vsRandom.games },
    vsPerfect: { losses: vsPerfect.loss, draws: vsPerfect.draw, games: vsPerfect.games },
    illegal: vsRandom.illegal + vsPerfect.illegal,
    movesPerGame: vsRandom.moves / vsRandom.games,
  });
}

const pad = (s, n) => String(s).padEnd(n);
console.log(`\nheld-out seeds 2100001+ · weights frozen · ${rows[0].vsRandom.games} games vs random, ${rows[0].vsPerfect.games} vs perfect\n`);
console.log(pad('condition', 42) + pad('vs random W/D/L', 20) + pad('score', 8) + pad('losses vs perfect', 19) + 'illegal');
console.log('-'.repeat(97));
for (const r of rows) {
  console.log(
    pad(r.condition, 42) +
    pad(`${r.vsRandom.win}/${r.vsRandom.draw}/${r.vsRandom.loss}`, 20) +
    pad(r.vsRandom.score.toFixed(3), 8) +
    pad(`${r.vsPerfect.losses}/${r.vsPerfect.games}`, 19) +
    r.illegal,
  );
}

mkdirSync(`${here}/../public/benchmarks`, { recursive: true });
writeFileSync(`${here}/../public/benchmarks/benchmark.json`, JSON.stringify({
  generated: new Date().toISOString().slice(0, 10),
  heldOutSeeds: { from: TEST[0], to: TEST.at(-1) },
  checkpoint: { seed: champion.seed, generations: champion.generations, validation: champion.championValidation },
  circuit: doc.counts, nParams: N_PARAMS, rows,
}, null, 1) + '\n');
console.log(`\nwrote public/benchmarks/benchmark.json`);
