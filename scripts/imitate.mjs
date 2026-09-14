/** DAgger-style supervised training against exact minimax, reading only circuit activity. */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCircuit, rewire } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { randomReadout } from '../src/flybrain/readout.ts';
import { TRAINING_POOL } from '../src/game/opponents.ts';
import { DEFAULT_IMITATE, FLY_SHAPE, assertFinite, collect, paramCount, train } from '../src/training/imitate.ts';
import { DIRECT_HIDDEN, DirectController } from '../src/training/baselines.ts';
import { N_CHANNELS } from '../src/flybrain/encode.ts';
import { evaluate, score } from '../src/training/evaluate.ts';

const here = dirname(fileURLToPath(import.meta.url));
const seed = Number(process.argv[2] ?? 20260914);
const rounds = Number(process.argv[3] ?? 6);
const variant = process.argv[4] ?? 'circuit';

let doc = JSON.parse(readFileSync(`${here}/../public/data/circuit.json`, 'utf8'));
if (variant === 'rewired') doc = rewire(doc, 7777);
const circuit = buildCircuit(doc);

const shape = variant === 'direct' ? { nIn: N_CHANNELS, nHidden: DIRECT_HIDDEN, nOut: 9 } : FLY_SHAPE;
const nParams = paramCount(shape);
let theta = randomReadout(seed, 0.3, nParams);
const ctl = variant === 'direct' ? new DirectController(theta) : new Controller(circuit, theta);
const VALIDATION = Array.from({ length: 24 }, (_, i) => 1100001 + i);

let all = [];
let best = { theta: Float64Array.from(theta), val: -1, round: -1 };
const started = Date.now();
for (let r = 0; r < rounds; r++) {
  // Collect under the CURRENT policy so training states match the states it actually visits.
  // Epsilon decays: broad coverage early, on-policy refinement later.
  const epsilon = r === 0 ? 0.9 : Math.max(0.1, 0.5 - 0.08 * r);
  ctl.setReadout(theta);
  const fresh = collect(ctl, TRAINING_POOL, Array.from({ length: 60 }, (_, i) => 1 + r * 60 + i), epsilon);
  all = all.concat(fresh);
  if (all.length > 120000) all = all.slice(all.length - 120000);

  let lastAcc = 0;
  theta = train(theta, all, { ...DEFAULT_IMITATE, epochs: r === 0 ? 30 : 15 }, seed + r,
    (_e, _l, acc) => { lastAcc = acc; }, shape);
  assertFinite(theta, `round ${r}`);
  ctl.setReadout(theta);
  const val = score(evaluate(ctl, TRAINING_POOL, VALIDATION));
  // Keep the best-by-validation round, not the last: validation wobbles by ~0.03 between
  // rounds and the final round is not reliably the strongest.
  if (val > best.val) best = { theta: Float64Array.from(theta), val, round: r };
  console.log(`  round ${r}  eps ${epsilon.toFixed(2)}  samples ${all.length}  optimal-move acc ${(lastAcc * 100).toFixed(1)}%  val ${val.toFixed(3)}${val > best.val - 1e-9 ? '  <- best' : ''}`);
}

theta = best.theta;
ctl.setReadout(theta);
const val = best.val;
const seconds = (Date.now() - started) / 1000;
console.log(`\n${variant}: imitation, seed ${seed}, ${rounds} rounds, ${seconds.toFixed(1)}s, best validation ${val.toFixed(3)} (round ${best.round})`);

const out = `${here}/../public/checkpoints`;
mkdirSync(out, { recursive: true });
const name = variant === 'circuit' ? 'champion-imitation' : `champion-imitation-${variant}`;
writeFileSync(`${out}/${name}.json`, JSON.stringify({
  version: 1, dataset: 'male-cns:v1.0', variant, seed, rounds,
  method: 'supervised imitation of exact minimax; readout sees only the 16 descending activities',
  nParams, shape, championValidation: val, trainingSeconds: seconds,
  circuitCounts: doc.counts, theta: Array.from(theta),
}) + '\n');
console.log(`wrote ${out}/${name}.json`);
