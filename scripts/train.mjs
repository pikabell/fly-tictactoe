/** Reproducible CEM training run. `node --experimental-strip-types scripts/train.mjs <seed> <generations>` */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCircuit, rewire } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { N_PARAMS } from '../src/flybrain/readout.ts';
import { TRAINING_POOL } from '../src/game/opponents.ts';
import { DEFAULT_CEM, runCem } from '../src/training/cem.ts';
import { evaluate, score } from '../src/training/evaluate.ts';
import { DIRECT_PARAMS, DirectController } from '../src/training/baselines.ts';

const here = dirname(fileURLToPath(import.meta.url));
const seed = Number(process.argv[2] ?? 20260914);
const generations = Number(process.argv[3] ?? DEFAULT_CEM.generations);
const variant = process.argv[4] ?? 'circuit';           // 'circuit' | 'rewired'
const outDir = process.argv[5] ?? `${here}/../public/checkpoints`;

let doc = JSON.parse(readFileSync(`${here}/../public/data/circuit.json`, 'utf8'));
if (variant === 'rewired') doc = rewire(doc, 7777);
const circuit = buildCircuit(doc);
const nParams = variant === 'direct' ? DIRECT_PARAMS : N_PARAMS;

// Disjoint seed ranges. Training opponents are redrawn each generation; validation is fixed.
// Each seed plays all 4 pool opponents as both X and O, so N seeds = 8N games.
// 12 training seeds = 96 games per candidate: enough that CEM is ranking skill, not noise.
const TRAIN_SEEDS_PER_GEN = 12;
const trainSeeds = (g) => Array.from({ length: TRAIN_SEEDS_PER_GEN }, (_, i) => 1 + g * TRAIN_SEEDS_PER_GEN + i);
const VALIDATION = Array.from({ length: 24 }, (_, i) => 1100001 + i);

const ctl = variant === 'direct'
  ? new DirectController(new Float64Array(nParams))
  : new Controller(circuit, new Float64Array(nParams));
const fitness = (theta, seeds) => { ctl.setReadout(theta); return score(evaluate(ctl, TRAINING_POOL, seeds)); };

const started = Date.now();
let last = 0;
const { champion, history } = runCem(seed, { ...DEFAULT_CEM, generations, nParams }, {
  train: (theta, g) => fitness(theta, trainSeeds(g)),
  validate: (theta) => fitness(theta, VALIDATION),
  onGeneration: (g) => {
    if (g.generation % 5 === 0 || g.generation === generations - 1) {
      process.stdout.write(`  gen ${String(g.generation).padStart(3)}  train ${g.bestTraining.toFixed(3)}  val ${g.validation.toFixed(3)}  champion ${g.championValidation.toFixed(3)}\n`);
    }
    last = g.championValidation;
  },
});

const seconds = (Date.now() - started) / 1000;
console.log(`\n${variant}: seed ${seed}, ${generations} generations, ${seconds.toFixed(1)}s, champion validation ${last.toFixed(3)}`);

mkdirSync(outDir, { recursive: true });
const name = variant === 'circuit' ? 'champion' : `champion-${variant}`;
writeFileSync(`${outDir}/${name}.json`, JSON.stringify({
  version: 1, dataset: 'male-cns:v1.0', variant, seed, generations,
  cem: { ...DEFAULT_CEM, generations },
  nParams, championValidation: last, trainingSeconds: seconds,
  circuitCounts: doc.counts,
  theta: Array.from(champion),
}) + '\n');
writeFileSync(`${outDir}/${name}-history.json`, JSON.stringify({ seed, variant, history }) + '\n');
console.log(`wrote ${outDir}/${name}.json`);
