/// <reference lib="webworker" />
/**
 * Training runs here so the page stays responsive. It uses the SAME controller, circuit and
 * rules modules as the visible game — there is no separate "fast" simulator that might
 * quietly disagree with what you watched.
 *
 * Default method is imitation of exact minimax, which is what the shipped checkpoint uses.
 * CEM is kept selectable because the comparison between the two is the most instructive
 * thing in the project: identical circuit, identical 425 parameters, wildly different play.
 */

import { buildCircuit, type CircuitDoc } from '../flybrain/circuit.ts';
import { Controller } from '../flybrain/controller.ts';
import { N_PARAMS, randomReadout } from '../flybrain/readout.ts';
import { TRAINING_POOL } from '../game/opponents.ts';
import { DEFAULT_CEM, runCem } from './cem.ts';
import { evaluate, score } from './evaluate.ts';
import { DEFAULT_IMITATE, assertFinite, collect, train } from './imitate.ts';

export type Method = 'imitation' | 'cem';
export type TrainRequest = { kind: 'train'; doc: CircuitDoc; seed: number; rounds: number; method: Method };
export type TrainProgress = { kind: 'progress'; method: Method; step: number; steps: number; detail: number; championValidation: number };
export type TrainDone = { kind: 'done'; method: Method; theta: number[]; championValidation: number; seconds: number };

const TRAIN_SEEDS_PER_GEN = 12;
const VALIDATION = Array.from({ length: 24 }, (_, i) => 1100001 + i);

let stop = false;

self.onmessage = (event: MessageEvent<TrainRequest | { kind: 'stop' }>) => {
  const data = event.data;
  if (data.kind === 'stop') { stop = true; return; }
  if (data.kind !== 'train') return;
  stop = false;

  const circuit = buildCircuit(data.doc);
  const ctl = new Controller(circuit, new Float64Array(N_PARAMS));
  const fitness = (theta: Float64Array, seeds: number[]) => { ctl.setReadout(theta); return score(evaluate(ctl, TRAINING_POOL, seeds)); };
  const started = Date.now();
  const post = (m: TrainProgress | TrainDone) => self.postMessage(m);

  if (data.method === 'cem') {
    const { champion, history } = runCem(data.seed, { ...DEFAULT_CEM, generations: data.rounds }, {
      train: (theta, g) => fitness(theta, Array.from({ length: TRAIN_SEEDS_PER_GEN }, (_, i) => 1 + g * TRAIN_SEEDS_PER_GEN + i)),
      validate: (theta) => fitness(theta, VALIDATION),
      shouldStop: () => stop,
      onGeneration: (g) => post({ kind: 'progress', method: 'cem', step: g.generation, steps: data.rounds, detail: g.validation, championValidation: g.championValidation }),
    });
    post({ kind: 'done', method: 'cem', theta: Array.from(champion), championValidation: history.at(-1)?.championValidation ?? 0, seconds: (Date.now() - started) / 1000 });
    return;
  }

  // Imitation: DAgger — recollect under the current policy each round so the training states
  // match the states the controller actually visits, with circuit state persisting as in play.
  let theta = randomReadout(data.seed, 0.3);
  let all: ReturnType<typeof collect> = [];
  let best = { theta: Float64Array.from(theta), val: -1 };
  for (let r = 0; r < data.rounds; r++) {
    if (stop) break;
    const epsilon = r === 0 ? 0.9 : Math.max(0.1, 0.5 - 0.08 * r);
    ctl.setReadout(theta);
    all = all.concat(collect(ctl, TRAINING_POOL, Array.from({ length: 60 }, (_, i) => 1 + r * 60 + i), epsilon));
    if (all.length > 120000) all = all.slice(all.length - 120000);
    theta = train(theta, all, { ...DEFAULT_IMITATE, epochs: r === 0 ? 30 : 15 }, data.seed + r);
    assertFinite(theta, `imitation round ${r}`);
    ctl.setReadout(theta);
    const val = fitness(theta, VALIDATION);
    if (val > best.val) best = { theta: Float64Array.from(theta), val };
    post({ kind: 'progress', method: 'imitation', step: r, steps: data.rounds, detail: val, championValidation: best.val });
  }
  post({ kind: 'done', method: 'imitation', theta: Array.from(best.theta), championValidation: best.val, seconds: (Date.now() - started) / 1000 });
};
