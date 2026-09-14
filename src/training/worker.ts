/// <reference lib="webworker" />
/**
 * Training runs here so the page stays responsive. It uses the SAME controller, circuit and
 * rules modules as the visible game — there is no separate "fast" simulator that might
 * quietly disagree with what you watched.
 */

import { buildCircuit, type CircuitDoc } from '../flybrain/circuit.ts';
import { Controller } from '../flybrain/controller.ts';
import { N_PARAMS } from '../flybrain/readout.ts';
import { TRAINING_POOL } from '../game/opponents.ts';
import { DEFAULT_CEM, runCem } from './cem.ts';
import { evaluate, score } from './evaluate.ts';

export type TrainRequest = { kind: 'train'; doc: CircuitDoc; seed: number; generations: number };
export type TrainProgress = { kind: 'progress'; generation: number; generations: number; bestTraining: number; validation: number; championValidation: number };
export type TrainDone = { kind: 'done'; theta: number[]; championValidation: number; history: unknown[]; seconds: number };

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

  const { champion, history } = runCem(data.seed, { ...DEFAULT_CEM, generations: data.generations }, {
    train: (theta, g) => fitness(theta, Array.from({ length: TRAIN_SEEDS_PER_GEN }, (_, i) => 1 + g * TRAIN_SEEDS_PER_GEN + i)),
    validate: (theta) => fitness(theta, VALIDATION),
    shouldStop: () => stop,
    onGeneration: (g) => {
      const message: TrainProgress = { kind: 'progress', generation: g.generation, generations: data.generations, bestTraining: g.bestTraining, validation: g.validation, championValidation: g.championValidation };
      self.postMessage(message);
    },
  });

  const done: TrainDone = {
    kind: 'done', theta: Array.from(champion),
    championValidation: history.at(-1)?.championValidation ?? 0,
    history, seconds: (Date.now() - started) / 1000,
  };
  self.postMessage(done);
};
