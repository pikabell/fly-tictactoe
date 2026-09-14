/**
 * Cross-entropy method: diagonal Gaussian, score-based policy search.
 *
 * Not backprop, not PPO, not NEAT. Keep a per-parameter mean and sigma, sample candidates,
 * keep the best few ("elites"), move the distribution towards them, repeat. About 30 lines
 * of real work, no gradients, and it is what Fly Dino uses — so results are comparable.
 *
 * It touches ONLY the 425 readout parameters. The measured graph never changes.
 */

import { N_PARAMS, type Readout } from '../flybrain/readout.ts';
import { rng } from '../game/opponents.ts';

export type CemConfig = {
  candidates: number; elites: number; generations: number;
  initSigma: number; sigmaFloor: number; blend: number;
  /** Parameter count. Defaults to the fly readout; the direct-board control uses its own. */
  nParams?: number;
};

export const DEFAULT_CEM: CemConfig = {
  candidates: 64, elites: 8, generations: 60,
  initSigma: 0.8, sigmaFloor: 0.07, blend: 0.3, // 0.3 old + 0.7 elite
};

export type Generation = { generation: number; bestTraining: number; validation: number; championValidation: number };

/** Box-Muller, sharing the seeded stream so a run is exactly reproducible. */
function gaussian(random: () => number): number {
  const u1 = Math.max(random(), 1e-12), u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export type CemHooks = {
  /** Fitness on freshly drawn training opponents for this generation. */
  train: (theta: Readout, generation: number) => number;
  /** Fitness on the fixed validation set. Never used to pick candidates, only the champion. */
  validate: (theta: Readout) => number;
  onGeneration?: (g: Generation, champion: Readout) => void;
  shouldStop?: () => boolean;
};

export function runCem(seed: number, config: CemConfig, hooks: CemHooks): { champion: Readout; history: Generation[] } {
  const random = rng(seed);
  const n = config.nParams ?? N_PARAMS;
  const mean = new Float64Array(n);
  const sigma = new Float64Array(n).fill(config.initSigma);

  let champion = new Float64Array(n);
  for (let i = 0; i < n; i++) champion[i] = 0.7 * gaussian(random);
  let championValidation = hooks.validate(champion);

  const history: Generation[] = [];

  for (let g = 0; g < config.generations; g++) {
    if (hooks.shouldStop?.()) break;

    const population: Readout[] = [];
    // Candidate 0 is the current champion, so a generation can never lose ground.
    population.push(Float64Array.from(champion));
    for (let c = 1; c < config.candidates; c++) {
      const theta = new Float64Array(n);
      for (let i = 0; i < n; i++) theta[i] = mean[i] + sigma[i] * gaussian(random);
      population.push(theta);
    }

    const scored = population
      .map(theta => ({ theta, fitness: hooks.train(theta, g) }))
      .sort((a, b) => b.fitness - a.fitness);

    const elites = scored.slice(0, config.elites);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (const e of elites) m += e.theta[i];
      m /= elites.length;
      let v = 0;
      for (const e of elites) v += (e.theta[i] - m) ** 2;
      v = Math.sqrt(v / elites.length);
      mean[i] = config.blend * mean[i] + (1 - config.blend) * m;
      sigma[i] = Math.max(config.sigmaFloor, config.blend * sigma[i] + (1 - config.blend) * v);
    }

    // Validation is separate from training fitness, so a candidate that got lucky on this
    // generation's opponents cannot become the champion.
    const best = scored[0].theta;
    const validation = hooks.validate(best);
    if (validation > championValidation) { champion = Float64Array.from(best); championValidation = validation; }

    const record = { generation: g, bestTraining: scored[0].fitness, validation, championValidation };
    history.push(record);
    hooks.onGeneration?.(record, champion);
  }

  return { champion, history };
}
