/**
 * The only trainable thing in this project.
 *
 * 16 descending activities -> 16 tanh hidden units -> 9 square scores.
 *   16*16 + 16 + 16*9 + 9 = 425 parameters.
 *
 * The measured graph, the sign policy, the dynamics and the encoder are all FROZEN.
 * Training moves these 425 numbers and nothing else. That matters for the honesty of the
 * result: whatever skill appears, it was learned by a 425-parameter perceptron reading a
 * fly circuit, not by the fly circuit itself. No biological synapse changes anywhere.
 *
 * Illegal squares are masked to -Infinity before the argmax, so an illegal move is
 * impossible by construction rather than something training has to learn to avoid.
 */

export const N_IN = 16;
export const N_HIDDEN = 16;
export const N_OUT = 9;
export const N_PARAMS = N_IN * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUT + N_OUT; // 425

export type Readout = Float64Array; // flat parameter vector, length N_PARAMS

export function scores(theta: Readout, input: Float64Array, into: Float64Array): Float64Array {
  let p = 0;
  const hidden = new Float64Array(N_HIDDEN);
  for (let j = 0; j < N_HIDDEN; j++) {
    let s = 0;
    for (let i = 0; i < N_IN; i++) s += theta[p++] * input[i];
    hidden[j] = Math.tanh(s + theta[N_IN * N_HIDDEN + j]);
  }
  p = N_IN * N_HIDDEN + N_HIDDEN;
  for (let k = 0; k < N_OUT; k++) {
    let s = 0;
    for (let j = 0; j < N_HIDDEN; j++) s += theta[p++] * hidden[j];
    into[k] = s + theta[N_IN * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUT + k];
  }
  return into;
}

/** Highest-scoring legal square. Ties go to the lowest index, so play is deterministic. */
export function chooseMove(raw: Float64Array, legal: readonly number[]): number {
  let best = legal[0], bestScore = -Infinity;
  for (const m of legal) {
    if (raw[m] > bestScore) { bestScore = raw[m]; best = m; }
  }
  return best;
}

export function randomReadout(seed: number, sigma = 0.7): Readout {
  let a = seed >>> 0;
  const rnd = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const theta = new Float64Array(N_PARAMS);
  for (let i = 0; i < N_PARAMS; i++) {
    // Box-Muller, matching the sampler used during training.
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
    theta[i] = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return theta;
}
