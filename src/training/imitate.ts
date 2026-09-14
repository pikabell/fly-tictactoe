/**
 * Supervised training: imitate exact minimax, reading only circuit activity.
 *
 * WHY THIS EXISTS. CEM optimises the final game result, which is a single bit of feedback
 * after up to nine decisions. That is brutal credit assignment, and the measured outcome was
 * a controller that takes an available immediate win only 46.9% of the time — barely above
 * the 42.7% you get by playing at random. It had learned *something* (78/100 draws against
 * perfect play as X, versus 36 untrained) but almost nothing tactical.
 *
 * Tic-tac-toe is solved, so we can do far better: label every position with the moves minimax
 * says are optimal and train the readout to match. The connectome's role does not change at
 * all — the readout still sees ONLY the 16 descending activities, never the board. What
 * changes is the quality of the learning signal.
 *
 * This also turns the project into a sharper probe of the circuit. "How well can a readout
 * play when it can only see 16 fly descending cells?" is now limited by how much board
 * information survives the bottleneck, not by how noisy CEM's reward was.
 *
 * Data is collected by PLAYING, with circuit state persisting across moves exactly as in the
 * real game, and re-collected each round under the current policy (DAgger) so the training
 * states match the states the controller actually visits.
 */

import { N_HIDDEN, N_IN, N_OUT, N_PARAMS, type Readout } from '../flybrain/readout.ts';
import type { Controller } from '../flybrain/controller.ts';
import { EMPTY_BOARD, type Mark, legalMoves, outcome, play, turn } from '../game/rules.ts';
import { type Opponent, bestMoves, rng } from '../game/opponents.ts';

export type Sample = { x: Float64Array; legal: number[]; optimal: number[] };

/**
 * Play games and record, at each of the controller's turns, the 16 descending activities it
 * saw and the moves minimax considers optimal. `epsilon` injects random play so the dataset
 * covers states a greedy policy would never reach.
 */
export function collect(
  ctl: Controller,
  pool: readonly { name: string; opponent: Opponent }[],
  seeds: readonly number[],
  epsilon: number,
): Sample[] {
  const data: Sample[] = [];
  for (const seed of seeds) {
    const random = rng(seed);
    for (const { opponent } of pool) {
      for (const me of ['X', 'O'] as Mark[]) {
        ctl.newGame();
        let board = EMPTY_BOARD;
        while (!outcome(board).done) {
          const mark = turn(board);
          if (mark === me) {
            const trace = ctl.move(board, me);
            const legal = legalMoves(board);
            data.push({ x: Float64Array.from(trace.outputs), legal, optimal: bestMoves(board, me) });
            const chosen = random() < epsilon ? legal[Math.floor(random() * legal.length) % legal.length] : trace.move;
            board = play(board, chosen, mark);
          } else {
            board = play(board, opponent(board, mark, random), mark);
          }
        }
      }
    }
  }
  return data;
}

export type Shape = { nIn: number; nHidden: number; nOut: number };
export const FLY_SHAPE: Shape = { nIn: N_IN, nHidden: N_HIDDEN, nOut: N_OUT };
export const paramCount = (s: Shape) => s.nIn * s.nHidden + s.nHidden + s.nHidden * s.nOut + s.nOut;
const offsets = (s: Shape) => {
  const W1 = 0, B1 = s.nIn * s.nHidden, W2 = B1 + s.nHidden, B2 = W2 + s.nHidden * s.nOut;
  return { W1, B1, W2, B2 };
};

/** Masked softmax over legal moves only — illegal squares never receive probability. */
function forward(theta: Readout, x: Float64Array, legal: number[], h: Float64Array, p: Float64Array, shape: Shape): void {
  const { W1, B1, W2, B2 } = offsets(shape);
  for (let j = 0; j < shape.nHidden; j++) {
    let s = theta[B1 + j];
    for (let i = 0; i < shape.nIn; i++) s += theta[W1 + j * shape.nIn + i] * x[i];
    h[j] = Math.tanh(s);
  }
  p.fill(0);
  let max = -Infinity;
  const o = new Float64Array(shape.nOut);
  for (const k of legal) {
    let s = theta[B2 + k];
    for (let j = 0; j < shape.nHidden; j++) s += theta[W2 + k * shape.nHidden + j] * h[j];
    o[k] = s;
    if (s > max) max = s;
  }
  let sum = 0;
  for (const k of legal) { p[k] = Math.exp(o[k] - max); sum += p[k]; }
  for (const k of legal) p[k] /= sum;
}

export type ImitateConfig = { epochs: number; batch: number; lr: number; l2: number };
export const DEFAULT_IMITATE: ImitateConfig = { epochs: 40, batch: 64, lr: 0.01, l2: 1e-5 };

/** Adam, because plain SGD on 425 parameters with a tanh layer is needlessly fiddly. */
export function train(
  theta: Readout,
  data: Sample[],
  config: ImitateConfig,
  seed: number,
  onEpoch?: (epoch: number, loss: number, acc: number) => void,
  shape: Shape = FLY_SHAPE,
): Readout {
  const random = rng(seed);
  const n = paramCount(shape);
  const { W1, B1, W2, B2 } = offsets(shape);
  const m = new Float64Array(n), v = new Float64Array(n), g = new Float64Array(n);
  const h = new Float64Array(shape.nHidden), p = new Float64Array(shape.nOut);
  const order = data.map((_, i) => i);
  let t = 0;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let loss = 0, correct = 0;
    for (let start = 0; start < order.length; start += config.batch) {
      const idxs = order.slice(start, start + config.batch);
      g.fill(0);
      for (const idx of idxs) {
        const s = data[idx];
        forward(theta, s.x, s.legal, h, p, shape);
        const target = 1 / s.optimal.length;
        let best = s.legal[0];
        for (const k of s.legal) if (p[k] > p[best]) best = k;
        if (s.optimal.includes(best)) correct++;
        for (const k of s.optimal) loss -= target * Math.log(Math.max(p[k], 1e-12));

        const dO = new Float64Array(shape.nOut);
        for (const k of s.legal) dO[k] = p[k] - (s.optimal.includes(k) ? target : 0);
        const dH = new Float64Array(shape.nHidden);
        for (const k of s.legal) {
          g[B2 + k] += dO[k];
          for (let j = 0; j < shape.nHidden; j++) {
            g[W2 + k * shape.nHidden + j] += dO[k] * h[j];
            dH[j] += dO[k] * theta[W2 + k * shape.nHidden + j];
          }
        }
        for (let j = 0; j < shape.nHidden; j++) {
          const d = dH[j] * (1 - h[j] * h[j]);
          g[B1 + j] += d;
          for (let i = 0; i < shape.nIn; i++) g[W1 + j * shape.nIn + i] += d * s.x[i];
        }
      }
      t++;
      const scale = 1 / idxs.length;
      const bc1 = 1 - Math.pow(0.9, t), bc2 = 1 - Math.pow(0.999, t);
      for (let i = 0; i < n; i++) {
        const grad = g[i] * scale + config.l2 * theta[i];
        m[i] = 0.9 * m[i] + 0.1 * grad;
        v[i] = 0.999 * v[i] + 0.001 * grad * grad;
        theta[i] -= config.lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + 1e-8);
      }
    }
    onEpoch?.(epoch, loss / data.length, correct / data.length);
  }
  return theta;
}

/**
 * Guard against the failure that bit us here: a NaN in the parameter vector makes every
 * score NaN, `raw[m] > bestScore` is then false everywhere, and the controller silently
 * degenerates to "always play the first legal square" — which scores ~66% on optimal-move
 * accuracy and reads as mediocre learning rather than as a broken run.
 */
export function assertFinite(theta: Readout, where: string): void {
  for (let i = 0; i < theta.length; i++) {
    if (!Number.isFinite(theta[i])) throw Error(`${where}: parameter ${i} is ${theta[i]}`);
  }
}
