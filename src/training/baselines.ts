/**
 * Control conditions. Without these the project is a demo, not an experiment.
 *
 * `DirectController` skips the circuit entirely and feeds the 18 board channels straight
 * into a network of comparable size. It is the upper bound: it shows how much of the task
 * is lost by forcing the board through a 98-cell fly circuit and reading only 16 descending
 * cells. If the fly controller matched it, that would mean the bottleneck costs nothing —
 * which would be surprising and worth doubting.
 */

import { type Board, type Mark, legalMoves } from '../game/rules.ts';
import { N_CHANNELS, features } from '../flybrain/encode.ts';
import { chooseMove } from '../flybrain/readout.ts';
import type { Trace } from '../flybrain/controller.ts';

export const DIRECT_HIDDEN = 16;
export const DIRECT_PARAMS = N_CHANNELS * DIRECT_HIDDEN + DIRECT_HIDDEN + DIRECT_HIDDEN * 9 + 9; // 457

export class DirectController {
  private theta: Float64Array;
  private readonly feat = new Float64Array(N_CHANNELS);
  private readonly sc = new Float64Array(9);
  constructor(theta: Float64Array) { this.theta = theta; }
  setReadout(theta: Float64Array): void { this.theta = theta; }
  newGame(): void { /* stateless: no circuit, so nothing to reset */ }
  get activity(): Float64Array { return new Float64Array(0); }

  move(board: Board, me: Mark): Trace {
    features(board, me, this.feat);
    const t = this.theta;
    let p = 0;
    const hidden = new Float64Array(DIRECT_HIDDEN);
    for (let j = 0; j < DIRECT_HIDDEN; j++) {
      let s = 0;
      for (let i = 0; i < N_CHANNELS; i++) s += t[p++] * (2 * (this.feat[i] - 0.5));
      hidden[j] = Math.tanh(s + t[N_CHANNELS * DIRECT_HIDDEN + j]);
    }
    p = N_CHANNELS * DIRECT_HIDDEN + DIRECT_HIDDEN;
    for (let k = 0; k < 9; k++) {
      let s = 0;
      for (let j = 0; j < DIRECT_HIDDEN; j++) s += t[p++] * hidden[j];
      this.sc[k] = s + t[N_CHANNELS * DIRECT_HIDDEN + DIRECT_HIDDEN + DIRECT_HIDDEN * 9 + k];
    }
    return { features: this.feat, outputs: new Float64Array(0), scores: this.sc, move: chooseMove(this.sc, legalMoves(board)) };
  }
}

/** The floor: uniformly random legal play. */
export class RandomController {
  private random: () => number;
  constructor(random: () => number) { this.random = random; }
  setReadout(): void {}
  newGame(): void {}
  get activity(): Float64Array { return new Float64Array(0); }
  move(board: Board, _me: Mark): Trace {
    const legal = legalMoves(board);
    return {
      features: new Float64Array(N_CHANNELS), outputs: new Float64Array(0), scores: new Float64Array(9),
      move: legal[Math.floor(this.random() * legal.length) % legal.length],
    };
  }
}
