/**
 * Board -> external drive on named cells.
 *
 * ASSUMED, and arbitrarily so. Assigning "square 4 is mine" to cell type LC12 has no
 * biological meaning whatsoever. LC12 is a visual projection neuron that in a real fly
 * responds to small moving objects; it has never seen a noughts-and-crosses grid.
 *
 * This is the stage every connectome game hand-waves, so it is worth being blunt: the
 * connectome contributes nothing here. We are choosing 18 cells and injecting numbers.
 */

import type { Board } from '../game/rules.ts';
import type { Mark } from '../game/rules.ts';
import { other } from '../game/rules.ts';
import type { Circuit } from './circuit.ts';

/** 18 channels: 0-8 "my mark is here", 9-17 "their mark is here". */
export const N_CHANNELS = 18;

/** Board is canonicalized to the side to move, so no turn-parity channel is needed. */
export function features(board: Board, me: Mark, into: Float64Array): Float64Array {
  const them = other(me);
  for (let i = 0; i < 9; i++) {
    into[i] = board[i] === me ? 1 : 0;
    into[i + 9] = board[i] === them ? 1 : 0;
  }
  return into;
}

/**
 * A driven cell receives u = 2*(feature - 0.5), so an empty square is -1 and an occupied
 * one +1: the circuit is driven in both directions rather than only being switched on.
 * Every other cell receives zero external drive.
 */
export function drive(circuit: Circuit, feat: Float64Array, into: Float64Array): Float64Array {
  into.fill(0);
  for (let ch = 0; ch < N_CHANNELS; ch++) into[circuit.inputCells[ch]] = 2 * (feat[ch] - 0.5);
  return into;
}
