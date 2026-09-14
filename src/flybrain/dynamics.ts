/**
 * How activity moves through the measured graph.
 *
 * ALL OF THIS IS ASSUMED. The connectome says which cells touch which, and with how many
 * synaptic contacts. It does not say what a cell computes, how fast it leaks, or how strongly
 * a contact drives its target. Those are the four numbers in DEFAULT_DYNAMICS below, and
 * changing them changes the behaviour of the "fly brain" completely.
 *
 *     h <- leak*h + (1-leak) * tanh( u + gain * (W^T h) )
 *
 * `h` is dimensionless. It is NOT a firing rate and NOT a membrane voltage. Anyone who tells
 * you their connectome game shows neural activity in volts is either running a real LIF model
 * (see tolatolatop/fly-chess, which does) or overclaiming.
 */

import type { Circuit } from './circuit.ts';

export type Dynamics = {
  /** Fraction of the previous state carried over each iteration. */
  leak: number;
  /** How hard recurrent input drives the state, relative to external drive. */
  gain: number;
  /** Synchronous update steps per decision. */
  iterations: number;
  /** Scaling applied to output-cell activity before the trainable readout sees it. */
  readoutGain: number;
};

export const DEFAULT_DYNAMICS: Dynamics = { leak: 0.3, gain: 1.4, iterations: 3, readoutGain: 4 };

export class CircuitState {
  readonly h: Float64Array;
  private readonly circuit: Circuit;
  private readonly dyn: Dynamics;
  constructor(circuit: Circuit, dyn: Dynamics = DEFAULT_DYNAMICS) {
    this.circuit = circuit;
    this.dyn = dyn;
    this.h = new Float64Array(circuit.n);
  }

  reset(): void { this.h.fill(0); }

  /**
   * Run `iterations` synchronous steps with external drive `u` (length n).
   * Synchronous: every cell updates from the same previous state, so ordering cannot
   * smuggle in a sequential computation the anatomy does not support.
   */
  step(u: Float64Array): void {
    const { incoming, n } = this.circuit;
    const { leak, gain, iterations } = this.dyn;
    const h = this.h;
    const next = new Float64Array(n);
    for (let it = 0; it < iterations; it++) {
      for (let i = 0; i < n; i++) {
        const { from, weight } = incoming[i];
        let sum = 0;
        for (let k = 0; k < from.length; k++) sum += weight[k] * h[from[k]];
        next[i] = leak * h[i] + (1 - leak) * Math.tanh(u[i] + gain * sum);
      }
      h.set(next);
    }
  }

  /** Activity of the 16 descending cells, scaled - the only thing the readout can see. */
  readOutputs(into: Float64Array): Float64Array {
    const { outputCells } = this.circuit;
    for (let i = 0; i < outputCells.length; i++) into[i] = this.dyn.readoutGain * this.h[outputCells[i]];
    return into;
  }
}
