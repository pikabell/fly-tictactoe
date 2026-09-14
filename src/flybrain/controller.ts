/**
 * The whole pipeline in one place: board -> features -> circuit -> readout -> move.
 *
 * ONE code path. The visible game and the headless training rollouts both call `move()`,
 * so what you watch on screen is exactly what was trained. (A surprising number of
 * projects have two implementations that quietly disagree.)
 */

import { type Board, type Mark, legalMoves } from '../game/rules.ts';
import type { Circuit } from './circuit.ts';
import { CircuitState, DEFAULT_DYNAMICS, type Dynamics } from './dynamics.ts';
import { N_CHANNELS, drive, features } from './encode.ts';
import { type Readout, N_IN, N_OUT, chooseMove, scores } from './readout.ts';

export type Trace = {
  features: Float64Array;   // 18 board channels
  outputs: Float64Array;    // 16 scaled descending activities
  scores: Float64Array;     // 9 raw square scores
  move: number;
};

export class Controller {
  private readonly state: CircuitState;
  private readonly feat = new Float64Array(N_CHANNELS);
  private readonly u: Float64Array;
  private readonly out = new Float64Array(N_IN);
  private readonly sc = new Float64Array(N_OUT);

  private readonly circuit: Circuit;
  private theta: Readout;
  /** Control condition: hold circuit activity at zero to test whether it matters. */
  private readonly silenced: boolean;

  constructor(circuit: Circuit, theta: Readout, silenced = false, dyn: Dynamics = DEFAULT_DYNAMICS) {
    this.circuit = circuit;
    this.theta = theta;
    this.silenced = silenced;
    this.state = new CircuitState(circuit, dyn);
    this.u = new Float64Array(circuit.n);
  }

  setReadout(theta: Readout): void { this.theta = theta; }

  /** Call at the start of each game. Activity persists across moves within a game. */
  newGame(): void { this.state.reset(); }

  move(board: Board, me: Mark): Trace {
    features(board, me, this.feat);
    if (this.silenced) {
      this.out.fill(0);
    } else {
      drive(this.circuit, this.feat, this.u);
      this.state.step(this.u);
      this.state.readOutputs(this.out);
    }
    scores(this.theta, this.out, this.sc);
    const move = chooseMove(this.sc, legalMoves(board));
    return { features: this.feat, outputs: this.out, scores: this.sc, move };
  }

  /** Live cell activity, for the 3D view. */
  get activity(): Float64Array { return this.state.h; }
}
