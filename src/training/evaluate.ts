/**
 * Playing a controller against an opponent, and scoring it.
 *
 * Draw = +0.5 deliberately. In tic-tac-toe a draw IS the correct result against good play,
 * so a scoring rule that only rewarded wins would push the controller into unsound
 * speculative play against opponents it cannot actually beat.
 */

import { EMPTY_BOARD, type Mark, legalMoves, outcome, play, turn } from '../game/rules.ts';
import { type Opponent, rng } from '../game/opponents.ts';
import type { Controller } from '../flybrain/controller.ts';

export type Tally = { win: number; draw: number; loss: number; illegal: number; moves: number; games: number };

export const emptyTally = (): Tally => ({ win: 0, draw: 0, loss: 0, illegal: 0, moves: 0, games: 0 });

export const score = (t: Tally): number => (t.win + 0.5 * t.draw) / Math.max(1, t.games);

/** One game. The controller plays `me`; activity resets at the start, persists across moves. */
export function playGame(ctl: Controller, me: Mark, opponent: Opponent, random: () => number, tally: Tally): void {
  ctl.newGame();
  let board = EMPTY_BOARD;
  while (!outcome(board).done) {
    const mark = turn(board);
    let m: number;
    if (mark === me) {
      m = ctl.move(board, me).move;
      if (!legalMoves(board).includes(m)) { tally.illegal++; m = legalMoves(board)[0]; }
    } else {
      m = opponent(board, mark, random);
    }
    board = play(board, m, mark);
    tally.moves++;
  }
  const end = outcome(board);
  if (end.draw) tally.draw++;
  else if (end.winner === me) tally.win++;
  else tally.loss++;
  tally.games++;
}

/** Plays both sides against every opponent in the pool, so neither colour is neglected. */
export function evaluate(
  ctl: Controller,
  pool: readonly { name: string; opponent: Opponent }[],
  seeds: readonly number[],
): Tally {
  const tally = emptyTally();
  for (const seed of seeds) {
    const random = rng(seed);
    for (const { opponent } of pool) {
      playGame(ctl, 'X', opponent, random, tally);
      playGame(ctl, 'O', opponent, random, tally);
    }
  }
  return tally;
}
