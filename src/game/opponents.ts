/**
 * Opponents the circuit is trained and measured against.
 *
 * `perfect` is exact memoized minimax over the whole game tree. Tic-tac-toe is a
 * solved game: with perfect play by both sides the result is always a draw, so
 * "never loses to perfect" is a hard, checkable bar rather than a vibe.
 */

import { type Board, type Mark, legalMoves, other, outcome, play, key } from './rules.ts';

/** Deterministic, seedable PRNG (mulberry32). Same stream in UI and in training. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const pick = <T>(items: readonly T[], random: () => number): T =>
  items[Math.floor(random() * items.length) % items.length];

/** Exact value of a position for `mark`: +1 win, 0 draw, -1 loss, with fewer plies preferred. */
const memo = new Map<string, number>();

function value(board: Board, mark: Mark, toMove: Mark): number {
  const end = outcome(board);
  if (end.done) return end.draw ? 0 : end.winner === mark ? 1 : -1;
  const k = `${key(board)}|${mark}|${toMove}`;
  const cached = memo.get(k);
  if (cached !== undefined) return cached;
  const scores = legalMoves(board).map(m => value(play(board, m, toMove), mark, other(toMove)));
  const result = toMove === mark ? Math.max(...scores) : Math.min(...scores);
  memo.set(k, result);
  return result;
}

/** Every optimal move for the side to move, ascending. Ties are all returned, not broken. */
export function bestMoves(board: Board, toMove: Mark): number[] {
  const moves = legalMoves(board);
  const scored = moves.map(m => ({ m, v: value(play(board, m, toMove), toMove, other(toMove)) }));
  const best = Math.max(...scored.map(s => s.v));
  return scored.filter(s => s.v === best).map(s => s.m);
}

export type Opponent = (board: Board, toMove: Mark, random: () => number) => number;

export const randomOpponent: Opponent = (board, _toMove, random) => pick(legalMoves(board), random);

export const perfectOpponent: Opponent = (board, toMove, random) => pick(bestMoves(board, toMove), random);

/** Plays optimally except with probability `epsilon`, where it plays uniformly at random. */
export const noisyPerfect = (epsilon: number): Opponent => (board, toMove, random) =>
  random() < epsilon ? randomOpponent(board, toMove, random) : perfectOpponent(board, toMove, random);

/**
 * The training pool. A controller trained only against random play learns nothing about
 * defence; trained only against perfect play it sees almost no variety, because perfect
 * play collapses the game tree. Mixing is what produces a controller that is both safe
 * and not merely memorizing one line.
 */
export const TRAINING_POOL: readonly { name: string; opponent: Opponent }[] = Object.freeze([
  { name: 'random', opponent: randomOpponent },
  { name: 'noisy-0.5', opponent: noisyPerfect(0.5) },
  { name: 'noisy-0.2', opponent: noisyPerfect(0.2) },
  { name: 'perfect', opponent: perfectOpponent },
]);
