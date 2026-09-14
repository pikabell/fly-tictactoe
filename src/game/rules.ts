/**
 * Tic-tac-toe rules. Deliberately boring and exact: the whole point of choosing
 * this game is that correctness here is not in question, so any interesting
 * behaviour must be coming from the circuit.
 *
 * Board is 9 cells, row-major:  0 1 2
 *                               3 4 5
 *                               6 7 8
 */

export type Mark = 'X' | 'O';
export type Cell = Mark | null;
export type Board = readonly Cell[]; // length 9

export const EMPTY_BOARD: Board = Object.freeze(Array<Cell>(9).fill(null));

export const LINES: readonly (readonly [number, number, number])[] = Object.freeze([
  [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
  [0, 4, 8], [2, 4, 6],            // diagonals
]);

export const other = (mark: Mark): Mark => (mark === 'X' ? 'O' : 'X');

/** Whose turn it is, derived from the board rather than tracked separately. */
export function turn(board: Board): Mark {
  let x = 0, o = 0;
  for (const c of board) { if (c === 'X') x++; else if (c === 'O') o++; }
  return x === o ? 'X' : 'O';
}

export function legalMoves(board: Board): number[] {
  const moves: number[] = [];
  for (let i = 0; i < 9; i++) if (board[i] === null) moves.push(i);
  return moves;
}

/** The winning mark, or null. Does not say whether the game is over. */
export function winner(board: Board): Mark | null {
  for (const [a, b, c] of LINES) {
    const v = board[a];
    if (v !== null && v === board[b] && v === board[c]) return v;
  }
  return null;
}

export type Outcome = { done: boolean; winner: Mark | null; draw: boolean };

export function outcome(board: Board): Outcome {
  const w = winner(board);
  if (w) return { done: true, winner: w, draw: false };
  const full = board.every(c => c !== null);
  return { done: full, winner: null, draw: full };
}

export function play(board: Board, square: number, mark: Mark): Board {
  if (square < 0 || square > 8 || !Number.isInteger(square)) throw Error(`Square out of range: ${square}`);
  if (board[square] !== null) throw Error(`Square ${square} is already ${board[square]}`);
  const next = board.slice();
  next[square] = mark;
  return next;
}

export const render = (board: Board): string =>
  [0, 3, 6].map(r => [0, 1, 2].map(c => board[r + c] ?? '.').join(' ')).join('\n');

/** Compact key for memoization and for the 5,478-position sweep. */
export const key = (board: Board): string => board.map(c => c ?? '.').join('');
