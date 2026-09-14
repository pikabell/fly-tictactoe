import test from 'node:test';
import assert from 'node:assert/strict';
import { EMPTY_BOARD, legalMoves, outcome, play, turn, winner, key } from '../src/game/rules.ts';
import { bestMoves, perfectOpponent, randomOpponent, rng } from '../src/game/opponents.ts';

test('turn alternates and is derived from the board', () => {
  assert.equal(turn(EMPTY_BOARD), 'X');
  assert.equal(turn(play(EMPTY_BOARD, 4, 'X')), 'O');
});

test('winner detects all 8 lines', () => {
  const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (const line of lines) {
    const b = Array(9).fill(null);
    for (const i of line) b[i] = 'X';
    assert.equal(winner(b), 'X', `line ${line}`);
  }
});

test('playing an occupied square throws', () => {
  const b = play(EMPTY_BOARD, 0, 'X');
  assert.throws(() => play(b, 0, 'O'));
});

test('the reachable game tree has exactly 5,478 positions', () => {
  const seen = new Set();
  const walk = (board) => {
    const k = key(board);
    if (seen.has(k)) return;
    seen.add(k);
    if (outcome(board).done) return;
    const mark = turn(board);
    for (const m of legalMoves(board)) walk(play(board, m, mark));
  };
  walk(EMPTY_BOARD);
  assert.equal(seen.size, 5478);
});

test('perfect play always draws — the bar every controller is measured against', () => {
  for (let seed = 1; seed <= 50; seed++) {
    const random = rng(seed);
    let board = EMPTY_BOARD;
    while (!outcome(board).done) board = play(board, perfectOpponent(board, turn(board), random), turn(board));
    assert.equal(outcome(board).draw, true, `seed ${seed} did not draw`);
  }
});

test('perfect play never loses to random play', () => {
  for (let seed = 1; seed <= 200; seed++) {
    const random = rng(seed);
    for (const perfectMark of ['X', 'O']) {
      let board = EMPTY_BOARD;
      while (!outcome(board).done) {
        const mark = turn(board);
        const move = mark === perfectMark ? perfectOpponent(board, mark, random) : randomOpponent(board, mark, random);
        board = play(board, move, mark);
      }
      const end = outcome(board);
      assert.notEqual(end.winner, perfectMark === 'X' ? 'O' : 'X', `perfect lost, seed ${seed}`);
    }
  }
});

test('a fork is found: X at 0 and 4, O at 8 -> X must have a winning move', () => {
  const b = ['X', null, null, null, 'X', null, null, null, 'O'];
  const moves = bestMoves(b, 'X');
  assert.ok(moves.length > 0);
});

test('an immediate win is taken', () => {
  const b = ['X', 'X', null, 'O', 'O', null, null, null, null];
  assert.deepEqual(bestMoves(b, 'X'), [2]);
});

test('an immediate loss is blocked', () => {
  // Legal position: two marks each, X to move, O threatens the top row at square 2.
  const b = ['O', 'O', null, null, null, 'X', null, 'X', null];
  assert.deepEqual(bestMoves(b, 'X'), [2]);
});
