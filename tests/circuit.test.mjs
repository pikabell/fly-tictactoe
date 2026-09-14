import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildCircuit, rewire, signOf } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { N_PARAMS, randomReadout } from '../src/flybrain/readout.ts';
import { EMPTY_BOARD, legalMoves, outcome, play, turn } from '../src/game/rules.ts';
import { randomOpponent, rng } from '../src/game/opponents.ts';
import { runCem } from '../src/training/cem.ts';

const doc = JSON.parse(readFileSync(new URL('../public/data/circuit.json', import.meta.url), 'utf8'));
const circuit = buildCircuit(doc);

test('circuit.json matches its own declared counts', () => {
  assert.equal(doc.cells.length, doc.counts.cells);
  assert.equal(doc.edges.length, doc.counts.edges);
  assert.equal(doc.edges.reduce((a, e) => a + e.contacts, 0), doc.counts.contacts);
  assert.equal(doc.inputCells.length, 18);
  assert.equal(doc.outputCells.length, 16);
  assert.equal(doc.dataset, 'male-cns:v1.0');
});

test('no edge is synthesized: every edge has at least one measured contact', () => {
  for (const e of doc.edges) assert.ok(e.contacts >= 1, 'contacts must be >= 1');
});

test('sign policy is explicit and total', () => {
  assert.equal(signOf('acetylcholine'), 1);
  assert.equal(signOf('gaba'), -1);
  assert.equal(signOf('glutamate'), -1);
  assert.equal(signOf('unclear'), 0);
  assert.equal(signOf('anything-else'), 0);
});

test('every output cell is reachable from an input through sign-carrying edges', () => {
  const adj = new Map();
  for (const e of doc.edges) {
    if (signOf(doc.cells[e.pre].nt) === 0) continue;
    if (!adj.has(e.pre)) adj.set(e.pre, []);
    adj.get(e.pre).push(e.post);
  }
  const seen = new Set(), stack = [...doc.inputCells];
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    for (const m of adj.get(n) ?? []) stack.push(m);
  }
  for (const o of doc.outputCells) assert.ok(seen.has(o), `output ${o} unreachable`);
});

test('activity responds to the board and neither saturates nor dies', () => {
  const ctl = new Controller(circuit, randomReadout(1));
  ctl.newGame();
  ctl.move(EMPTY_BOARD, 'X');
  const a = Float64Array.from(ctl.activity);
  ctl.newGame();
  ctl.move(['X', 'O', 'X', null, 'O', null, null, null, null], 'X');
  const b = Float64Array.from(ctl.activity);
  let moved = 0;
  for (let i = 0; i < a.length; i++) moved += Math.abs(a[i] - b[i]);
  assert.ok(moved / a.length > 0.01, 'activity barely changed between two boards');
  for (const v of b) {
    assert.ok(Number.isFinite(v), 'non-finite activity');
    assert.ok(Math.abs(v) < 0.999, 'a cell saturated');
  }
});

test('silencing the circuit changes the moves it plays', () => {
  const theta = randomReadout(4);
  const live = new Controller(circuit, theta, false);
  const dead = new Controller(circuit, theta, true);
  live.newGame(); dead.newGame();
  let different = 0;
  let board = EMPTY_BOARD;
  for (let i = 0; i < 4; i++) {
    if (live.move(board, 'X').move !== dead.move(board, 'X').move) different++;
    board = play(board, legalMoves(board)[0], turn(board));
    board = play(board, legalMoves(board).at(-1), turn(board));
  }
  assert.ok(different > 0, 'silenced controller played identically — the circuit is decorative');
});

test('a silenced controller is board-blind: identical outputs for every board', () => {
  const dead = new Controller(circuit, randomReadout(5), true);
  dead.newGame();
  const a = Array.from(dead.move(EMPTY_BOARD, 'X').scores);
  const b = Array.from(dead.move(['X', null, 'O', null, 'X', null, null, null, null], 'X').scores);
  assert.deepEqual(a, b, 'silenced controller saw the board through some other path');
});

test('no illegal move across 200 games, trained or not', () => {
  for (const theta of [randomReadout(2), randomReadout(3)]) {
    const ctl = new Controller(circuit, theta);
    for (let seed = 1; seed <= 100; seed++) {
      const random = rng(seed);
      ctl.newGame();
      let board = EMPTY_BOARD;
      const me = seed % 2 ? 'X' : 'O';
      while (!outcome(board).done) {
        const mark = turn(board);
        const m = mark === me ? ctl.move(board, me).move : randomOpponent(board, mark, random);
        assert.ok(legalMoves(board).includes(m), `illegal move ${m}`);
        board = play(board, m, mark);
      }
    }
  }
});

test('rewiring preserves edge count, contacts and in-degree, and changes targets', () => {
  const r = rewire(doc, 7777);
  assert.equal(r.edges.length, doc.edges.length);
  assert.equal(r.edges.reduce((a, e) => a + e.contacts, 0), doc.counts.contacts);
  const deg = (d) => { const m = new Map(); for (const e of d.edges) m.set(e.pre, (m.get(e.pre) ?? 0) + 1); return m; };
  assert.deepEqual([...deg(r)].sort(), [...deg(doc)].sort(), 'out-degree changed');
  assert.ok(r.edges.some((e, i) => e.post !== doc.edges[i].post), 'rewiring changed nothing');
});

test('CEM is deterministic for a seed', () => {
  const cfg = { candidates: 8, elites: 2, generations: 3, initSigma: 0.5, sigmaFloor: 0.07, blend: 0.3 };
  const fit = (theta) => { let s = 0; for (let i = 0; i < N_PARAMS; i += 37) s -= (theta[i] - 0.25) ** 2; return s; };
  const a = runCem(99, cfg, { train: fit, validate: fit });
  const b = runCem(99, cfg, { train: fit, validate: fit });
  assert.deepEqual(Array.from(a.champion), Array.from(b.champion));
  assert.deepEqual(a.history, b.history);
});

test('game flow: marks strictly alternate whichever side the circuit takes', () => {
  // Regression guard. The UI once started a circuit-first game with a stale mark, so the
  // circuit played O and the human was then also assigned O. `flyMove` now refuses to move
  // when it is not that mark's turn; this asserts the invariant it protects.
  for (const flyMark of ['X', 'O']) {
    const human = flyMark === 'X' ? 'O' : 'X';
    const ctl = new Controller(circuit, randomReadout(11));
    ctl.newGame();
    let board = EMPTY_BOARD;
    const played = [];
    while (!outcome(board).done) {
      const mark = turn(board);
      assert.equal(mark, played.length % 2 === 0 ? 'X' : 'O', 'X must move on even plies');
      const m = mark === flyMark ? ctl.move(board, flyMark).move : legalMoves(board)[0];
      board = play(board, m, mark);
      played.push(mark);
    }
    assert.deepEqual(played, played.map((_, i) => (i % 2 === 0 ? 'X' : 'O')));
    assert.ok(played.filter(m => m === flyMark).length >= 2, 'the circuit must actually have moved');
    assert.ok(played.filter(m => m === human).length >= 2, 'the human must actually have moved');
  }
});
