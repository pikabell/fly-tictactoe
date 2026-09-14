/**
 * "Has it learnt anything?" — answered exactly, because tic-tac-toe is solved.
 *
 * Three checks, over every reachable position:
 *   1. When a move wins immediately, does it take it?
 *   2. When the opponent threatens to win next move, does it block?
 *   3. Over the whole game tree, how often does it play a move minimax calls optimal?
 *
 * Caveat stated up front: the controller is trained with circuit state persisting across
 * moves, so scoring isolated positions with a fresh state is slightly off-distribution.
 * Both are reported.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCircuit, rewire } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { randomReadout } from '../src/flybrain/readout.ts';
import { EMPTY_BOARD, key, legalMoves, other, outcome, play, turn, winner } from '../src/game/rules.ts';
import { bestMoves, perfectOpponent, rng } from '../src/game/opponents.ts';
import { DirectController } from '../src/training/baselines.ts';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => (existsSync(`${here}/${p}`) ? JSON.parse(readFileSync(`${here}/${p}`, 'utf8')) : null);
const doc = read('../public/data/circuit.json');
const circuit = buildCircuit(doc);
const champ = read('../public/checkpoints/champion.json');
const rw = read('../public/checkpoints/champion-rewired.json');
const dir = read('../public/checkpoints/champion-direct.json');
const imi = read('../public/checkpoints/champion-imitation.json');
const imiRw = read('../public/checkpoints/champion-imitation-rewired.json');

// --- enumerate every reachable non-terminal position
const positions = [];
const seen = new Set();
(function walk(board) {
  const k = key(board);
  if (seen.has(k)) return;
  seen.add(k);
  if (outcome(board).done) return;
  positions.push(board);
  const m = turn(board);
  for (const sq of legalMoves(board)) walk(play(board, sq, m));
})(EMPTY_BOARD);

const winsNow = (board, mark) => legalMoves(board).filter(sq => winner(play(board, sq, mark)) === mark);

const wins = [], blocks = [];
for (const b of positions) {
  const m = turn(b), opp = other(m);
  const mine = winsNow(b, m);
  if (mine.length) { wins.push({ b, m, good: mine }); continue; }
  const theirs = winsNow(b, opp);
  if (theirs.length === 1) blocks.push({ b, m, good: theirs });
}

function assess(name, make) {
  const ctl = make();
  const rate = (set) => {
    let ok = 0;
    for (const { b, m, good } of set) { ctl.newGame(); if (good.includes(ctl.move(b, m).move)) ok++; }
    return ok / set.length;
  };
  let opt = 0;
  for (const b of positions) {
    const m = turn(b);
    ctl.newGame();
    if (bestMoves(b, m).includes(ctl.move(b, m).move)) opt++;
  }
  // full games vs perfect, split by side, state persisting as in real play
  const vsPerfect = { X: { w: 0, d: 0, l: 0 }, O: { w: 0, d: 0, l: 0 } };
  for (const me of ['X', 'O']) {
    for (let seed = 1; seed <= 100; seed++) {
      const random = rng(seed);
      ctl.newGame();
      let board = EMPTY_BOARD;
      while (!outcome(board).done) {
        const mark = turn(board);
        board = play(board, mark === me ? ctl.move(board, me).move : perfectOpponent(board, mark, random), mark);
      }
      const e = outcome(board);
      vsPerfect[me][e.draw ? 'd' : e.winner === me ? 'w' : 'l']++;
    }
  }
  return {
    name,
    win: rate(wins), block: rate(blocks), optimal: opt / positions.length,
    X: vsPerfect.X, O: vsPerfect.O,
  };
}

const rows = [];
if (imi) rows.push(assess('Circuit + IMITATION readout', () => new Controller(circuit, Float64Array.from(imi.theta))));
if (imiRw) rows.push(assess('Rewired + imitation readout', () => new Controller(buildCircuit(rewire(doc, 7777)), Float64Array.from(imiRw.theta))));
rows.push(
  assess('Circuit + CEM readout', () => new Controller(circuit, Float64Array.from(champ.theta))),
  assess('Circuit + UNTRAINED readout', () => new Controller(circuit, randomReadout(1))),
  assess('Same readout, circuit SILENCED', () => new Controller(circuit, Float64Array.from(imi ? imi.theta : champ.theta), true)),
);
if (rw) rows.push(assess('Rewired graph, CEM', () => new Controller(buildCircuit(rewire(doc, 7777)), Float64Array.from(rw.theta))));
if (dir) rows.push(assess('Raw board -> net, trained', () => new DirectController(Float64Array.from(dir.theta))));
rows.push(assess('Random legal move', () => {
  const r = rng(4242);   // seeded, so this baseline row is reproducible like every other
  return { newGame() {}, move(b) { const l = legalMoves(b); return { move: l[Math.floor(r() * l.length) % l.length] }; } };
}));

console.log(`\n${positions.length} reachable non-terminal positions · ${wins.length} with an immediate win available · ${blocks.length} needing a block\n`);
const pad = (s, n) => String(s).padEnd(n);
console.log(pad('controller', 34) + pad('takes win', 11) + pad('blocks', 9) + pad('optimal', 9) + pad('as X vs perfect', 17) + 'as O vs perfect');
console.log('-'.repeat(100));
for (const r of rows) {
  console.log(
    pad(r.name, 34) +
    pad((r.win * 100).toFixed(1) + '%', 11) +
    pad((r.block * 100).toFixed(1) + '%', 9) +
    pad((r.optimal * 100).toFixed(1) + '%', 9) +
    pad(`${r.X.w}W ${r.X.d}D ${r.X.l}L`, 17) +
    `${r.O.w}W ${r.O.d}D ${r.O.l}L`);
}
