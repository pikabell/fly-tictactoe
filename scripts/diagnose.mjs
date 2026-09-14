/** Checkpoint 2: look at real activity before trusting any training run. */
import { readFileSync } from 'node:fs';
import { buildCircuit } from '../src/flybrain/circuit.ts';
import { Controller } from '../src/flybrain/controller.ts';
import { DEFAULT_DYNAMICS } from '../src/flybrain/dynamics.ts';
import { randomReadout } from '../src/flybrain/readout.ts';
import { EMPTY_BOARD, legalMoves, outcome, play, turn } from '../src/game/rules.ts';
import { randomOpponent, rng } from '../src/game/opponents.ts';

const doc = JSON.parse(readFileSync(new URL('../public/data/circuit.json', import.meta.url), 'utf8'));
const circuit = buildCircuit(doc);
console.log(`circuit: ${circuit.n} cells, ${doc.edges.length} edges`);
const live = circuit.incoming.reduce((a, b) => a + b.from.length, 0);
console.log(`edges with non-zero sign (actually carrying drive): ${live} of ${doc.edges.length}`);
console.log(`dynamics: ${JSON.stringify(DEFAULT_DYNAMICS)}\n`);

const theta = randomReadout(1);
const ctl = new Controller(circuit, theta);

// --- does activity respond to the board at all?
ctl.newGame();
const t1 = ctl.move(EMPTY_BOARD, 'X');
const hEmpty = Float64Array.from(ctl.activity);
ctl.newGame();
const b2 = ['X', 'O', 'X', null, 'O', null, null, null, null];
const t2 = ctl.move(b2, 'X');
const hFull = Float64Array.from(ctl.activity);
let diff = 0, maxDiff = 0;
for (let i = 0; i < hEmpty.length; i++) { const d = Math.abs(hEmpty[i] - hFull[i]); diff += d; maxDiff = Math.max(maxDiff, d); }
console.log(`sensitivity: mean |dh| between two boards = ${(diff / hEmpty.length).toFixed(4)}, max = ${maxDiff.toFixed(4)}`);

const stat = (a, label) => {
  const abs = Array.from(a, Math.abs);
  const sat = abs.filter(v => v > 0.99).length;
  const dead = abs.filter(v => v < 1e-6).length;
  console.log(`  ${label}: mean|h| ${(abs.reduce((x, y) => x + y, 0) / a.length).toFixed(4)}  max ${Math.max(...abs).toFixed(4)}  saturated(>0.99) ${sat}/${a.length}  dead(<1e-6) ${dead}/${a.length}`);
};
console.log('activity distribution:');
stat(hEmpty, 'empty board  ');
stat(hFull, 'mid-game board');
stat(t2.outputs, 'output cells (x4)');
console.log(`  raw scores: [${Array.from(t2.scores, v => v.toFixed(2)).join(', ')}]`);

// --- per-channel MARGINAL influence: flip one channel 0->1 against an otherwise empty
// board and measure how far the 16 descending activities move. This isolates the channel;
// an earlier version drove all 18 channels at once and measured almost nothing.
console.log('\nper-channel marginal influence on the 16 descending cells (flip one channel):');
const baseCtl = new Controller(circuit, theta);
baseCtl.newGame();
const baseOut = Float64Array.from(baseCtl.move(EMPTY_BOARD, 'X').outputs);
const rows = [];
for (let ch = 0; ch < 18; ch++) {
  const c2 = new Controller(circuit, theta);
  c2.newGame();
  const board = Array(9).fill(null);
  board[ch % 9] = ch < 9 ? 'X' : 'O';
  const t = c2.move(board, 'X');
  let mag = 0;
  for (let i = 0; i < 16; i++) mag += Math.abs(t.outputs[i] - baseOut[i]);
  rows.push({ ch, type: doc.cells[doc.inputCells[ch]].type, mag: mag / 16 });
}
rows.sort((a, b) => b.mag - a.mag);
for (const r of rows) console.log(`  ch${String(r.ch).padStart(2)} ${r.type.padEnd(7)} mean|delta out| ${r.mag.toFixed(4)}`);
const mags = rows.map(r => r.mag);
console.log(`  ratio loudest:quietest = ${(mags[0] / mags[mags.length - 1]).toFixed(1)}:1  (direct-contact ratio was 247:1)`);

// --- 100 untrained games: legality and completion
let illegal = 0, games = 0, moves = 0;
const results = { win: 0, draw: 0, loss: 0 };
for (let seed = 1; seed <= 100; seed++) {
  const random = rng(seed);
  ctl.newGame();
  let board = EMPTY_BOARD;
  const me = seed % 2 ? 'X' : 'O';
  while (!outcome(board).done) {
    const mark = turn(board);
    let m;
    if (mark === me) { m = ctl.move(board, me).move; if (!legalMoves(board).includes(m)) illegal++; }
    else m = randomOpponent(board, mark, random);
    board = play(board, m, mark);
    moves++;
  }
  const end = outcome(board);
  results[end.draw ? 'draw' : end.winner === me ? 'win' : 'loss']++;
  games++;
}
console.log(`\n100 untrained games vs random: ${JSON.stringify(results)}  illegal moves: ${illegal}  mean moves/game ${(moves / games).toFixed(1)}`);
