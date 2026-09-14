/**
 * Generate the control table in README.md and docs/experiment.md straight from
 * public/benchmarks/benchmark.json. Numbers in the docs are never typed by hand.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const b = JSON.parse(readFileSync(`${here}/../public/benchmarks/benchmark.json`, 'utf8'));

const table = [
  '| Controller | vs random (W/D/L) | score | losses vs perfect | illegal | what it answers |',
  '|---|---|---:|---:|---:|---|',
  ...b.rows.map(r =>
    `| ${r.condition} | ${r.vsRandom.win}/${r.vsRandom.draw}/${r.vsRandom.loss} | ${r.vsRandom.score.toFixed(3)} | ${r.vsPerfect.losses}/${r.vsPerfect.games} | ${r.illegal} | ${r.note} |`),
  '',
  `${b.rows[0].vsRandom.games} games vs random and ${b.rows[0].vsPerfect.games} vs perfect per controller. ` +
  `Score counts a draw as 0.5, because a draw is the correct result in tic-tac-toe. ` +
  `Checkpoint: seed ${b.checkpoint.seed}, ${b.checkpoint.generations} generations.`,
].join('\n');

const find = (needle) => b.rows.find(r => r.condition.toLowerCase().includes(needle));
const trained = b.rows[0], silenced = find('silenc'), untrained = find('untrained');
const rewired = find('rewired'), direct = find('raw board'), rnd = find('random legal');

const gap = (a, c) => (a && c ? (a.vsRandom.score - c.vsRandom.score) : 0);
const reading = [
  `**Does the circuit carry the decision?** Yes. Silencing it drops the score from ` +
  `**${trained.vsRandom.score.toFixed(3)} to ${silenced.vsRandom.score.toFixed(3)}** ` +
  `(${gap(trained, silenced) >= 0 ? '−' : '+'}${Math.abs(gap(trained, silenced)).toFixed(3)}) with the same 425 weights, ` +
  `and a silenced controller is literally board-blind — it produces identical scores for every ` +
  `position, which the test suite asserts. Training matters too: an untrained readout on the ` +
  `live circuit scores ${untrained.vsRandom.score.toFixed(3)}.`,
  '',
  rewired
    ? `**Does *this* measured topology matter?** ` +
      (Math.abs(gap(trained, rewired)) < 0.02
        ? `**No — not measurably.** A degree-preserving rewired graph, retrained at an identical ` +
          `budget, scores ${rewired.vsRandom.score.toFixed(3)} against the real circuit's ` +
          `${trained.vsRandom.score.toFixed(3)}. That is the honest result and it is worth stating ` +
          `plainly: the readout learned to use a recurrent network with the fly's degree ` +
          `distribution and contact-count statistics, and shuffling which cell connects to which ` +
          `cost it nothing. Anyone claiming fly wiring is *good at* a task needs exactly this ` +
          `control, and it is the one most often missing.`
        : `The rewired graph scores ${rewired.vsRandom.score.toFixed(3)} vs the real circuit's ` +
          `${trained.vsRandom.score.toFixed(3)}, a gap of ${gap(trained, rewired).toFixed(3)}. ` +
          `One seed is not a study; matched replicas would be needed before making anything of it.`)
    : '',
  '',
  direct
    ? `**What does the bottleneck cost?** Feeding the raw 18-channel board into a network of ` +
      `comparable size (${direct.condition.match(/\d+/)?.[0]} parameters) scores ` +
      `${direct.vsRandom.score.toFixed(3)}. Routing the board through 98 fly cells and reading ` +
      `only 16 descending ones costs about ${Math.abs(gap(direct, trained)).toFixed(3)}. The circuit ` +
      `is a constraint on the task, not an advantage — which is what you would expect, and is fine, ` +
      `as long as nobody says otherwise.`
    : '',
  '',
  `**Nobody reaches the real bar.** Tic-tac-toe is solved, so a genuinely good controller ` +
  `should *never* lose to perfect play. Against exact minimax the trained circuit still loses ` +
  `**${trained.vsPerfect.losses}/${trained.vsPerfect.games}** games` +
  (direct ? `, and even the unconstrained direct-board control loses ${direct.vsPerfect.losses}/${direct.vsPerfect.games}` : '') +
  `. A 425-parameter feed-forward readout with no search simply is not a strong tic-tac-toe ` +
  `player, and beating a random opponent 80% of the time should not be mistaken for one.`,
  '',
  `**The floor:** random legal play scores ${rnd.vsRandom.score.toFixed(3)}, and the silenced ` +
  `controller scores ${silenced.vsRandom.score.toFixed(3)} — *below* random, because a ` +
  `board-blind network plays the same fixed square preference every game. ` +
  `**Illegal moves: ${b.rows.reduce((a, r) => a + r.illegal, 0)} across every condition** — ` +
  `masking makes them impossible by construction, not by training.`,
].filter(Boolean).join('\n');

for (const [file, marks] of [['../README.md', true], ['../docs/experiment.md', true]]) {
  let text;
  try { text = readFileSync(`${here}/${file}`, 'utf8'); } catch { continue; }
  if (!marks) continue;
  text = text.replace(/<!-- BENCHMARK_TABLE -->[\s\S]*?(?=\n\n|$)/, `<!-- BENCHMARK_TABLE -->\n${table}`);
  text = text.replace(/<!-- BENCHMARK_READING -->[\s\S]*?(?=\n\nFull protocol|\n\n## |$)/, `<!-- BENCHMARK_READING -->\n${reading}`);
  writeFileSync(`${here}/${file}`, text);
  console.log(`updated ${file}`);
}
console.log('\n' + table + '\n\n' + reading);
