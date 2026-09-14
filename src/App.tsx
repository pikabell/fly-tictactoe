import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { Attribution } from './components/Attribution';
import { Board, Channels, Decision } from './components/Board';
import { asset, loadAtlas, type Atlas } from './lib/atlas';
import type { ActivityFrame } from './lib/replay';
import { buildCircuit, type Circuit, type CircuitDoc } from './flybrain/circuit.ts';
import { Controller, type Trace } from './flybrain/controller.ts';
import { N_PARAMS, randomReadout } from './flybrain/readout.ts';
import { EMPTY_BOARD, LINES, type Board as BoardType, type Mark, legalMoves, other, outcome, play, turn, winner } from './game/rules.ts';
import { noisyPerfect, perfectOpponent, randomOpponent, rng } from './game/opponents.ts';
import type { TrainDone, TrainProgress } from './training/worker.ts';

type Level = 'random' | 'easy' | 'hard' | 'perfect';
const OPPONENTS: Record<Level, ReturnType<typeof noisyPerfect>> = {
  random: randomOpponent, easy: noisyPerfect(0.5), hard: noisyPerfect(0.2), perfect: perfectOpponent,
};

/** h is signed and dimensionless; the atlas shader wants [0,1]. We show MAGNITUDE of activity. */
const toFrame = (circuit: Circuit, h: Float64Array, visible: ReadonlySet<number>): ActivityFrame => ({
  time: 0,
  values: Array.from(circuit.bodyIds)
    .map((id, i) => [id, Math.min(1, Math.abs(h[i]))] as [number, number])
    .filter(([id]) => visible.has(id)),
});

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [doc, setDoc] = useState<CircuitDoc | null>(null);
  const [theta, setTheta] = useState<Float64Array | null>(null);
  const [checkpointName, setCheckpointName] = useState('loading…');
  const [error, setError] = useState('');

  const [board, setBoard] = useState<BoardType>(EMPTY_BOARD);
  const [me, setMe] = useState<Mark>('O');         // the circuit's mark; you are X by default
  const [level, setLevel] = useState<Level>('easy');
  const [silenced, setSilenced] = useState(false);
  const [resetEachMove, setResetEachMove] = useState(false);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [frame, setFrame] = useState<ActivityFrame | null>(null);
  const [status, setStatus] = useState('Your move.');
  const [tally, setTally] = useState({ you: 0, fly: 0, draw: 0 });

  const [training, setTraining] = useState<TrainProgress | null>(null);
  const worker = useRef<Worker | null>(null);
  const random = useRef(rng(Date.now() >>> 0));

  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch(e => { if (!abort.signal.aborted) setError(String(e)); });
    void fetch(asset('data/circuit.json'), { signal: abort.signal })
      .then(r => { if (!r.ok) throw Error('circuit.json unavailable'); return r.json(); })
      .then(setDoc).catch(e => { if (!abort.signal.aborted) setError(String(e)); });
    void fetch(asset('checkpoints/champion.json'), { signal: abort.signal })
      .then(r => (r.ok ? r.json() : null))
      .then(c => {
        if (c?.theta?.length === N_PARAMS) { setTheta(Float64Array.from(c.theta)); setCheckpointName(`trained · seed ${c.seed} · ${c.generations} generations`); }
        else { setTheta(randomReadout(1)); setCheckpointName('untrained (random weights)'); }
      })
      .catch(() => { setTheta(randomReadout(1)); setCheckpointName('untrained (random weights)'); });
    return () => abort.abort();
  }, []);

  const circuit = useMemo(() => (doc ? buildCircuit(doc) : null), [doc]);
  const controller = useMemo(
    () => (circuit && theta ? new Controller(circuit, theta, silenced) : null),
    [circuit, theta, silenced],
  );
  useEffect(() => { controller?.newGame(); }, [controller]);

  const inputTypes = useMemo(() => doc?.inputCells.map(i => doc.cells[i].type) ?? [], [doc]);
  const outputTypes = useMemo(() => doc?.outputCells.map(i => doc.cells[i].type) ?? [], [doc]);

  const flyMove = useCallback((current: BoardType) => {
    if (!controller || !circuit || !atlas) return;
    if (resetEachMove) controller.newGame();
    const t = controller.move(current, me);
    setTrace({ ...t, features: Float64Array.from(t.features), outputs: Float64Array.from(t.outputs), scores: Float64Array.from(t.scores) });
    setFrame(toFrame(circuit, controller.activity, atlas.visibleIds));
    const next = play(current, t.move, me);
    setBoard(next);
    const end = outcome(next);
    if (end.done) finish(end.winner);
    else setStatus('Your move.');
  }, [controller, circuit, atlas, me, resetEachMove]);

  const finish = (w: Mark | null) => {
    if (!w) { setStatus('Draw — which is the correct result in tic-tac-toe.'); setTally(t => ({ ...t, draw: t.draw + 1 })); }
    else if (w === me) { setStatus('The circuit wins.'); setTally(t => ({ ...t, fly: t.fly + 1 })); }
    else { setStatus('You win.'); setTally(t => ({ ...t, you: t.you + 1 })); }
  };

  const youPlay = (square: number) => {
    if (outcome(board).done || board[square] !== null) return;
    const youAre = other(me);
    if (turn(board) !== youAre) return;
    const next = play(board, square, youAre);
    setBoard(next);
    const end = outcome(next);
    if (end.done) { finish(end.winner); return; }
    setStatus('The circuit is deciding…');
    setTimeout(() => flyMove(next), 220);
  };

  const newGame = (flyMark: Mark = me) => {
    setMe(flyMark);
    setBoard(EMPTY_BOARD);
    setTrace(null); setFrame(null);
    controller?.newGame();
    if (flyMark === 'X') { setStatus('The circuit opens…'); setTimeout(() => flyMove(EMPTY_BOARD), 220); }
    else setStatus('Your move.');
  };

  // "Watch it play itself" — the circuit against a chosen opponent, for when you want to see
  // the controls (silencing, per-move reset) change behaviour without playing 30 games yourself.
  const [auto, setAuto] = useState(false);
  useEffect(() => {
    if (!auto || !controller) return;
    const end = outcome(board);
    if (end.done) { const t = setTimeout(() => newGame(me), 900); return () => clearTimeout(t); }
    const t = setTimeout(() => {
      if (turn(board) === me) flyMove(board);
      else {
        const next = play(board, OPPONENTS[level](board, turn(board), random.current), turn(board));
        setBoard(next);
        const e2 = outcome(next);
        if (e2.done) finish(e2.winner);
      }
    }, 320);
    return () => clearTimeout(t);
  }, [auto, board, controller, level, me, flyMove]);

  const startTraining = () => {
    if (!doc || training) return;
    const w = new Worker(new URL('./training/worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (event: MessageEvent<TrainProgress | TrainDone>) => {
      if (event.data.kind === 'progress') setTraining(event.data);
      else {
        setTheta(Float64Array.from(event.data.theta));
        setCheckpointName(`trained in this browser · validation ${event.data.championValidation.toFixed(3)} · ${event.data.seconds.toFixed(0)}s`);
        setTraining(null); w.terminate(); worker.current = null;
      }
    };
    w.postMessage({ kind: 'train', doc, seed: 20260914, generations: 60 });
  };
  const stopTraining = () => { worker.current?.postMessage({ kind: 'stop' }); };
  useEffect(() => () => worker.current?.terminate(), []);

  const line = useMemo(() => {
    const w = winner(board);
    return w ? (LINES.find(l => l.every(i => board[i] === w)) ?? null) : null;
  }, [board]);
  const counts = doc?.counts as Record<string, number> | undefined;
  const yourTurn = !outcome(board).done && turn(board) === other(me) && !auto;

  return <>
    <header>
      <h1>FLY TIC-TAC-TOE</h1>
      <span>A measured MaleCNS circuit picks the moves</span>
      <a href="https://github.com/cobanov/awesome-fly">awesome-fly ↗</a>
    </header>
    <main>
      <div className="toolbar">
        <span className="status">{status}</span>
        <div className="controls">
          <button onClick={() => newGame('O')}>New game (you first)</button>
          <button onClick={() => newGame('X')}>New game (circuit first)</button>
          <button aria-pressed={auto} onClick={() => setAuto(a => !a)}>{auto ? 'Stop autoplay' : 'Autoplay'}</button>
          <label className="select">Opponent
            <select value={level} onChange={e => setLevel(e.target.value as Level)}>
              <option value="random">random</option><option value="easy">easy</option>
              <option value="hard">hard</option><option value="perfect">perfect</option>
            </select>
          </label>
          <label className="check"><input type="checkbox" checked={silenced} onChange={e => setSilenced(e.target.checked)} /> Silence circuit</label>
          <label className="check"><input type="checkbox" checked={resetEachMove} onChange={e => setResetEachMove(e.target.checked)} /> Reset state each move</label>
        </div>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {silenced && <p className="warn">Circuit silenced: descending activity is held at zero, so the readout is deciding from nothing. Play should collapse.</p>}

      <div className="workbench">
        <section className="panel environment-panel">
          <h2>01 / GAME <span>you {tally.you} · circuit {tally.fly} · draws {tally.draw}</span></h2>
          <Board board={board} onPlay={youPlay} disabled={!yourTurn} lastMove={trace?.move ?? null} winningLine={line} />
          <Channels features={trace?.features ?? null} types={inputTypes} />
          <div className="panel-bottom">18 channels drive 18 visual-projection cells · assignment is arbitrary engineering</div>
        </section>

        <section className="panel brain-panel">
          <h2>02 / BRAIN <span>MaleCNS v1.0</span></h2>
          {atlas ? <BrainScene atlas={atlas} frame={frame} /> : <p className="loading" role="status">Loading measured anatomy…</p>}
          <div className="panel-bottom">
            {counts ? `${counts.cells} cells · ${counts.edges} edges · ${counts.contacts.toLocaleString('en-US')} contacts` : '…'} in {atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} somata
            <a href={asset('data/circuit.json')}>circuit.json ↗</a>
          </div>
        </section>

        <section className="panel fly-panel">
          <h2>03 / BODY <span>Flybody</span></h2>
          <FlyScene />
          <div className="panel-bottom">Anatomical mesh · not driven by this circuit <span>Drag to rotate</span></div>
        </section>
      </div>

      <section className="panel decision-panel">
        <h2>04 / THE DECISION <span>{checkpointName}</span></h2>
        <Decision outputs={trace?.outputs ?? null} scores={trace?.scores ?? null} board={board} outputTypes={outputTypes} move={trace?.move ?? null} />
        <div className="train">
          <button onClick={startTraining} disabled={!doc || !!training}>Train in this browser</button>
          {training && <>
            <button onClick={stopTraining}>Stop</button>
            <span className="train-status">gen {training.generation + 1}/{training.generations} · best {training.bestTraining.toFixed(3)} · champion {training.championValidation.toFixed(3)}</span>
            <span className="train-bar"><span style={{ width: `${((training.generation + 1) / training.generations) * 100}%` }} /></span>
          </>}
          {!training && <span className="train-status">425 readout parameters · cross-entropy method · the measured graph never changes</span>}
        </div>
      </section>

      <section className="model-status" aria-label="Model provenance">
        <strong>PREDICTED / SIMULATED OUTPUT</strong>
        <p>Cell activity shown here is computed by an assumed dynamical model, not measured from a fly. It is dimensionless — neither a firing rate nor a membrane voltage.</p>
        <p>Measured: which cells connect to which, and their synaptic contact counts (MaleCNS v1.0). Assumed: the board-to-cell encoder, the neurotransmitter sign policy, the leak/gain dynamics, and the trainable readout.</p>
      </section>

      <details>
        <summary>What is real here, and what is not</summary>
        <p>The circuit is {counts?.cells ?? '…'} cells selected from MaleCNS v1.0 by anatomy alone, before any training: 18 visual-projection cells (one per board channel), 16 descending cells read out by the network, and 64 bridge cells between them. Every measured directed edge among those cells is retained, including single-contact and recurrent ones. Nothing is synthesized.</p>
        <p>Assigning a board square to a cell type has no biological meaning. LC12 responds to small moving objects in a real fly; it has never seen a game board. Training moves only the 425 readout parameters — no biological synapse changes, and there is no plasticity anywhere in this model.</p>
        <p>Try it: tick <em>Silence circuit</em> and watch play collapse. That is the control which shows the circuit is actually carrying the decision, rather than decorating a readout that could have done the job alone.</p>
        <p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Atlas source and hashes</a>.</p>
      </details>
    </main>
    <Attribution />
  </>;
}
