import type { Board as BoardType, Mark } from '../game/rules.ts';

/** The environment. Click a square to play; the circuit answers. */
export function Board({ board, onPlay, disabled, lastMove, winningLine }: {
  board: BoardType;
  onPlay: (square: number) => void;
  disabled: boolean;
  lastMove: number | null;
  winningLine: readonly number[] | null;
}) {
  return <div className="ttt-board" role="grid" aria-label="Tic-tac-toe board">
    {board.map((cell, i) => {
      const won = winningLine?.includes(i) ?? false;
      return <button
        key={i}
        role="gridcell"
        className={`ttt-cell${cell ? ` mark-${cell}` : ''}${won ? ' won' : ''}${lastMove === i ? ' last' : ''}`}
        disabled={disabled || cell !== null}
        aria-label={`Square ${i + 1}${cell ? `, ${cell}` : ', empty'}`}
        onClick={() => onPlay(i)}
      >{cell ?? ''}</button>;
    })}
  </div>;
}

/** The 18 board channels, as the circuit receives them. */
export function Channels({ features, types }: { features: Float64Array | null; types: (string | null)[] }) {
  return <div className="channels">
    {['mine', 'theirs'].map((group, g) => <div key={group} className="channel-group">
      <span className="channel-label">{group}</span>
      <div className="channel-row">
        {Array.from({ length: 9 }, (_, i) => {
          const ch = g * 9 + i;
          const on = (features?.[ch] ?? 0) > 0.5;
          return <span key={ch} className={`channel${on ? ' on' : ''}`} title={`ch${ch} · ${types[ch] ?? '?'} · square ${i + 1} · drive ${on ? '+1' : '-1'}`}>
            {types[ch] ?? '?'}
          </span>;
        })}
      </div>
    </div>)}
  </div>;
}

/** Descending-cell activities and the resulting square scores. The decision, in the open. */
export function Decision({ outputs, scores, board, outputTypes, move }: {
  outputs: Float64Array | null;
  scores: Float64Array | null;
  board: BoardType;
  outputTypes: (string | null)[];
  move: number | null;
}) {
  const maxOut = outputs ? Math.max(1e-6, ...Array.from(outputs, Math.abs)) : 1;
  const legalScores = scores ? board.map((c, i) => (c === null ? scores[i] : null)) : null;
  const finite = legalScores?.filter((v): v is number => v !== null) ?? [];
  const lo = finite.length ? Math.min(...finite) : 0, hi = finite.length ? Math.max(...finite) : 1;
  return <div className="decision">
    <div className="decision-block">
      <h3>16 descending cells <span>activity × 4</span></h3>
      <div className="bars">
        {Array.from({ length: 16 }, (_, i) => {
          const v = outputs?.[i] ?? 0;
          return <div key={i} className="bar-row" title={`${outputTypes[i] ?? '?'} · ${v.toFixed(3)}`}>
            <span className="bar-name">{outputTypes[i] ?? '?'}</span>
            <span className="bar-track">
              <span className={`bar-fill${v < 0 ? ' neg' : ''}`} style={{ width: `${(Math.abs(v) / maxOut) * 50}%`, [v < 0 ? 'right' : 'left']: '50%' }} />
            </span>
          </div>;
        })}
      </div>
    </div>
    <div className="decision-block">
      <h3>9 square scores <span>illegal squares masked</span></h3>
      <div className="score-grid">
        {board.map((cell, i) => {
          const v = legalScores?.[i];
          const t = v === null || v === undefined ? 0 : hi > lo ? (v - lo) / (hi - lo) : 1;
          return <span key={i} className={`score-cell${cell !== null ? ' masked' : ''}${move === i ? ' chosen' : ''}`}
            style={cell === null ? { background: `color-mix(in oklab, var(--accent) ${(t * 100).toFixed(0)}%, transparent)` } : undefined}
            title={cell !== null ? 'occupied — masked out' : `score ${v?.toFixed(3)}`}>
            {cell !== null ? '·' : v?.toFixed(1)}
          </span>;
        })}
      </div>
    </div>
  </div>;
}
