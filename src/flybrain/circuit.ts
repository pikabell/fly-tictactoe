/**
 * The measured part of this project.
 *
 * `circuit.json` is a MaleCNS v1.0 subgraph: which cells connect to which, and how many
 * synaptic contacts each connection has. That is the connectome. Everything this file does
 * on top of it - turning a predicted neurotransmitter into a +1/-1 sign, normalizing weights
 * per postsynaptic cell - is a MODELLING ASSUMPTION, stated here rather than hidden.
 */

export type CircuitCell = {
  bodyId: number;
  type: string | null;
  superclass: string | null;
  nt: string;
  soma: [number, number, number];
  role: 'input' | 'bridge' | 'output';
};

export type CircuitEdge = { pre: number; post: number; contacts: number };

export type CircuitDoc = {
  version: 1;
  dataset: 'male-cns:v1.0';
  selection: Record<string, unknown>;
  signPolicy: Record<string, unknown>;
  counts: Record<string, unknown>;
  source: { reference: string; license: string; attribution: string; sha256: Record<string, string> };
  inputCells: number[];
  outputCells: number[];
  cells: CircuitCell[];
  edges: CircuitEdge[];
};

/** Predicted transmitter -> assumed presynaptic sign. Not a measurement of synaptic sign. */
export function signOf(nt: string): number {
  if (nt === 'acetylcholine') return 1;
  if (nt === 'gaba' || nt === 'glutamate') return -1;
  return 0; // unclear / modulatory: keeps its anatomy, contributes no modelled drive
}

export type Circuit = {
  doc: CircuitDoc;
  n: number;
  /** Column-major CSC-ish: for each postsynaptic cell i, the (j, w) pairs driving it. */
  incoming: { from: Int32Array; weight: Float64Array }[];
  inputCells: Int32Array;
  outputCells: Int32Array;
  bodyIds: Uint32Array;
  signs: Int8Array;
};

/**
 * Signed weights normalized by total absolute signed contact count at each POSTsynaptic cell:
 *
 *   W[j][i] = c[j][i] * s[j] / sum_k( c[k][i] * |s[k]| )
 *
 * so every cell receives drive on a comparable scale regardless of how many contacts it has.
 * A cell whose presynaptic partners are all sign-0 gets zero drive rather than a divide by zero.
 */
export function buildCircuit(doc: CircuitDoc): Circuit {
  const n = doc.cells.length;
  // Fail loudly if a circuit.json is rebuilt with different dimensions than the encoder and
  // readout compiled against, rather than indexing past the end of an array at move time.
  if (doc.inputCells.length !== 18) throw Error(`circuit has ${doc.inputCells.length} input cells, encoder expects 18`);
  if (doc.outputCells.length !== 16) throw Error(`circuit has ${doc.outputCells.length} output cells, readout expects 16`);
  for (const i of [...doc.inputCells, ...doc.outputCells]) {
    if (!Number.isInteger(i) || i < 0 || i >= n) throw Error(`cell index ${i} out of range for ${n} cells`);
  }
  const signs = new Int8Array(n);
  for (let i = 0; i < n; i++) signs[i] = signOf(doc.cells[i].nt);

  const denom = new Float64Array(n);
  for (const e of doc.edges) denom[e.post] += e.contacts * Math.abs(signs[e.pre]);

  const bucket: { from: number[]; weight: number[] }[] = Array.from({ length: n }, () => ({ from: [], weight: [] }));
  for (const e of doc.edges) {
    const s = signs[e.pre];
    if (s === 0) continue;              // anatomy retained, drive is zero
    const d = denom[e.post];
    if (d === 0) continue;
    bucket[e.post].from.push(e.pre);
    bucket[e.post].weight.push((e.contacts * s) / d);
  }

  return {
    doc,
    n,
    incoming: bucket.map(b => ({ from: Int32Array.from(b.from), weight: Float64Array.from(b.weight) })),
    inputCells: Int32Array.from(doc.inputCells),
    outputCells: Int32Array.from(doc.outputCells),
    bodyIds: Uint32Array.from(doc.cells.map(c => c.bodyId)),
    signs,
  };
}

/**
 * Degree-preserving rewiring control. Keeps every cell's in-degree, out-degree and the
 * multiset of contact counts, and destroys only WHICH cell connects to which. If a
 * controller trained on this does as well as one trained on the real circuit, then the
 * measured topology contributed nothing and only the degree statistics mattered.
 */
export function rewire(doc: CircuitDoc, seed: number): CircuitDoc {
  let a = seed >>> 0;
  const rnd = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const posts = doc.edges.map(e => e.post);
  for (let i = posts.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [posts[i], posts[j]] = [posts[j], posts[i]];
  }
  return { ...doc, edges: doc.edges.map((e, i) => ({ ...e, post: posts[i] })) };
}
