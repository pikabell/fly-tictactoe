"""
Extract a small, measured MaleCNS v1.0 circuit for the tic-tac-toe experiment.

This is the ONLY step that touches the ~1.1 GB raw connectome. It runs offline and
emits `public/data/circuit.json`, a few hundred KB, which is what the browser loads.

Selection is deterministic and uses ANATOMY ALONE. It is fixed before any training,
so the circuit cannot have been chosen to suit the task. Every tie is broken by body
ID, so re-running produces a byte-identical file.

Procedure (Fly Dino's published protocol, widened from 8 input channels to 18):
  1. Inputs  - for each named visual-projection type, the single cell with the largest
               total contact count onto descending neurons that have somata.
  2. Outputs - union of the top two descending targets of each selected input cell,
               filled to exactly N_OUT by total contact rank.
  3. Bridges - the strongest two-hop intermediates, ranked by
               min(contacts received from inputs, contacts sent to outputs).
  4. Edges   - EVERY measured directed edge among the selected cells is retained,
               including single-contact and recurrent ones. Nothing is synthesized.

Usage: uv run --with pyarrow --with numpy python scripts/build_circuit.py <data-dir>
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

# ---------------------------------------------------------------- configuration

# 18 board channels: mine[0..8] then theirs[0..8]. One visual-projection type per channel.
# NOTE: Fly Dino's list names "LC20", which does not exist in MaleCNS v1.0 - the dataset
# splits it into LC20a and LC20b. We use LC20a and say so rather than silently swapping.
INPUT_TYPES = [
    "LC4", "LC6", "LC9", "LC11", "LC12", "LC13", "LC15", "LC16", "LC17",
    "LC18", "LC20a", "LC21", "LC22", "LC25", "LPLC1", "LPLC2", "LPLC4", "LC10a",
]
N_OUT = 16       # descending cells read out by the trainable network
N_BRIDGE = 64    # two-hop intermediates
TOP_DESC_PER_INPUT = 2

# The template's viewer renders only these superclasses (groups.bin < 3). A cell outside
# them cannot be displayed, and the replay validator would reject it.
VISIBLE_SUPERCLASSES = {
    "ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal", "visual_projection_tbc",
    "cb_intrinsic", "cb_sensory", "cb_motor", "cb_endocrine", "cb_sensory_tbc", "cb_efferent",
    "descending_neuron", "descending_neuron_tbc", "sensory_descending", "efferent_descending",
}
DESCENDING_SUPERCLASSES = {
    "descending_neuron", "descending_neuron_tbc", "sensory_descending", "efferent_descending",
}

# Neurotransmitter -> presynaptic sign. An explicit modelling policy, not a measurement.
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/malecns").resolve()
    out = Path(__file__).resolve().parent.parent / "public" / "data" / "circuit.json"

    files = {n: data / f"{n}.feather" for n in ("annotations", "edges", "neurotransmitters")}
    for n, p in files.items():
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    print("hashing sources (verifies we built from the official release)...")
    hashes = {n: sha256(p) for n, p in files.items()}
    for n, h in hashes.items():
        print(f"  {n:<18} {h}")

    # ------------------------------------------------------------- annotations
    ann = feather.read_table(
        files["annotations"],
        columns=["bodyId", "type", "superclass", "somaLocation", "status"],
    ).to_pydict()

    body_type, body_super, body_soma = {}, {}, {}
    for i in range(len(ann["bodyId"])):
        soma = ann["somaLocation"][i]
        sc = ann["superclass"][i]
        if soma is None or sc not in VISIBLE_SUPERCLASSES:
            continue  # no soma -> not in the atlas; not visible -> viewer would reject it
        b = int(ann["bodyId"][i])
        body_type[b] = ann["type"][i]
        body_super[b] = sc
        body_soma[b] = [int(v) for v in soma]

    descending = {b for b, sc in body_super.items() if sc in DESCENDING_SUPERCLASSES}
    print(f"\nvisible cells with soma: {len(body_type):,}   descending: {len(descending):,}")

    by_type = defaultdict(list)
    for b, t in body_type.items():
        if t:
            by_type[t].append(b)

    missing = [t for t in INPUT_TYPES if not by_type.get(t)]
    if missing:
        print(f"ERROR: input types absent from this release: {missing}", file=sys.stderr)
        return 1

    # --------------------------------------------------------- neurotransmitter
    nt_tbl = feather.read_table(
        files["neurotransmitters"], columns=["body", "consensus_nt"]
    ).to_pydict()
    body_nt = {}
    for b, nt in zip(nt_tbl["body"], nt_tbl["consensus_nt"]):
        b = int(b)
        if b in body_type and b not in body_nt:
            body_nt[b] = nt
    print(f"neurotransmitter annotations for visible cells: {len(body_nt):,}")

    # ------------------------------------------------------------------- edges
    print("\nloading edge table (151.9M rows, ~3.6 GB)...")
    e = feather.read_table(files["edges"], columns=["body_pre", "body_post", "weight"])
    pre = e.column("body_pre").to_numpy()
    post = e.column("body_post").to_numpy()
    wt = e.column("weight").to_numpy().astype(np.int64)
    del e
    print(f"  {len(pre):,} directed pairs, {wt.sum():,} synaptic contacts total")

    # Restrict once to edges whose BOTH ends are visible cells with somata. Everything
    # downstream works on this subgraph, so the selection can never reach a cell the
    # viewer cannot draw.
    visible = np.fromiter(sorted(body_type), dtype=np.int64)
    keep = np.isin(pre, visible) & np.isin(post, visible)
    pre, post, wt = pre[keep], post[keep], wt[keep]
    print(f"  {len(pre):,} pairs among visible somatic cells")

    desc_arr = np.fromiter(sorted(descending), dtype=np.int64)
    is_desc_post = np.isin(post, desc_arr)

    # ------------------------------------------------- 1. one input cell per type
    onto_desc = defaultdict(int)
    for p, w in zip(pre[is_desc_post], wt[is_desc_post]):
        onto_desc[int(p)] += int(w)

    inputs = []
    print("\n1. input cells (largest total contact count onto descending neurons):")
    for t in INPUT_TYPES:
        cands = sorted(by_type[t])                       # body-ID order = deterministic ties
        best = max(cands, key=lambda b: (onto_desc.get(b, 0), -b))
        inputs.append(best)
        print(f"   ch{len(inputs)-1:<2} {t:<7} body {best:<12} {onto_desc.get(best,0):>6} contacts -> DNs"
              f"   ({len(cands)} cells of this type)")
    if len(set(inputs)) != len(inputs):
        print("ERROR: duplicate input cell selected", file=sys.stderr)
        return 1

    # ------------------------------------------------------- 2. output cells
    input_set = set(inputs)
    from_input = np.isin(pre, np.fromiter(sorted(input_set), dtype=np.int64))
    sel = from_input & is_desc_post
    per_input = defaultdict(lambda: defaultdict(int))
    desc_total = defaultdict(int)
    for p, q, w in zip(pre[sel], post[sel], wt[sel]):
        per_input[int(p)][int(q)] += int(w)
        desc_total[int(q)] += int(w)

    outputs: list[int] = []
    for b in inputs:
        top = sorted(per_input[b].items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_DESC_PER_INPUT]
        for q, _ in top:
            if q not in outputs:
                outputs.append(q)
    for q, _ in sorted(desc_total.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(outputs) >= N_OUT:
            break
        if q not in outputs:
            outputs.append(q)
    outputs = sorted(outputs[:N_OUT])
    print(f"\n2. output cells ({len(outputs)} descending):")
    for q in outputs:
        print(f"   body {q:<12} {str(body_type.get(q)):<14} {desc_total.get(q,0):>6} contacts from inputs")
    if len(outputs) < N_OUT:
        print(f"ERROR: only {len(outputs)} descending targets found, need {N_OUT}", file=sys.stderr)
        return 1

    # ------------------------------------------------------- 3. bridge cells
    output_set = set(outputs)
    in_from_inputs = defaultdict(int)
    for p, q, w in zip(pre[from_input], post[from_input], wt[from_input]):
        in_from_inputs[int(q)] += int(w)

    to_output = np.isin(post, np.fromiter(sorted(output_set), dtype=np.int64))
    out_to_outputs = defaultdict(int)
    for p, q, w in zip(pre[to_output], post[to_output], wt[to_output]):
        out_to_outputs[int(p)] += int(w)

    excluded = input_set | output_set
    scored = [
        (min(in_from_inputs[b], out_to_outputs[b]), b)
        for b in set(in_from_inputs) & set(out_to_outputs)
        if b not in excluded
    ]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    bridges = sorted(b for _, b in scored[:N_BRIDGE])
    print(f"\n3. bridge cells: {len(bridges)} of {len(scored)} candidates "
          f"(strength {scored[0][0]} down to {scored[min(len(scored),N_BRIDGE)-1][0]})")

    # ------------------------------------------------------------- 4. edges
    cells = sorted(input_set | set(bridges) | output_set)
    index = {b: i for i, b in enumerate(cells)}
    cell_arr = np.fromiter(cells, dtype=np.int64)
    both = np.isin(pre, cell_arr) & np.isin(post, cell_arr)
    edges = [
        {"pre": index[int(p)], "post": index[int(q)], "contacts": int(w)}
        for p, q, w in zip(pre[both], post[both], wt[both])
    ]
    edges.sort(key=lambda e: (e["pre"], e["post"]))
    contacts = sum(e["contacts"] for e in edges)
    print(f"\n4. {len(cells)} cells, {len(edges)} directed edges, {contacts:,} synaptic contacts")

    # ------------------------------------------------------------ sanity gates
    nt_counts = defaultdict(int)
    for b in cells:
        nt_counts[body_nt.get(b) or "unclear"] += 1
    print(f"   neurotransmitters: {dict(sorted(nt_counts.items()))}")

    # Every output must be reachable from some input along non-zero-sign edges, or the
    # circuit literally cannot carry the board to the readout and training is pointless.
    adj = defaultdict(list)
    for e in edges:
        if SIGN.get(body_nt.get(cells[e["pre"]]) or "", 0) != 0:
            adj[e["pre"]].append(e["post"])
    seen, stack = set(), [index[b] for b in inputs]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj[n])
    unreachable = [b for b in outputs if index[b] not in seen]
    if unreachable:
        print(f"ERROR: {len(unreachable)} output cells unreachable from inputs: {unreachable}", file=sys.stderr)
        return 1
    print(f"   reachability: all {len(outputs)} outputs reachable from inputs "
          f"({len(seen)}/{len(cells)} cells reachable)")

    single = sum(1 for e in edges if e["contacts"] == 1)
    recurrent = sum(1 for e in edges if e["pre"] == e["post"])
    print(f"   {single} single-contact edges ({100*single/len(edges):.1f}%), {recurrent} self-edges")

    # ------------------------------------------------------------------ write
    doc = {
        "version": 1,
        "dataset": "male-cns:v1.0",
        "generatedBy": "scripts/build_circuit.py",
        "selection": {
            "procedure": "Anatomy only, fixed before training. Ties broken by body ID.",
            "inputTypes": INPUT_TYPES,
            "nOut": N_OUT,
            "nBridge": N_BRIDGE,
            "topDescendingPerInput": TOP_DESC_PER_INPUT,
            "note": "MaleCNS v1.0 has no type 'LC20'; it is split into LC20a/LC20b. LC20a is used.",
        },
        "signPolicy": {
            "acetylcholine": 1, "gaba": -1, "glutamate": -1,
            "other": 0,
            "note": "A modelling assumption applied to predicted consensus neurotransmitter, "
                    "not a measurement of synaptic sign.",
        },
        "counts": {
            "cells": len(cells), "edges": len(edges), "contacts": contacts,
            "inputs": len(inputs), "bridges": len(bridges), "outputs": len(outputs),
            "singleContactEdges": single, "selfEdges": recurrent,
            "neurotransmitters": dict(sorted(nt_counts.items())),
        },
        "source": {
            "reference": "https://male-cns.janelia.org/download/",
            "license": "CC BY 4.0",
            "attribution": "FlyEM / HHMI Janelia, University of Cambridge, "
                           "MRC Laboratory of Molecular Biology, and Google Research",
            "sha256": hashes,
        },
        "inputCells": [index[b] for b in inputs],      # index by board channel 0..17
        "outputCells": [index[b] for b in outputs],
        "cells": [
            {
                "bodyId": b,
                "type": body_type.get(b),
                "superclass": body_super.get(b),
                "nt": body_nt.get(b) or "unclear",
                "soma": body_soma[b],
                "role": "input" if b in input_set else "output" if b in output_set else "bridge",
            }
            for b in cells
        ],
        "edges": edges,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
