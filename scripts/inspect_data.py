"""Look before building: print the real schema of each MaleCNS table."""
import sys
from pathlib import Path
import pyarrow.feather as feather

root = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/malecns")

for name in ["annotations.feather", "neurotransmitters.feather", "edges.feather"]:
    p = root / name
    if not p.exists():
        print(f"\n### {name}: NOT DOWNLOADED YET\n")
        continue
    print(f"\n{'='*70}\n### {name}  ({p.stat().st_size/1e6:.1f} MB)\n{'='*70}")
    t = feather.read_table(p)
    print(f"rows={t.num_rows:,}  cols={t.num_columns}")
    for f in t.schema:
        print(f"  {f.name:<28} {f.type}")
    print("\n--- head(5) ---")
    d = t.slice(0, 5).to_pydict()
    for k, v in d.items():
        print(f"  {k:<28} {v}")
