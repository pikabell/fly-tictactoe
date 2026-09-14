"""Do the visual-projection types we plan to use actually exist, with somata?"""
import sys
from collections import Counter
from pathlib import Path
import pyarrow.feather as feather

root = Path(sys.argv[1])
t = feather.read_table(root / "annotations.feather",
                       columns=["bodyId", "type", "superclass", "somaLocation", "status"])
d = t.to_pydict()
n = len(d["bodyId"])
print(f"rows: {n:,}")

has_soma = [i for i in range(n) if d["somaLocation"][i] is not None]
print(f"with somaLocation: {len(has_soma):,}   (template atlas count = 140,024)")

traced = [i for i in has_soma if d["status"][i] == "Traced"]
print(f"  of those, status=='Traced': {len(traced):,}")

print("\n--- superclass distribution (cells with soma) ---")
for k, v in Counter(d["superclass"][i] for i in has_soma).most_common(20):
    print(f"  {str(k):<28} {v:>8,}")

CAND = ["LC4","LC6","LC9","LC11","LC12","LC13","LC15","LC16","LC17","LC18",
        "LC20","LC21","LC22","LC25","LPLC1","LPLC2","LPLC4","LC10a"]
by_type = Counter(d["type"][i] for i in has_soma if d["type"][i])
print("\n--- candidate input types: cells WITH soma ---")
missing = []
for c in CAND:
    cnt = by_type.get(c, 0)
    print(f"  {c:<8} {cnt:>5}" + ("   <-- MISSING" if cnt == 0 else ""))
    if cnt == 0:
        missing.append(c)

print(f"\nmissing: {missing if missing else 'none'}")

lc_like = sorted((ty, c) for ty, c in by_type.items()
                 if ty and (ty.startswith("LC") or ty.startswith("LPLC")))
print(f"\n--- all LC*/LPLC* types present: {len(lc_like)} ---")
print("  " + ", ".join(f"{ty}({c})" for ty, c in lc_like))

desc = [i for i in has_soma if d["superclass"][i] and "descending" in d["superclass"][i]]
print(f"\ndescending-ish cells with soma: {len(desc):,}")
print("  superclasses:", dict(Counter(d["superclass"][i] for i in desc)))
