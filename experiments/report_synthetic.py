"""Turn research/results/synthetic_planted.json into the tables of research/results/synthetic_planted.md."""

from __future__ import annotations

import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "research/results/synthetic_planted.json"
raw = json.load(open(path))
res, meta = raw["results"], raw["meta"]
seeds = meta["seeds"]


def col(name, key):
    return np.array([r[key] for r in res[name]])


def ms(a):
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"


lines = []
lines.append(f"Seeds: {seeds} (n = {len(seeds)}). Mean ± sample std over seeds. Metric: mean average precision over competencies "
             "against the planted ground truth (known only because the data are synthetic).\n")
keys = [("val_ap", "validation"), ("test_seen_ap", "test, seen skills"), ("test_unseen_ap", "test, unseen skills"),
        ("test_unseen_ap_conj2", "unseen: conj2"), ("test_unseen_ap_conj3", "unseen: conj3"), ("test_unseen_ap_alt3", "unseen: alt3")]
lines.append("| config | params | " + " | ".join(k[1] for k in keys) + " |")
lines.append("|---|---|" + "---|" * len(keys))
for name in res:
    lines.append(f"| {name} | {meta['n_params'][name]} | " + " | ".join(ms(col(name, k[0])) for k in keys) + " |")

lines.append("\nPaired differences across seeds (same data, same seed), on the test splits:\n")
lines.append("| comparison | split / type | mean diff | min | max | seeds with diff > 0 |")
lines.append("|---|---|---|---|---|---|")
for a, b in (("k3_nnpu", "k2_nnpu"), ("k2_nnpu", "k1_nnpu"), ("k3_nnpu", "k2_rank21_nnpu"), ("k2_nnpu", "k2_unknown_neg"), ("k3_nnpu", "k3_unknown_neg")):
    for key, lab in keys[1:]:
        d = col(a, key) - col(b, key)
        lines.append(f"| {a} − {b} | {lab} | {d.mean():+.3f} | {d.min():+.3f} | {d.max():+.3f} | {(d > 0).sum()}/{len(d)} |")
print("\n".join(lines))
