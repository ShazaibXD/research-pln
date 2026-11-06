"""Controlled SYNTHETIC experiment: does the order k matter on a planted task? NOT a real-data result.

Protocol (fixed before the final run; see research/results/synthetic_planted.md):
  * data seeds 0..4 (seed 100 was used for exploration and is excluded), model seed = data seed
  * 80 epochs, all other hyper-parameters at the defaults of configs/mcn_default.yaml
  * model selection by validation AP against the planted truth; the test splits are evaluated once per run
  * configurations: k=1, k=2, k=3 (nnPU); k=2 with rank 21 (parameter-matched control to k=3);
    k=2 and k=3 trained with the biased unknown-as-negative objective
  * splits are by Jaccard-component groups; test_unseen contains skills never seen in training
Run: python experiments/synthetic_planted.py [--seeds 0 1 2 3 4] [--out research/results/synthetic_planted.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import run_one  # noqa: E402
from pln.algorithm.config import MCNConfig, TrainConfig  # noqa: E402
from pln.algorithm.interaction_model import MCN  # noqa: E402
from pln.data.synthetic import SyntheticSpec, generate  # noqa: E402

CONFIGS = {
    "k1_nnpu": dict(max_order=1, rank=16, objective="nnpu"),
    "k2_nnpu": dict(max_order=2, rank=16, objective="nnpu"),
    "k3_nnpu": dict(max_order=3, rank=16, objective="nnpu"),
    "k2_rank21_nnpu": dict(max_order=2, rank=21, objective="nnpu"),
    "k2_unknown_neg": dict(max_order=2, rank=16, objective="unknown_as_negative"),
    "k3_unknown_neg": dict(max_order=3, rank=16, objective="unknown_as_negative"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default="research/results/synthetic_planted.json")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = ap.parse_args()
    configs = {n: CONFIGS[n] for n in args.configs}
    results = {name: [] for name in configs}
    n_params = {}
    for seed in args.seeds:
        data = generate(SyntheticSpec(seed=seed))
        for name, c in configs.items():
            mcfg = MCNConfig(in_dim_skill=data.skill_feats.shape[1], in_dim_comp=data.comp_feats.shape[1],
                             max_order=c["max_order"], rank=c["rank"], n_max=7)
            tcfg = TrainConfig(seed=seed, epochs=args.epochs, objective=c["objective"])
            n_params[name] = sum(p.numel() for p in MCN(mcfg).parameters())
            t = time.time()
            r = run_one(data, mcfg, tcfg)
            r.update(seed=seed, seconds=round(time.time() - t, 1))
            results[name].append(r)
            print(f"seed {seed} {name}: val {r['val_ap']:.3f} seen {r['test_seen_ap']:.3f} unseen {r['test_unseen_ap']:.3f} "
                  f"({r['seconds']}s)", flush=True)
    meta = dict(synthetic=True, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                platform=platform.platform(), epochs=args.epochs, seeds=args.seeds, n_params=n_params)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
