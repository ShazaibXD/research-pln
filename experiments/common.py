"""Shared helpers for synthetic experiments. Everything here runs on SYNTHETIC data only."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from pln.algorithm.config import MCNConfig, TrainConfig
from pln.algorithm.interaction_model import MCN
from pln.data.synthetic import SyntheticData
from pln.evaluation import metrics
from pln.training.train import SetTensors, fit


def mean_ap(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    aps = [metrics.average_precision(y_true[:, c].numpy(), logits[:, c].numpy()) for c in range(y_true.shape[1])]
    return float(np.nanmean(aps))


def per_type_ap(logits: torch.Tensor, y_true: torch.Tensor, comp_type) -> Dict[str, float]:
    out = {}
    for t in sorted(set(comp_type)):
        cols = [c for c, ct in enumerate(comp_type) if ct == t]
        out[t] = float(np.nanmean([metrics.average_precision(y_true[:, c].numpy(), logits[:, c].numpy()) for c in cols]))
    return out


def run_one(data: SyntheticData, mcfg: MCNConfig, tcfg: TrainConfig) -> Dict[str, float]:
    sp = data.split
    mk = lambda name: SetTensors(data.idx[sp[name]], data.mask[sp[name]], data.labels[sp[name]])
    train, val = mk("train"), mk("val")
    y_val = data.y_true[sp["val"]]
    torch.manual_seed(tcfg.seed)
    model = MCN(mcfg)
    res = fit(model, train, val, data.skill_feats, data.comp_feats, data.priors, tcfg,
              lambda logits, _v: mean_ap(logits, y_val))
    model.eval()
    out = {"best_epoch": res.best_epoch, "val_ap": res.best_score}
    with torch.no_grad():
        for name in ("test_seen", "test_unseen"):
            rows = sp[name]
            logits = model(data.idx[rows], data.mask[rows], data.skill_feats, data.comp_feats)
            y = data.y_true[rows]
            out[f"{name}_ap"] = mean_ap(logits, y)
            for t, v in per_type_ap(logits, y, data.comp_type).items():
                out[f"{name}_ap_{t}"] = v
            out[f"{name}_prevalence"] = float(y.mean())
    return out
