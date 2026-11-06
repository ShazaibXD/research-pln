"""Training loop and reproducibility helpers (derivation.md section 12).

Generic over data: `fit` takes tensors and an `eval_fn`. Model selection uses the validation
score returned by `eval_fn`; the test split must never be passed in.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from pln.algorithm.config import MCNConfig, TrainConfig
from pln.algorithm.interaction_model import MCN
from pln.algorithm.objective import total_loss
from pln.errors import PLNInputError

log = logging.getLogger("pln.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise PLNInputError("device 'cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise PLNInputError("device 'mps' requested but MPS is not available")
    return torch.device(name)


@dataclass
class SetTensors:
    idx: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    weights: Optional[torch.Tensor] = None

    def take(self, rows: np.ndarray) -> "SetTensors":
        r = torch.as_tensor(rows, dtype=torch.long)
        return SetTensors(self.idx[r], self.mask[r], self.labels[r], None if self.weights is None else self.weights[r])


@dataclass
class FitResult:
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("-inf")


def fit(model: MCN, train: SetTensors, val: SetTensors, skill_feats: torch.Tensor, comp_feats: torch.Tensor,
        priors: torch.Tensor, cfg: TrainConfig, eval_fn: Callable[[torch.Tensor, SetTensors], float]) -> FitResult:
    """Train `model` in place; restore the weights of the best validation epoch before returning."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)
    skill_feats, comp_feats, priors = skill_feats.to(device), comp_feats.to(device), priors.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n = train.idx.shape[0]
    res = FitResult()
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        losses = []
        for start in range(0, n, cfg.batch_size):
            rows = perm[start : start + cfg.batch_size]
            idx, mask, labels = train.idx[rows].to(device), train.mask[rows].to(device), train.labels[rows].to(device)
            w = None if train.weights is None else train.weights[rows].to(device)
            out = total_loss(model, cfg, idx, mask, labels, skill_feats, comp_feats, priors, w, gen)
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(out["total"].detach()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val.idx.to(device), val.mask.to(device), skill_feats, comp_feats).cpu()
        score = float(eval_fn(val_logits, val))
        res.history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": score})
        log.debug("epoch %d loss %.4f val %.4f", epoch, res.history[-1]["train_loss"], score)
        if score > res.best_score:
            res.best_score, res.best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return res
