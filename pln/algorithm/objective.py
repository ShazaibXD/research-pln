"""Training objective (derivation.md section 11).

Labels per (set, competency): +1 annotated positive, -1 expert-reviewed negative, 0 unknown.
Unknown is never treated as negative by the default `nnpu` objective:

    R_c = pi_c R_P^+ + max(0, R_U^- - pi_c R_P^-)         (non-negative PU risk, Kiryo et al. 2017)

where U is the pool of ALL candidate sets for competency c (labelled or not), which is the correct
reference for the single-sample ("censoring") setting; see derivation.md erratum E1. The
`unknown_as_negative` objective exists only as a biased baseline for experiments.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pln.algorithm.config import TrainConfig
from pln.algorithm.interaction_model import MCN, Coeffs
from pln.errors import PLNInputError


def validate_labels(labels: torch.Tensor, weights: Optional[torch.Tensor], priors: torch.Tensor, shape) -> None:
    if labels.shape != tuple(shape):
        raise PLNInputError(f"labels must have shape {tuple(shape)}, got {tuple(labels.shape)}")
    if not ((labels == 1) | (labels == -1) | (labels == 0)).all():
        raise PLNInputError("labels must be in {+1, -1, 0} (0 means unknown)")
    if weights is not None:
        if weights.shape != labels.shape or (weights <= 0).any() or (weights > 1).any():
            raise PLNInputError("weights must match labels and lie in (0, 1]")
    if priors.shape != (shape[1],) or (priors <= 0).any() or (priors >= 1).any():
        raise PLNInputError("priors must have shape (M,) with values in (0, 1)")


def _logistic(f: torch.Tensor, y: float) -> torch.Tensor:
    return F.softplus(-y * f)


def nnpu_risk(logits: torch.Tensor, labels: torch.Tensor, priors: torch.Tensor, weights: Optional[torch.Tensor] = None,
              mode: str = "kiryo", beta: float = 0.0, gamma: float = 1.0) -> torch.Tensor:
    """Sum over competencies of the non-negative PU risk. Competencies with no positive in the batch are skipped.

    logits (B, M); labels (B, M) in {+1,-1,0}; priors (M,) = P(support) in the candidate pool.
    """
    validate_labels(labels, weights, priors, logits.shape)
    w = torch.ones_like(logits) if weights is None else weights
    pos = (labels == 1).to(logits.dtype)
    wp = pos * w
    n_pos = wp.sum(0)  # (M,)
    has = n_pos > 0
    r_pos_p = (wp * _logistic(logits, 1.0)).sum(0) / n_pos.clamp_min(1e-12)
    r_neg_p = (wp * _logistic(logits, -1.0)).sum(0) / n_pos.clamp_min(1e-12)
    r_neg_u = _logistic(logits, -1.0).mean(0)  # pool = every row for this competency
    neg_part = r_neg_u - priors * r_neg_p
    if mode == "clip":
        per_c = priors * r_pos_p + neg_part.clamp_min(0.0)
    elif mode == "kiryo":
        ok = neg_part >= -beta
        per_c = torch.where(ok, priors * r_pos_p + neg_part, -gamma * neg_part)
    else:
        raise PLNInputError(f"unknown nnpu mode {mode!r}")
    return (per_c * has.to(per_c.dtype)).sum()


def unknown_as_negative_risk(logits: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Biased baseline: positives are positive, EVERYTHING else (unknown and reviewed negatives) is negative."""
    w = torch.ones_like(logits) if weights is None else weights
    pos = labels == 1
    loss = torch.where(pos, w * _logistic(logits, 1.0), _logistic(logits, -1.0))
    return loss.mean(0).sum()


def reviewed_negative_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean logistic loss over reviewed negatives only (derivation.md 11.2); 0 if the batch has none."""
    neg = labels == -1
    if not neg.any():
        return logits.sum() * 0.0
    return _logistic(logits, -1.0)[neg].mean()


def sparsity_penalty(co: Coeffs, kappa: float) -> torch.Tensor:
    """kappa * sum_{j>=2} (j-1) * mean over rows and competencies of sum_T |m_c(T)| (derivation.md 11.3)."""
    total = co.bias.sum() * 0.0
    for oc in co.orders:
        if oc.order < 2 or oc.m.shape[1] == 0:
            continue
        total = total + (oc.order - 1) * oc.m.abs().sum(1).mean()
    return kappa * total


def monotonicity_penalty(model: MCN, idx: torch.Tensor, mask: torch.Tensor, skill_feats: torch.Tensor,
                         comp_feats: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mean of relu(-(f(A + i) - f(A)))^2 for one random (A, i) per row, A a random subset of the other skills (11.4)."""
    B, L = idx.shape
    r = torch.rand(B, L, generator=generator, device=idx.device)
    r = torch.where(mask, r, torch.full_like(r, -1.0))
    pick = r.argmax(dim=1)  # chosen skill i per row
    onehot = torch.nn.functional.one_hot(pick, L).bool() & mask
    in_A = (torch.rand(B, L, generator=generator, device=idx.device) < 0.5) & mask & ~onehot
    with_i = (in_A | onehot).to(skill_feats.dtype)
    only_A = in_A.to(skill_feats.dtype)
    f_with = model(idx, mask, skill_feats, comp_feats, weights=with_i, validate=False)
    f_wo = model(idx, mask, skill_feats, comp_feats, weights=only_A, validate=False)
    return F.relu(-(f_with - f_wo)).pow(2).mean()


def total_loss(model: MCN, cfg: TrainConfig, idx: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
               skill_feats: torch.Tensor, comp_feats: torch.Tensor, priors: torch.Tensor,
               weights: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """L = sum_c R_c + lambda_N L_neg + L_sparse + lambda_m L_mono (derivation.md eq. 10)."""
    logits = model(idx, mask, skill_feats, comp_feats)
    if cfg.objective == "nnpu":
        risk = nnpu_risk(logits, labels, priors, weights, cfg.nnpu_mode, cfg.nnpu_beta, cfg.nnpu_gamma)
        neg = reviewed_negative_loss(logits, labels) * cfg.lambda_neg
    else:
        risk = unknown_as_negative_risk(logits, labels, weights)
        neg = logits.sum() * 0.0
    sparse = logits.sum() * 0.0
    if cfg.kappa > 0 and model.cfg.max_order >= 2:
        sparse = sparsity_penalty(model.coefficients(idx, mask, skill_feats, comp_feats, validate=False), cfg.kappa)
    mono = logits.sum() * 0.0
    if cfg.lambda_mono > 0:
        if generator is None:
            raise PLNInputError("a torch.Generator is required when lambda_mono > 0")
        mono = cfg.lambda_mono * monotonicity_penalty(model, idx, mask, skill_feats, comp_feats, generator)
    return {"total": risk + neg + sparse + mono, "risk": risk.detach(), "neg": neg.detach(),
            "sparse": sparse.detach(), "mono": mono.detach()}
