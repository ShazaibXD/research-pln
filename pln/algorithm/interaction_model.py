"""Moebius Competency Network (MCN): embedding-conditioned k-additive support function.

Implements derivation.md section 6. For a competency c and a (weighted) set of skills x,

    f_c(x) = b_c + sum_{j=1..k} sum_{|T|=j} m_c(T) prod_{s in T} x_s,
    m_c(T) = prod_{s in T} a_{c,s} * h_j(eps_c, {e_s : s in T}),
    a_{c,s} = sigmoid(<q_c, e_s> + beta_0).

Two coefficient generators h_j are provided:
  * "cp": h_j = sum_r lambda_{j,r}(eps_c) prod_{s in T} z_{s,r}; evaluated in O(|S| k R) by the
          ANOVA dynamic programme, without enumerating subsets.
  * "ds": h_j = rho_j([eps_c ; sum_{s in T} psi_j(e_s)]) (Deep-Sets); evaluated by enumerating subsets.

Both expose `coefficients()` (explicit m_c(T) per subset, used by explanations, readiness and
sparsity), which is tested to reproduce `forward()` exactly.

Sets are given as padded index tensors `idx` (B, L) with a boolean `mask` (B, L).
`weights` (B, L) in [0, 1] replace the 0/1 presence indicator by the multilinear extension
(mastery probabilities), derivation.md section 8.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from pln.algorithm.config import MCNConfig
from pln.errors import PLNInputError


@dataclass
class OrderCoeffs:
    """Coefficients m_c(T) of one order j for every subset T of the padded positions."""

    order: int
    combos: torch.Tensor  # (n_j, j) long: position indices within the padded set
    valid: torch.Tensor  # (B, n_j) bool: all members are real (unpadded) skills
    m: torch.Tensor  # (B, n_j, Mc); exactly zero where not valid


@dataclass
class Coeffs:
    bias: torch.Tensor  # (Mc,)
    orders: List[OrderCoeffs]
    length: int  # padded set length L
    comp_idx: torch.Tensor  # (Mc,) competency indices these coefficients refer to


def validate_sets(idx: torch.Tensor, mask: torch.Tensor, n_skills: int, n_max: int,
                  weights: Optional[torch.Tensor] = None) -> None:
    """Raise PLNInputError if a batch of sets violates the input contract."""
    if idx.dtype != torch.long:
        raise PLNInputError(f"idx must be torch.long, got {idx.dtype}")
    if idx.ndim != 2 or mask.shape != idx.shape:
        raise PLNInputError(f"idx and mask must be 2-D with equal shape, got {tuple(idx.shape)} and {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise PLNInputError(f"mask must be bool, got {mask.dtype}")
    if idx.shape[1] > n_max:
        raise PLNInputError(f"padded set length {idx.shape[1]} exceeds n_max={n_max}")
    if (mask.sum(dim=1) > n_max).any():
        raise PLNInputError(f"a set has more than n_max={n_max} skills")
    real = idx[mask]
    if real.numel() and (real.min() < 0 or real.max() >= n_skills):
        raise PLNInputError(f"skill index out of range [0, {n_skills})")
    same = (idx[:, :, None] == idx[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(idx.shape[1], dtype=torch.bool, device=idx.device)[None]
    if (same & ~eye).any():
        raise PLNInputError("a set contains a duplicate skill")
    if weights is not None:
        if weights.shape != idx.shape:
            raise PLNInputError("weights must have the same shape as idx")
        if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise PLNInputError("weights must be finite and within [0, 1]")


class MCN(nn.Module):
    """Moebius Competency Network."""

    def __init__(self, cfg: MCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, R, k = cfg.embed_dim, cfg.rank, cfg.max_order
        self.W_e = nn.Linear(cfg.in_dim_skill, d)
        self.W_eps = nn.Linear(cfg.in_dim_comp, d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(cfg.gate_bias)))
        self.bias_head = nn.Linear(d, 1)
        if cfg.variant == "cp":
            self.W_z = nn.Linear(d, R, bias=False)
            self.V = nn.Parameter(torch.randn(k, R, d) * cfg.init_scale / (d ** 0.5))
        else:
            H = cfg.ds_hidden
            self.psi = nn.ModuleList([nn.Sequential(nn.Linear(d, H), nn.ReLU(), nn.Linear(H, H)) for _ in range(k)])
            self.rho = nn.ModuleList([nn.Sequential(nn.Linear(d + H, H), nn.ReLU(), nn.Linear(H, 1)) for _ in range(k)])
        self._combo_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers
    def _combos(self, L: int, j: int, device: torch.device) -> torch.Tensor:
        key = (L, j, str(device))
        if key not in self._combo_cache:
            c = list(itertools.combinations(range(L), j))
            t = torch.tensor(c, dtype=torch.long, device=device).reshape(len(c), j)
            self._combo_cache[key] = t
        return self._combo_cache[key]

    def embed_skills(self, skill_feats: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = torch.where(mask, idx, torch.zeros_like(idx))
        return self.W_e(skill_feats[safe])  # (B, L, d)

    def embed_comps(self, comp_feats: torch.Tensor, comp_idx: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if comp_idx is None:
            comp_idx = torch.arange(comp_feats.shape[0], device=comp_feats.device)
        return self.W_eps(comp_feats[comp_idx]), comp_idx  # (Mc, d)

    def _gate(self, e: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_gate:
            return torch.ones(e.shape[0], e.shape[1], eps.shape[0], device=e.device, dtype=e.dtype)
        q = self.W_q(eps)  # (Mc, d)
        return torch.sigmoid(torch.einsum("bld,md->blm", e, q) + self.gate_bias)

    def _check_order(self, max_order: Optional[int]) -> int:
        kk = self.cfg.max_order if max_order is None else max_order
        if not 1 <= kk <= self.cfg.max_order:
            raise PLNInputError(f"max_order must be in [1, {self.cfg.max_order}], got {kk}")
        return kk

    # ------------------------------------------------------------------ forward
    def forward(self, idx: torch.Tensor, mask: torch.Tensor, skill_feats: torch.Tensor, comp_feats: torch.Tensor,
                weights: Optional[torch.Tensor] = None, comp_idx: Optional[torch.Tensor] = None,
                max_order: Optional[int] = None, return_parts: bool = False, validate: bool = True):
        """Support logits f_c(S) of shape (B, Mc). With `return_parts`, also order parts F_j of shape (B, Mc, kk)."""
        if validate:
            validate_sets(idx, mask, skill_feats.shape[0], self.cfg.n_max, weights)
        kk = self._check_order(max_order)
        if self.cfg.variant == "ds":
            return self._forward_ds(idx, mask, skill_feats, comp_feats, weights, comp_idx, kk, return_parts)
        B, L = idx.shape
        e = self.embed_skills(skill_feats, idx, mask)
        eps, _ = self.embed_comps(comp_feats, comp_idx)
        x = mask.to(e.dtype) if weights is None else weights.to(e.dtype) * mask.to(e.dtype)
        z = self.W_z(e)  # (B, L, R)
        g = self._gate(e, eps)  # (B, L, Mc)
        zt = z[:, :, None, :] * g[:, :, :, None] * x[:, :, None, None]  # (B, L, Mc, R)
        Mc, R = eps.shape[0], self.cfg.rank
        a = [torch.ones(B, Mc, R, device=e.device, dtype=e.dtype)] + [
            torch.zeros(B, Mc, R, device=e.device, dtype=e.dtype) for _ in range(kk)]
        for l in range(L):  # ANOVA dynamic programme: a[t] <- a[t] + a[t-1] * z_l (descending t)
            zl = zt[:, l]
            for t in range(kk, 0, -1):
                a[t] = a[t] + a[t - 1] * zl
        lam = torch.einsum("krd,md->mkr", self.V[:kk], eps)  # (Mc, kk, R)
        parts = torch.stack([(a[j] * lam[:, j - 1][None]).sum(-1) for j in range(1, kk + 1)], dim=-1)  # (B, Mc, kk)
        logits = self.bias_head(eps).squeeze(-1)[None] + parts.sum(-1)
        return (logits, parts) if return_parts else logits

    def _forward_ds(self, idx, mask, skill_feats, comp_feats, weights, comp_idx, kk, return_parts):
        co = self.coefficients(idx, mask, skill_feats, comp_feats, comp_idx=comp_idx, max_order=kk, validate=False)
        x = mask.to(co.bias.dtype) if weights is None else weights.to(co.bias.dtype) * mask.to(co.bias.dtype)
        parts = []
        for oc in co.orders:
            px = x[:, oc.combos].prod(dim=-1)  # (B, n_j)
            parts.append((oc.m * px[:, :, None]).sum(1))  # (B, Mc)
        parts_t = torch.stack(parts, dim=-1)
        logits = co.bias[None] + parts_t.sum(-1)
        return (logits, parts_t) if return_parts else logits

    # ------------------------------------------------------------------ explicit coefficients
    def coefficients(self, idx: torch.Tensor, mask: torch.Tensor, skill_feats: torch.Tensor, comp_feats: torch.Tensor,
                     comp_idx: Optional[torch.Tensor] = None, max_order: Optional[int] = None,
                     validate: bool = True) -> Coeffs:
        """Enumerate m_c(T) for every subset T (|T| <= max_order) of every padded set. Cost grows as C(L, <=k)."""
        if validate:
            validate_sets(idx, mask, skill_feats.shape[0], self.cfg.n_max)
        kk = self._check_order(max_order)
        B, L = idx.shape
        e = self.embed_skills(skill_feats, idx, mask)
        eps, cidx = self.embed_comps(comp_feats, comp_idx)
        g = self._gate(e, eps)  # (B, L, Mc)
        bias = self.bias_head(eps).squeeze(-1)
        orders: List[OrderCoeffs] = []
        if self.cfg.variant == "cp":
            z = self.W_z(e)
            lam = torch.einsum("krd,md->mkr", self.V[:kk], eps)
        for j in range(1, kk + 1):
            combos = self._combos(L, j, idx.device)
            n = combos.shape[0]
            if n == 0:
                orders.append(OrderCoeffs(j, combos, torch.zeros(B, 0, dtype=torch.bool, device=idx.device),
                                          torch.zeros(B, 0, eps.shape[0], device=idx.device, dtype=e.dtype)))
                continue
            valid = mask[:, combos].all(dim=-1)  # (B, n)
            gate = g[:, combos].prod(dim=2)  # (B, n, Mc)
            if self.cfg.variant == "cp":
                zc = z[:, combos].prod(dim=2)  # (B, n, R)
                h = torch.einsum("bnr,mr->bnm", zc, lam[:, j - 1])
            else:
                s = self.psi[j - 1](e[:, combos]).sum(dim=2)  # (B, n, H)
                Mc = eps.shape[0]
                inp = torch.cat([eps[None, None].expand(B, n, Mc, -1), s[:, :, None, :].expand(B, n, Mc, -1)], dim=-1)
                h = self.rho[j - 1](inp).squeeze(-1)
            m = h * gate * valid[:, :, None].to(h.dtype)
            orders.append(OrderCoeffs(j, combos, valid, m))
        return Coeffs(bias=bias, orders=orders, length=L, comp_idx=cidx)


def logits_from_coeffs(co: Coeffs, x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Evaluate f from explicit coefficients: bias + sum_T m(T) prod x. x (B, L) defaults to all ones."""
    B = co.orders[0].m.shape[0]
    out = co.bias[None].expand(B, -1).clone()
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        px = torch.ones(B, oc.combos.shape[0], device=oc.m.device, dtype=oc.m.dtype) if x is None else x[:, oc.combos].prod(-1)
        out = out + (oc.m * px[:, :, None]).sum(1)
    return out
