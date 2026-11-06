"""Explanations computed from the model's own coefficients (derivation.md sections 6.3 and 7).

Everything here is a closed-form function of the Moebius coefficients m_c(T) returned by
`MCN.coefficients`, or of `MCN.forward` differences. None of it is a separate explainer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from pln.algorithm import setfunc
from pln.algorithm.interaction_model import MCN, Coeffs
from pln.errors import PLNInputError


def order_parts(co: Coeffs) -> torch.Tensor:
    """F_j(S) = sum_{T within S, |T|=j} m(T), shape (B, Mc, k). The score is bias + sum_j F_j (P2)."""
    return torch.stack([oc.m.sum(dim=1) for oc in co.orders], dim=-1)


def shapley(co: Coeffs) -> torch.Tensor:
    """Per-skill Shapley credit phi_s = sum_{T contains s} m(T)/|T|, shape (B, L, Mc) (P3)."""
    B = co.orders[0].m.shape[0]
    Mc = co.bias.shape[0]
    phi = torch.zeros(B, co.length, Mc, dtype=co.bias.dtype, device=co.bias.device)
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        for col in range(oc.order):
            phi.index_add_(1, oc.combos[:, col], oc.m / oc.order)
    return phi


def pair_interaction(co: Coeffs) -> torch.Tensor:
    """Pair interaction index I_{st} = sum_{T contains {s,t}} m(T)/(|T|-1), shape (B, L, L, Mc), symmetric, zero diagonal (P3)."""
    B = co.orders[0].m.shape[0]
    Mc = co.bias.shape[0]
    out = torch.zeros(B, co.length, co.length, Mc, dtype=co.bias.dtype, device=co.bias.device)
    b_idx = torch.arange(B, device=out.device)[:, None]
    for oc in co.orders:
        if oc.order < 2 or oc.m.shape[1] == 0:
            continue
        val = oc.m / (oc.order - 1)
        n = oc.combos.shape[0]
        for a in range(oc.order):
            for c in range(a + 1, oc.order):
                P = oc.combos[:, a][None].expand(B, n)
                Q = oc.combos[:, c][None].expand(B, n)
                out.index_put_((b_idx.expand(B, n), P, Q), val, accumulate=True)
                out.index_put_((b_idx.expand(B, n), Q, P), val, accumulate=True)
    return out


def integration_index(co: Coeffs, idx: torch.Tensor, mask: torch.Tensor, skill_domains: torch.Tensor) -> torch.Tensor:
    """Cross-domain integration index J_c(S) of derivation.md eq. (5), shape (B, Mc).

    `skill_domains` is a (N, D) boolean multi-hot domain matrix. J is the share of absolute coefficient
    mass on groups T whose union of domains has size >= 2. It is 0 when all coefficients are 0.
    It is a descriptive statistic of the fitted function, not evidence that integration is useful.
    """
    if skill_domains.dtype != torch.bool or skill_domains.ndim != 2:
        raise PLNInputError("skill_domains must be a 2-D bool tensor (N, D)")
    safe = torch.where(mask, idx, torch.zeros_like(idx))
    dom = skill_domains[safe]  # (B, L, D)
    B = idx.shape[0]
    total = torch.zeros(B, co.bias.shape[0], dtype=co.bias.dtype, device=co.bias.device)
    cross = torch.zeros_like(total)
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        union = dom[:, oc.combos].any(dim=2)  # (B, n, D)
        multi = union.sum(-1) >= 2  # (B, n)
        a = oc.m.abs()
        total = total + a.sum(1)
        cross = cross + (a * multi[:, :, None].to(a.dtype)).sum(1)
    return torch.where(total > 0, cross / total.clamp_min(1e-30), torch.zeros_like(total))


@torch.no_grad()
def completion_values(model: MCN, idx_row: torch.Tensor, candidates: torch.Tensor, skill_feats: torch.Tensor,
                      comp_feats: torch.Tensor, comp_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Marginal completion value f_c(S + s) - f_c(S) for each candidate s not in S, computed from coefficients (P4).

    idx_row: (n,) long, the real skills of S. candidates: (C,) long. Returns (C, Mc).
    """
    if idx_row.ndim != 1 or candidates.ndim != 1:
        raise PLNInputError("idx_row and candidates must be 1-D")
    if set(idx_row.tolist()) & set(candidates.tolist()):
        raise PLNInputError("candidates must not already be in the set")
    n, C = idx_row.shape[0], candidates.shape[0]
    rows = torch.cat([idx_row[None].expand(C, n), candidates[:, None]], dim=1)  # new skill is the LAST position
    mask = torch.ones_like(rows, dtype=torch.bool)
    co = model.coefficients(rows, mask, skill_feats, comp_feats, comp_idx=comp_idx)
    gain = torch.zeros(C, co.bias.shape[0], dtype=co.bias.dtype, device=co.bias.device)
    last = n  # position index of the appended skill
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        has_last = (oc.combos == last).any(dim=1)  # (n_j,)
        gain = gain + (oc.m * has_last[None, :, None].to(oc.m.dtype)).sum(1)
    return gain


def set_function_table(co: Coeffs, mask_row: torch.Tensor, row: int, col: int) -> np.ndarray:
    """Full value table f[S'] over all subsets S' of the REAL skills of one set, for one competency column.

    Returned array has length 2^n (n = number of real skills, n <= 20); bit i refers to the i-th real skill.
    Includes the bias.
    """
    real = [i for i, v in enumerate(mask_row.tolist()) if v]
    n = len(real)
    if n > setfunc.MAX_EXACT:
        raise PLNInputError(f"set too large for an exact table (n={n} > {setfunc.MAX_EXACT})")
    local = {p: i for i, p in enumerate(real)}
    m = np.zeros(1 << n)
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        combos = oc.combos.tolist()
        valid = oc.valid[row].tolist()
        vals = oc.m[row, :, col].detach().cpu().double().numpy()
        for t, combo in enumerate(combos):
            if valid[t]:
                m[sum(1 << local[p] for p in combo)] += vals[t]
    return setfunc.zeta_transform(m) + float(co.bias[col].detach().cpu())


def purified_energy(co: Coeffs, mask_row: torch.Tensor, row: int, col: int, rho: float = 0.5, order: int = 2) -> float:
    """Pure order>`order` energy ratio E_{>2} of derivation.md eq. (6) under the product measure with presence prob. rho.

    Defined for sets of at most 12 real skills. Returns 0.0 if f is constant under the reference measure.
    This describes the fitted function; it is not a test of whether higher-order structure is real.
    """
    table = set_function_table(co, mask_row, row, col)
    n = table.shape[0].bit_length() - 1
    if n > 12:
        raise PLNInputError("purified energy is limited to sets of at most 12 skills")
    resid, var = setfunc.least_squares_residual_energy(table, n, order, rho)
    return 0.0 if var <= 1e-18 else float(resid / var)
