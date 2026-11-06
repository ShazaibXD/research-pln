"""Learner readiness: propagation of mastery uncertainty through the multilinear support function.

Implements derivation.md section 8. With independent latent mastery X_s ~ Bernoulli(mu_s):

    E[f_c(X)]   = f_c(mu)                                                   (3)
    Var[f_c(X)] = sum_{U != empty} (d_U f_c(mu))^2 prod_{s in U} mu_s(1-mu_s)   (4)
    d_U f_c(mu) = sum_{T contains U} m_c(T) prod_{s in T minus U} mu_s

and readiness rho_c = E[sigmoid(f_c(X)/tau)], approximated by a probit-style formula or computed
exactly by enumeration for at most 16 relevant skills. Readiness is a belief about whether the
current component skills form a sufficient foundation. It is not mastery of the competency.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from pln.algorithm import setfunc
from pln.algorithm.interaction_model import Coeffs
from pln.errors import PLNInputError

EXACT_LIMIT = 16


@dataclass
class Moments:
    mean: np.ndarray  # (Mc,) f_c(mu)
    var: np.ndarray  # (Mc,) total variance
    var_by_order: np.ndarray  # (Mc, k): variance attributed to uncertainty in order-j skill groups


def _check_mu(mu: np.ndarray, L: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (L,):
        raise PLNInputError(f"mu must have shape ({L},), got {mu.shape}")
    if not np.isfinite(mu).all() or (mu < 0).any() or (mu > 1).any():
        raise PLNInputError("mu must be finite and within [0, 1]")
    return mu


def mastery_moments(co: Coeffs, mu: np.ndarray, row: int = 0) -> Moments:
    """Exact mean and variance of f_c(X) for one set `row`, X_s ~ Bernoulli(mu_s) independent (P5)."""
    L = co.length
    mu = _check_mu(mu, L)
    Mc = co.bias.shape[0]
    k = len(co.orders)
    mean = co.bias.detach().cpu().double().numpy().copy()
    d: Dict[Tuple[int, ...], np.ndarray] = {}
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        combos = oc.combos.tolist()
        valid = oc.valid[row].tolist()
        m = oc.m[row].detach().cpu().double().numpy()  # (n, Mc)
        for t, T in enumerate(combos):
            if not valid[t]:
                continue
            mean += m[t] * float(np.prod(mu[list(T)]))
            for r in range(1, len(T) + 1):
                for U in itertools.combinations(T, r):
                    rest = [p for p in T if p not in U]
                    w = float(np.prod(mu[rest])) if rest else 1.0
                    d.setdefault(U, np.zeros(Mc))
                    d[U] = d[U] + m[t] * w
    var_by_order = np.zeros((Mc, k))
    for U, dU in d.items():
        var_by_order[:, len(U) - 1] += dU ** 2 * float(np.prod(mu[list(U)] * (1 - mu[list(U)])))
    return Moments(mean=mean, var=var_by_order.sum(axis=1), var_by_order=var_by_order)


def readiness_probit(mean: np.ndarray, var: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """sigmoid( mean / (tau * sqrt(1 + pi * var / (8 tau^2))) ): Gaussian-logistic approximation of E[sigmoid(f/tau)]."""
    if tau <= 0:
        raise PLNInputError("tau must be positive")
    return 1.0 / (1.0 + np.exp(-(mean / tau) / np.sqrt(1.0 + math.pi * var / (8.0 * tau ** 2))))


def readiness_exact(co: Coeffs, mu: np.ndarray, tau: float = 1.0, row: int = 0) -> Dict[str, np.ndarray]:
    """Exact E[sigmoid(f/tau)], E[f] and Var[f] by enumerating all 2^n mastery states of the n real skills of `row`."""
    L = co.length
    mu = _check_mu(mu, L)
    if tau <= 0:
        raise PLNInputError("tau must be positive")
    real = sorted({p for oc in co.orders if oc.m.shape[1] for t, T in enumerate(oc.combos.tolist())
                   if bool(oc.valid[row, t]) for p in T})
    # positions that carry any valid subset; singletons make every real skill appear when max_order >= 1
    n = len(real)
    if n > EXACT_LIMIT:
        raise PLNInputError(f"exact readiness is limited to {EXACT_LIMIT} relevant skills, got {n}")
    local = {p: i for i, p in enumerate(real)}
    Mc = co.bias.shape[0]
    m = np.zeros((Mc, 1 << n))
    for oc in co.orders:
        if oc.m.shape[1] == 0:
            continue
        vals = oc.m[row].detach().cpu().double().numpy()
        valid = oc.valid[row].tolist()
        for t, T in enumerate(oc.combos.tolist()):
            if valid[t]:
                m[:, sum(1 << local[p] for p in T)] += vals[t]
    f = setfunc.zeta_transform(m) + co.bias.detach().cpu().double().numpy()[:, None]
    p = np.ones(1)
    for i in range(n):  # mask bit i corresponds to local skill i
        mi = mu[real[i]]
        p = np.concatenate([p * (1 - mi), p * mi])
    # concatenate builds index = old + (bit * 2^i): [bit_i=0 block, bit_i=1 block] -> correct bit order
    ex_f = f @ p
    var = (f ** 2) @ p - ex_f ** 2
    ex_sig = (1.0 / (1.0 + np.exp(-f / tau))) @ p
    return {"readiness": ex_sig, "mean": ex_f, "var": var}


def readiness_credible_interval(co: Coeffs, alpha: np.ndarray, beta: np.ndarray, tau: float = 1.0, row: int = 0,
                                n_samples: int = 200, quantiles: Sequence[float] = (0.05, 0.5, 0.95),
                                seed: int = 0) -> np.ndarray:
    """Credible interval of probit readiness when mastery means themselves are uncertain: mu_s ~ Beta(alpha_s, beta_s).

    Returns an array (len(quantiles), Mc). Uses a fixed RNG seed for reproducibility.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if alpha.shape != (co.length,) or beta.shape != alpha.shape or (alpha <= 0).any() or (beta <= 0).any():
        raise PLNInputError("alpha and beta must be positive with shape (L,)")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        mu = rng.beta(alpha, beta)
        mo = mastery_moments(co, mu, row)
        out.append(readiness_probit(mo.mean, mo.var, tau))
    return np.quantile(np.stack(out), list(quantiles), axis=0)
