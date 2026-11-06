"""NumPy reference implementations of set-function algebra (Moebius basis).

These are deliberately simple and brute-force. They are the ground truth that the
vectorised PyTorch code in `interaction_model.py`, `explain.py` and `readiness.py`
is tested against, and they are used for small-set diagnostics. Sets are bitmasks
over L <= 20 local positions.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, Iterator, List, Sequence

import numpy as np

MAX_EXACT = 20


def popcount(x: int) -> int:
    return bin(x).count("1")


def members(mask: int) -> List[int]:
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


def submasks(mask: int) -> Iterator[int]:
    """All submasks of `mask`, including 0 and `mask`."""
    s = mask
    while True:
        yield s
        if s == 0:
            return
        s = (s - 1) & mask


def zeta_transform(m: np.ndarray) -> np.ndarray:
    """f[S] = sum_{T subset S} m[T] over all 2^L masks (last axis of length 2^L, others batched)."""
    f = np.array(m, dtype=np.float64, copy=True)
    size = f.shape[-1]
    L = size.bit_length() - 1
    if size != 1 << L or L > MAX_EXACT:
        raise ValueError("zeta_transform needs a last axis of length 2^L with L <= %d" % MAX_EXACT)
    for i in range(L):
        v = f.reshape(f.shape[:-1] + (-1, 2, 1 << i))
        v[..., 1, :] += v[..., 0, :]
    return f


def moebius_transform(v: np.ndarray) -> np.ndarray:
    """Inverse of `zeta_transform`: m[T] = sum_{U subset T} (-1)^{|T|-|U|} v[U]."""
    m = np.array(v, dtype=np.float64, copy=True)
    size = m.shape[-1]
    L = size.bit_length() - 1
    if size != 1 << L or L > MAX_EXACT:
        raise ValueError("moebius_transform needs a last axis of length 2^L with L <= %d" % MAX_EXACT)
    for i in range(L):
        w = m.reshape(m.shape[:-1] + (-1, 2, 1 << i))
        w[..., 1, :] -= w[..., 0, :]
    return m


def popcounts(L: int) -> np.ndarray:
    pc = np.zeros(1 << L, dtype=np.int64)
    for i in range(L):
        pc[1 << i : 1 << (i + 1)] = pc[: 1 << i] + 1
    return pc


def shapley_bruteforce(v: Callable[[int], float], n: int, i: int) -> float:
    others = [j for j in range(n) if j != i]
    total = 0.0
    for r in range(len(others) + 1):
        w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
        for K in itertools.combinations(others, r):
            mask = sum(1 << j for j in K)
            total += w * (v(mask | (1 << i)) - v(mask))
    return total


def interaction_bruteforce(v: Callable[[int], float], n: int, i: int, j: int) -> float:
    """Shapley interaction index of Grabisch and Roubens (1999) for the pair (i, j)."""
    others = [t for t in range(n) if t not in (i, j)]
    total = 0.0
    for r in range(len(others) + 1):
        w = math.factorial(n - r - 2) * math.factorial(r) / math.factorial(n - 1)
        for K in itertools.combinations(others, r):
            mask = sum(1 << t for t in K)
            d = v(mask | (1 << i) | (1 << j)) - v(mask | (1 << i)) - v(mask | (1 << j)) + v(mask)
            total += w * d
    return total


def table_from_coefficients(m_by_mask: Dict[int, float], bias: float, L: int) -> np.ndarray:
    """Value table f[S] for all 2^L masks from a sparse {mask: m(T)} dict."""
    m = np.zeros(1 << L)
    for T, c in m_by_mask.items():
        m[T] += c
    f = zeta_transform(m)
    return f + bias


def least_squares_residual_energy(vals: np.ndarray, L: int, order: int, p: float) -> "tuple[float, float]":
    """Residual energy of the P_p-weighted L2 projection of `vals` onto monomials of degree <= order.

    P_p is the product measure with presence probability p per position. Returns
    (residual_energy, variance_of_vals). The ratio is the pure order>`order` energy of
    derivation.md section 7.4 under that reference distribution.
    """
    if L > 12:
        raise ValueError("purified energy is enumerated exactly and limited to L <= 12")
    pc = popcounts(L)
    cols = [T for T in range(1 << L) if pc[T] <= order]
    A = np.array([[1.0 if (S & T) == T else 0.0 for T in cols] for S in range(1 << L)])
    w = np.array([p ** pc[S] * (1 - p) ** (L - pc[S]) for S in range(1 << L)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], vals * sw, rcond=None)
    resid = vals - A @ coef
    mean = float(w @ vals)
    return float(w @ (resid ** 2)), float(w @ (vals ** 2) - mean ** 2)
