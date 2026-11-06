import itertools

import numpy as np

from pln.algorithm import setfunc


def test_moebius_zeta_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=1 << 5)
    m = setfunc.moebius_transform(v)
    assert np.allclose(setfunc.zeta_transform(m), v)
    # matches the definition
    for T in range(1 << 5):
        ref = sum((-1) ** (setfunc.popcount(T) - setfunc.popcount(U)) * v[U] for U in setfunc.submasks(T))
        assert abs(m[T] - ref) < 1e-12


def test_batched_transform():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 1 << 4))
    assert np.allclose(setfunc.zeta_transform(setfunc.moebius_transform(v)), v)


def test_k_additive_has_zero_high_order_moebius():
    rng = np.random.default_rng(2)
    L, k = 5, 2
    md = {T: rng.normal() for T in range(1, 1 << L) if setfunc.popcount(T) <= k}
    f = setfunc.table_from_coefficients(md, 0.0, L)
    m = setfunc.moebius_transform(f - f[0])
    assert all(abs(m[T]) < 1e-12 for T in range(1 << L) if setfunc.popcount(T) > k)


def test_shapley_efficiency_bruteforce():
    rng = np.random.default_rng(3)
    L = 4
    f = rng.normal(size=1 << L)
    f[0] = 0
    v = lambda S: f[S]
    total = sum(setfunc.shapley_bruteforce(v, L, i) for i in range(L))
    assert abs(total - f[-1]) < 1e-12


def test_and3_residual_energy_known_values():
    vals = np.array([1.0 if S == 0b111 else 0.0 for S in range(8)])
    resid, var = setfunc.least_squares_residual_energy(vals, 3, 2, 0.5)
    assert abs(var - 7 / 64) < 1e-12
    assert abs(resid - 1 / 64) < 1e-12  # the pure 3-way Walsh component has coefficient 1/8
    resid3, _ = setfunc.least_squares_residual_energy(vals, 3, 3, 0.5)
    assert resid3 < 1e-20
