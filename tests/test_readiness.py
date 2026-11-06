import math

import numpy as np
import pytest
import torch

from pln.algorithm import readiness
from pln.errors import PLNInputError
from tests.conftest import coeffs_from_dict, make_model

TOY = {(0,): 0.3, (1,): 0.2, (2,): 0.1, (0, 1): 0.4, (0, 2): 0.1, (1, 2): 0.2, (0, 1, 2): 1.5}


def test_toy_example_matches_derivation_section_10():
    co = coeffs_from_dict(TOY, bias=-3.0, L=3)
    mu = np.array([0.9, 0.8, 0.5])
    mo = readiness.mastery_moments(co, mu)
    assert abs(mo.mean[0] - (-1.5670)) < 5e-4
    assert abs(mo.var[0] - 1.1028) < 5e-4
    ex = readiness.readiness_exact(co, mu)
    assert abs(ex["readiness"][0] - 0.2202) < 5e-4
    approx = readiness.readiness_probit(mo.mean, mo.var)
    assert abs(approx[0] - 0.2127) < 5e-4
    assert abs(1 / (1 + math.exp(1.5670)) - 0.1726) < 5e-4  # naive sigmoid(f(mu)) differs by ~0.048


def test_moments_match_exact_enumeration(variant, world):
    model = make_model(variant)
    idx = torch.tensor([[1, 5, 7, 9, 11]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    co = model.coefficients(idx, mask, **world)
    mu = np.array([0.9, 0.2, 0.5, 0.7, 0.35])
    mo = readiness.mastery_moments(co, mu)
    ex = readiness.readiness_exact(co, mu)
    assert np.allclose(mo.mean, ex["mean"], atol=1e-9)
    assert np.allclose(mo.var, ex["var"], atol=1e-9)
    assert np.allclose(mo.var_by_order.sum(1), mo.var, atol=1e-12)
    f_mu = model(idx, mask, weights=torch.tensor(mu)[None], **world).detach().numpy()[0]
    assert np.allclose(mo.mean, f_mu, atol=1e-9)  # E f(X) = f(mu)


def test_point_mastery_has_zero_variance():
    co = coeffs_from_dict(TOY, -3.0, 3)
    mu = np.array([1.0, 0.0, 1.0])
    mo = readiness.mastery_moments(co, mu)
    assert mo.var[0] < 1e-18
    ex = readiness.readiness_exact(co, mu)
    f_S = -3.0 + 0.3 + 0.1 + 0.1  # set {0, 2}
    assert abs(ex["readiness"][0] - 1 / (1 + math.exp(-f_S))) < 1e-12


def test_probit_close_to_exact_when_moderate():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(30):
        co = coeffs_from_dict({(0,): rng.normal(), (1,): rng.normal(), (0, 1): rng.normal(), (2,): rng.normal(),
                               (0, 1, 2): rng.normal()}, rng.normal(), 3)
        mu = rng.uniform(0.1, 0.9, 3)
        mo = readiness.mastery_moments(co, mu)
        errs.append(abs(readiness.readiness_probit(mo.mean, mo.var)[0] - readiness_exact_val(co, mu)))
    assert max(errs) < 0.08  # an approximation, documented as such; mean error is much smaller
    assert np.mean(errs) < 0.03


def readiness_exact_val(co, mu):
    return readiness.readiness_exact(co, mu)["readiness"][0]


def test_credible_interval_reproducible_and_ordered():
    co = coeffs_from_dict(TOY, -3.0, 3)
    a, b = np.array([9.0, 8.0, 2.0]), np.array([1.0, 2.0, 2.0])
    q1 = readiness.readiness_credible_interval(co, a, b, seed=1)
    q2 = readiness.readiness_credible_interval(co, a, b, seed=1)
    assert np.array_equal(q1, q2)
    assert (q1[0] <= q1[1]).all() and (q1[1] <= q1[2]).all()


def test_input_validation():
    co = coeffs_from_dict(TOY, -3.0, 3)
    with pytest.raises(PLNInputError):
        readiness.mastery_moments(co, np.array([0.5, 0.5]))
    with pytest.raises(PLNInputError):
        readiness.mastery_moments(co, np.array([0.5, 1.2, 0.1]))
    with pytest.raises(PLNInputError):
        readiness.readiness_probit(np.array([0.0]), np.array([1.0]), tau=0.0)
    with pytest.raises(PLNInputError):
        readiness.readiness_credible_interval(co, np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
