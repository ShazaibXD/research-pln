import numpy as np
import pytest
import torch

from pln.algorithm import explain, setfunc
from pln.errors import PLNInputError
from tests.conftest import coeffs_from_dict, make_model


def test_parts_sum_to_score(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    co = model.coefficients(idx, mask, **world)
    f = model(idx, mask, **world)
    assert torch.allclose(co.bias[None] + explain.order_parts(co).sum(-1), f, atol=1e-10)


def test_shapley_and_interaction_match_bruteforce(variant, world):
    model = make_model(variant)
    idx = torch.tensor([[1, 5, 7, 9, 11]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    co = model.coefficients(idx, mask, **world)
    phi = explain.shapley(co)
    inter = explain.pair_interaction(co)
    L = 5
    for c in range(world["comp_feats"].shape[0]):
        md = {}
        for oc in co.orders:
            for t, T in enumerate(oc.combos.tolist()):
                md[sum(1 << p for p in T)] = float(oc.m[0, t, c].detach())
        table = setfunc.table_from_coefficients(md, 0.0, L)
        v = lambda S: table[S]
        for i in range(L):
            assert abs(phi[0, i, c].item() - setfunc.shapley_bruteforce(v, L, i)) < 1e-9
        for i in range(L):
            for j in range(i + 1, L):
                ref = setfunc.interaction_bruteforce(v, L, i, j)
                assert abs(inter[0, i, j, c].item() - ref) < 1e-9
                assert abs(inter[0, j, i, c].item() - ref) < 1e-9
        assert abs(phi[0, :, c].sum().item() - (table[-1] - table[0])) < 1e-9  # efficiency


def test_completion_equals_forward_difference(variant, world):
    model = make_model(variant)
    S = torch.tensor([2, 4, 6])
    cands = torch.tensor([1, 8, 15])
    gains = explain.completion_values(model, S, cands, **world)
    mask = torch.ones(1, 3, dtype=torch.bool)
    base = model(S[None], mask, **world)
    for n, s in enumerate(cands.tolist()):
        aug = torch.cat([S, torch.tensor([s])])[None]
        f1 = model(aug, torch.ones(1, 4, dtype=torch.bool), **world)
        assert torch.allclose(gains[n], (f1 - base)[0], atol=1e-9)
    with pytest.raises(PLNInputError):
        explain.completion_values(model, S, torch.tensor([2]), **world)


def test_completion_identifies_missing_conjunct():
    co = coeffs_from_dict({(0, 1, 2): 2.0}, bias=-1.0, L=3)
    # f({0,1}) = -1, f({0,1,2}) = +1: adding position 2 gains exactly 2
    gain = sum(oc.m[0, :, 0][(oc.combos == 2).any(1)].sum() for oc in co.orders)
    assert abs(gain.item() - 2.0) < 1e-12


def test_integration_index_toy():
    # positions 0,1 in domain 0 (CS), position 2 in domain 1 (STAT)
    co = coeffs_from_dict({(0,): 0.5, (1,): 0.4, (2,): 0.3, (0, 1): 0.1, (0, 2): 0.9, (0, 1, 2): 0.6}, 0.0, L=3)
    idx = torch.tensor([[0, 1, 2]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    dom = torch.tensor([[1, 0], [1, 0], [0, 1]]).bool()
    J = explain.integration_index(co, idx, mask, dom)
    assert abs(J.item() - (0.9 + 0.6) / (0.5 + 0.4 + 0.3 + 0.1 + 0.9 + 0.6)) < 1e-12
    zero = coeffs_from_dict({(0,): 0.0}, 0.0, L=3)
    assert explain.integration_index(zero, idx, mask, dom).item() == 0.0
    with pytest.raises(PLNInputError):
        explain.integration_index(co, idx, mask, dom.long())


def test_purified_energy_and3_and_nonmonotone_check():
    co = coeffs_from_dict({(0, 1, 2): 1.0}, 0.0, L=3)
    mask = torch.ones(3, dtype=torch.bool)
    assert abs(explain.purified_energy(co, mask, 0, 0, rho=0.5) - (1 / 64) / (7 / 64)) < 1e-9
    pair_only = coeffs_from_dict({(0, 1): 1.0, (1, 2): -0.5, (0,): 0.3}, 0.0, L=3)
    assert explain.purified_energy(pair_only, mask, 0, 0, rho=0.5) < 1e-12  # nothing beyond order 2
    # a 3-way term is detected at several reference measures, with size depending on rho
    vals = [explain.purified_energy(co, mask, 0, 0, rho=r) for r in (0.2, 0.5, 0.8)]
    assert all(v > 0 for v in vals) and vals[0] > vals[1] > vals[2]  # matches numeric_checks C5: 51.6%, 14.3%, 1.6%


def test_set_function_table_matches_forward(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    co = model.coefficients(idx, mask, **world)
    f = model(idx, mask, **world)
    for b in range(idx.shape[0]):
        for c in range(4):
            table = explain.set_function_table(co, mask[b], b, c)
            assert abs(table[-1] - f[b, c].item()) < 1e-9
