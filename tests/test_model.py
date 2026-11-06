import itertools

import numpy as np
import pytest
import torch

from pln.algorithm import setfunc
from pln.algorithm.interaction_model import MCN, logits_from_coeffs, validate_sets
from pln.errors import PLNInputError
from tests.conftest import make_model


def reference_cp_logits(model, idx, mask, skill_feats, comp_feats):
    """Independent CP reference: raw parameters, python loops over subsets, no DP and no use of MCN.coefficients."""
    cfg = model.cfg
    out = np.zeros((idx.shape[0], comp_feats.shape[0]))
    with torch.no_grad():
        eps = model.W_eps(comp_feats)
        q = model.W_q(eps)
        bias = model.bias_head(eps).squeeze(-1)
        for b in range(idx.shape[0]):
            S = [int(idx[b, l]) for l in range(idx.shape[1]) if mask[b, l]]
            e = {s: model.W_e(skill_feats[s]) for s in S}
            z = {s: model.W_z(e[s]) for s in S}
            for c in range(comp_feats.shape[0]):
                total = float(bias[c])
                lam = torch.einsum("krd,d->kr", model.V, eps[c])
                for j in range(1, cfg.max_order + 1):
                    for T in itertools.combinations(S, j):
                        gate = 1.0
                        for s in T:
                            gate *= float(torch.sigmoid(e[s] @ q[c] + model.gate_bias)) if cfg.use_gate else 1.0
                        prod = torch.ones(cfg.rank, dtype=torch.float64)
                        for s in T:
                            prod = prod * z[s]
                        total += gate * float((lam[j - 1] * prod).sum())
                out[b, c] = total
    return out


def test_forward_matches_coefficients(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    f = model(idx, mask, **world)
    co = model.coefficients(idx, mask, **world)
    assert torch.allclose(f, logits_from_coeffs(co), atol=1e-10)


def test_cp_matches_independent_reference(world, sets):
    model = make_model("cp")
    idx, mask = sets
    ref = reference_cp_logits(model, idx, mask, **world)
    assert np.allclose(model(idx, mask, **world).detach().numpy(), ref, atol=1e-10)


def test_forward_matches_setfunction_table(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    co = model.coefficients(idx, mask, **world)
    f = model(idx, mask, **world)
    for b in range(idx.shape[0]):
        real = [p for p in range(idx.shape[1]) if mask[b, p]]
        local = {p: i for i, p in enumerate(real)}
        for c in range(world["comp_feats"].shape[0]):
            md = {}
            for oc in co.orders:
                for t, T in enumerate(oc.combos.tolist()):
                    if oc.valid[b, t]:
                        md[sum(1 << local[p] for p in T)] = md.get(sum(1 << local[p] for p in T), 0.0) + float(oc.m[b, t, c].detach())
            table = setfunc.table_from_coefficients(md, float(co.bias[c].detach()), len(real))
            assert abs(table[-1] - float(f[b, c].detach())) < 1e-9


def test_permutation_and_padding_invariance(variant, world):
    model = make_model(variant)
    idx_a = torch.tensor([[3, 8, 12, 0, 0, 0]])
    mask_a = torch.tensor([[1, 1, 1, 0, 0, 0]]).bool()
    idx_b = torch.tensor([[12, 0, 3, 8]])  # permuted, padding in the middle, shorter
    mask_b = torch.tensor([[1, 0, 1, 1]]).bool()
    assert torch.allclose(model(idx_a, mask_a, **world), model(idx_b, mask_b, **world), atol=1e-10)


def test_binary_weights_equal_subset(variant, world):
    model = make_model(variant)
    idx = torch.tensor([[1, 5, 7, 9]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    sub = model(torch.tensor([[1, 7, 9]]), torch.ones(1, 3, dtype=torch.bool), **world)
    assert torch.allclose(model(idx, mask, weights=w, **world), sub, atol=1e-10)


def test_fractional_weights_are_multilinear_extension(variant, world):
    model = make_model(variant)
    idx = torch.tensor([[2, 4, 6, 8]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    mu = torch.tensor([[0.9, 0.2, 0.5, 0.7]], dtype=torch.float64)
    co = model.coefficients(idx, mask, **world)
    expected = co.bias[None].clone()
    for oc in co.orders:
        px = mu[:, oc.combos].prod(-1)
        expected = expected + (oc.m * px[:, :, None]).sum(1)
    assert torch.allclose(model(idx, mask, weights=mu, **world), expected, atol=1e-10)
    # and E_X f(X) over Bernoulli(mu) equals f(mu)
    total = torch.zeros_like(expected)
    for bits in itertools.product([0, 1], repeat=4):
        p = np.prod([mu[0, i].item() if b else 1 - mu[0, i].item() for i, b in enumerate(bits)])
        total = total + p * model(idx, mask, weights=torch.tensor([bits], dtype=torch.float64), **world)
    assert torch.allclose(total, expected, atol=1e-9)


def test_max_order_restriction_and_parts(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    full, parts = model(idx, mask, return_parts=True, **world)
    bias = model.bias_head(model.W_eps(world["comp_feats"])).squeeze(-1)
    assert torch.allclose(full, bias[None] + parts.sum(-1), atol=1e-10)
    only1 = model(idx, mask, max_order=1, **world)
    assert torch.allclose(only1, bias[None] + parts[..., 0], atol=1e-10)
    with pytest.raises(PLNInputError):
        model(idx, mask, max_order=4, **world)


def test_empty_gate_kills_interactions(variant, world, sets):
    model = make_model(variant)
    with torch.no_grad():
        model.gate_bias.fill_(-60.0)
    idx, mask = sets
    f, parts = model(idx, mask, return_parts=True, **world)
    assert parts.abs().max() < 1e-20  # every coefficient carries at least one near-zero gate factor


def test_competency_subset_selection(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    full = model(idx, mask, **world)
    sel = torch.tensor([2, 0])
    assert torch.allclose(model(idx, mask, comp_idx=sel, **world), full[:, sel], atol=1e-10)


@pytest.mark.parametrize("bad", ["dup", "range", "dtype", "length", "weights", "maskdtype"])
def test_input_validation(bad, world):
    model = make_model("cp", n_max=4)
    idx = torch.tensor([[1, 2, 3]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    kw = dict(weights=None)
    if bad == "dup":
        idx = torch.tensor([[1, 2, 2]])
    elif bad == "range":
        idx = torch.tensor([[1, 2, 99]])
    elif bad == "dtype":
        idx = idx.float()
    elif bad == "length":
        idx, mask = torch.zeros(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.bool)
    elif bad == "weights":
        kw["weights"] = torch.tensor([[0.5, 1.5, 0.2]], dtype=torch.float64)
    elif bad == "maskdtype":
        mask = mask.long()
    with pytest.raises(PLNInputError):
        model(idx, mask, **world, **kw)


def test_padding_with_duplicate_index_is_allowed(world):
    model = make_model("cp")
    idx = torch.tensor([[4, 4, 4]])  # padding positions repeat index 4, only the first is real
    mask = torch.tensor([[1, 0, 0]]).bool()
    validate_sets(idx, mask, 20, 6)
    assert model(idx, mask, **world).shape == (1, 4)


def test_gradients_are_finite_and_nonzero(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    model(idx, mask, **world).pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_gradcheck_wrt_presence_weights(variant, world):
    model = make_model(variant, k=3)
    idx = torch.tensor([[1, 5, 7, 9]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = torch.tensor([[0.3, 0.8, 0.5, 0.9]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: model(idx, mask, weights=x, **world, validate=False), (w,), eps=1e-6, atol=1e-6)
