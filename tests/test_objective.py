import numpy as np
import pytest
import torch

from pln.algorithm import objective
from pln.algorithm.config import MCNConfig, TrainConfig
from pln.errors import PLNConfigError, PLNInputError
from tests.conftest import make_model

sp = lambda z: float(np.log1p(np.exp(z)))


def reference_nnpu(logits, labels, priors, mode="clip"):
    total = 0.0
    for c in range(logits.shape[1]):
        f, y = logits[:, c], labels[:, c]
        P = f[y == 1]
        if len(P) == 0:
            continue
        r_pos = np.mean([sp(-v) for v in P])
        r_neg_p = np.mean([sp(v) for v in P])
        r_neg_u = np.mean([sp(v) for v in f])
        part = r_neg_u - priors[c] * r_neg_p
        total += priors[c] * r_pos + (max(part, 0.0) if mode == "clip" else part)
    return total


def test_nnpu_matches_hand_reference_and_skips_competencies_without_positives():
    f = torch.tensor([[2.0, -1.0, 0.5], [0.3, 0.2, 0.1], [-1.5, 1.0, 0.0], [0.7, -0.2, 0.4]], dtype=torch.float64)
    y = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, -1, 0]], dtype=torch.float64)  # competency 2 has no positive
    pri = torch.tensor([0.3, 0.4, 0.2], dtype=torch.float64)
    got = objective.nnpu_risk(f, y, pri, mode="clip").item()
    assert abs(got - reference_nnpu(f.numpy(), y.numpy(), pri.numpy())) < 1e-12
    # competency without positives contributes no gradient
    f2 = f.clone().requires_grad_(True)
    objective.nnpu_risk(f2, y, pri).backward()
    assert f2.grad[:, 2].abs().sum() == 0


def test_kiryo_branch_switches_when_negative_risk_is_negative():
    # pool is half confident positives (labelled) and half confident negatives; with a very high prior the
    # estimated negative part R_U^- - pi R_P^- is negative, so the ascent branch (loss = -gamma * part) applies
    f = torch.tensor([[8.0], [8.0], [8.0], [-8.0], [-8.0], [-8.0]], dtype=torch.float64)
    y = torch.tensor([[1], [1], [1], [0], [0], [0]], dtype=torch.float64)
    pri = torch.tensor([0.99], dtype=torch.float64)
    clip = objective.nnpu_risk(f, y, pri, mode="clip").item()
    kir = objective.nnpu_risk(f, y, pri, mode="kiryo", gamma=1.0).item()
    r_neg_u = (3 * sp(8.0) + 3 * sp(-8.0)) / 6
    part = r_neg_u - 0.99 * sp(8.0)
    assert part < 0
    assert abs(kir - (-1.0 * part)) < 1e-9
    assert abs(clip - 0.99 * sp(-8.0)) < 1e-9  # clipped: only the positive term remains


def test_pu_risk_is_unbiased_in_censoring_setting_and_naive_is_not():
    rng = np.random.default_rng(0)
    n, pi, label_freq = 400_000, 0.3, 0.5
    y = rng.random(n) < pi
    x = np.where(y[:, None], rng.normal([1.0, 0.5], 1.0, (n, 2)), rng.normal([-1.0, -0.5], 1.0, (n, 2)))
    labelled = y & (rng.random(n) < label_freq)
    w, b = np.array([1.2, -0.4]), -0.3
    f = torch.tensor(x @ w + b)[:, None]
    labels = torch.tensor(np.where(labelled, 1.0, 0.0))[:, None]
    pn = (objective._logistic(f, 1.0)[torch.tensor(y)].sum() + objective._logistic(f, -1.0)[~torch.tensor(y)].sum()) / n
    # PN risk of the derivation: pi * E_pos[l(+)] + (1-pi) * E_neg[l(-)]
    true = pi * objective._logistic(f, 1.0)[torch.tensor(y)].mean() + (1 - pi) * objective._logistic(f, -1.0)[~torch.tensor(y)].mean()
    pu = objective.nnpu_risk(f, labels, torch.tensor([pi], dtype=torch.float64), mode="clip")
    naive = objective.unknown_as_negative_risk(f, labels) * 1.0
    assert abs(pu.item() - true.item()) < 0.01
    # naive: every non-labelled row negative, labelled rows positive
    assert abs(naive.item() - true.item()) > 0.05


def test_unknown_as_negative_baseline_treats_unknown_as_negative():
    f = torch.zeros(4, 1, dtype=torch.float64)
    y = torch.tensor([[1], [0], [0], [-1]], dtype=torch.float64)
    w = torch.ones(4, 1, dtype=torch.float64)
    assert abs(objective.unknown_as_negative_risk(f, y, w).item() - np.log(2)) < 1e-12


def test_reviewed_negative_loss_and_empty_case():
    f = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([[-1], [0], [-1]], dtype=torch.float64)
    assert abs(objective.reviewed_negative_loss(f, y).item() - (sp(1.0) + sp(3.0)) / 2) < 1e-12
    assert objective.reviewed_negative_loss(f, torch.zeros_like(y)).item() == 0.0


@pytest.mark.parametrize("what", ["label", "prior", "weights", "shape"])
def test_validation(what):
    f = torch.zeros(3, 2)
    y = torch.zeros(3, 2)
    pri = torch.tensor([0.3, 0.3])
    w = None
    if what == "label":
        y[0, 0] = 2
    elif what == "prior":
        pri = torch.tensor([0.0, 0.3])
    elif what == "weights":
        w = torch.full((3, 2), 1.5)
    elif what == "shape":
        y = torch.zeros(2, 2)
    with pytest.raises(PLNInputError):
        objective.nnpu_risk(f, y, pri, w)


def test_sparsity_penalty_value(variant, world, sets):
    model = make_model(variant)
    idx, mask = sets
    co = model.coefficients(idx, mask, **world)
    expected = sum((oc.order - 1) * oc.m.abs().sum(1).mean().item() for oc in co.orders if oc.order >= 2)
    assert abs(objective.sparsity_penalty(co, 0.5).item() - 0.5 * expected) < 1e-9


def test_monotonicity_penalty_nonnegative_and_differentiable(world, sets):
    model = make_model("cp")
    idx, mask = sets
    g = torch.Generator().manual_seed(0)
    pen = objective.monotonicity_penalty(model, idx, mask, **world, generator=g)
    assert pen.item() >= 0
    pen.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()) or pen.item() == 0


def test_total_loss_backward_and_config_validation(world, sets):
    model = make_model("cp")
    idx, mask = sets
    labels = torch.tensor([[1, 0, 0, 0], [0, 1, -1, 0], [1, 0, 0, 1]], dtype=torch.float64)
    cfg = TrainConfig(kappa=1e-3, lambda_mono=0.1)
    pri = torch.full((4,), 0.3, dtype=torch.float64)
    out = objective.total_loss(model, cfg, idx, mask, labels, **world, priors=pri, generator=torch.Generator().manual_seed(0))
    out["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(PLNConfigError):
        TrainConfig(objective="bogus")
    with pytest.raises(PLNConfigError):
        MCNConfig(in_dim_skill=3, in_dim_comp=3, variant="x")
