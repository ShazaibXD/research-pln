"""Tests on SYNTHETIC data only. Nothing here says anything about real learners."""
import os

import numpy as np
import pytest
import torch

from pln.algorithm.config import MCNConfig, TrainConfig, load_yaml
from pln.algorithm.interaction_model import MCN
from pln.data.synthetic import SyntheticSpec, generate
from pln.errors import PLNConfigError, PLNInputError
from pln.evaluation import metrics
from pln.training.checkpoint import load_checkpoint, save_checkpoint
from pln.training.train import SetTensors, fit, resolve_device

pytestmark = pytest.mark.synthetic

SPEC = SyntheticSpec(n_sets=700, audit_size=100, reviewed_neg_frac=0.05, seed=1)


@pytest.fixture(scope="module")
def data():
    return generate(SPEC)


def test_generator_is_deterministic():
    a, b = generate(SPEC), generate(SPEC)
    assert torch.equal(a.idx, b.idx) and torch.equal(a.labels, b.labels) and np.array_equal(a.groups, b.groups)


def test_labels_never_contradict_truth_and_unknown_is_not_negative(data):
    y, lab = data.y_true, data.labels
    assert (y[lab == 1] == 1).all()
    assert (y[lab == -1] == 0).all()
    assert ((lab == 0) & (y == 1)).any()  # some true positives are unknown, and must stay unknown (not negative)
    frac = ((lab == 1).sum() / y.sum()).item()
    assert abs(frac - SPEC.annotate_prob) < 0.1


def test_splits_are_disjoint_group_pure_and_hold_out_skills(data):
    sp = data.split
    names = list(sp)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not set(sp[names[i]]) & set(sp[names[j]])
    g = {n: set(data.groups[sp[n]]) for n in names}
    assert not g["train"] & g["val"] and not g["train"] & g["test_seen"] and not g["val"] & g["test_unseen"]
    held = set(data.heldout_skills.tolist())
    has = lambda rows: np.array([bool(held & set(data.idx[r][data.mask[r]].tolist())) for r in rows])
    assert not has(sp["train"]).any() and not has(sp["val"]).any()
    assert has(sp["test_unseen"]).all() and not has(sp["test_seen"]).any()


def test_no_near_duplicate_crosses_train_and_test(data):
    sets = [set(data.idx[r][data.mask[r]].tolist()) for r in range(data.idx.shape[0])]
    tr, te = data.split["train"][:150], np.concatenate([data.split["test_seen"], data.split["test_unseen"]])[:150]
    for i in tr:
        for j in te:
            jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            assert jac < SPEC.jaccard_group_threshold


def test_priors_and_shapes(data):
    assert ((data.priors > 0) & (data.priors < 1)).all()
    assert data.priors.shape == (len(data.comp_type),) and data.is_synthetic


def _fit(data, seed, epochs=3, **mkw):
    sp = data.split
    mk = lambda n: SetTensors(data.idx[sp[n]], data.mask[sp[n]], data.labels[sp[n]])
    mcfg = MCNConfig(in_dim_skill=data.skill_feats.shape[1], in_dim_comp=data.comp_feats.shape[1], max_order=2, n_max=7, **mkw)
    torch.manual_seed(seed)
    model = MCN(mcfg)
    yv = data.y_true[sp["val"]]
    res = fit(model, mk("train"), mk("val"), data.skill_feats, data.comp_feats, data.priors,
              TrainConfig(seed=seed, epochs=epochs, batch_size=64),
              lambda l, _: np.nanmean([metrics.average_precision(yv[:, c].numpy(), l[:, c].numpy()) for c in range(yv.shape[1])]))
    return model, res


def test_training_is_reproducible_and_seed_sensitive(data):
    m1, r1 = _fit(data, 0)
    m2, r2 = _fit(data, 0)
    m3, _ = _fit(data, 1)
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    assert r1.history == r2.history
    assert any(not torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m3.state_dict().values()))


def test_training_reduces_loss_and_beats_prevalence(data):
    model, res = _fit(data, 0, epochs=15)
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    prevalence = data.y_true[data.split["val"]].mean().item()
    assert res.best_score > 1.5 * prevalence  # sanity only: the planted task is well-specified for this model class


def test_checkpoint_roundtrip_and_errors(data, tmp_path):
    model, _ = _fit(data, 0)
    path = str(tmp_path / "ck" / "m.pt")
    save_checkpoint(path, model, TrainConfig(seed=3), extra={"note": "synthetic"})
    loaded, tcfg, extra = load_checkpoint(path)
    rows = data.split["val"][:20]
    a = model.eval()(data.idx[rows], data.mask[rows], data.skill_feats, data.comp_feats)
    b = loaded.eval()(data.idx[rows], data.mask[rows], data.skill_feats, data.comp_feats)
    assert torch.allclose(a, b) and tcfg.seed == 3 and extra["note"] == "synthetic"
    assert not os.path.exists(path + ".tmp")
    with pytest.raises(PLNInputError):
        load_checkpoint(str(tmp_path / "missing.pt"))
    torch.save({"format_version": 99}, str(tmp_path / "bad.pt"))
    with pytest.raises(PLNInputError):
        load_checkpoint(str(tmp_path / "bad.pt"))


def test_default_yaml_config_loads():
    mcfg, tcfg = load_yaml(os.path.join(os.path.dirname(__file__), "..", "configs", "mcn_default.yaml"))
    assert mcfg.max_order == 3 and tcfg.objective == "nnpu"
    with pytest.raises(PLNConfigError):
        MCNConfig.from_dict({"in_dim_skill": 3, "in_dim_comp": 3, "bogus": 1})


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without CUDA")
def test_cuda_request_without_cuda_is_a_clear_error():
    with pytest.raises(PLNInputError):
        resolve_device("cuda")
