import itertools

import numpy as np
import pytest

from pln.errors import PLNInputError
from pln.evaluation import metrics
from pln.learner.mastery import Evidence, mastery_mean, posterior


def test_completion_does_not_change_mastery():
    ev = [Evidence(0, 1.0, 1.0, "completion"), Evidence(0, 2.0, 1.0, "activity")]
    a, b, expo = posterior(ev, 2, now=3.0)
    assert np.allclose(mastery_mean(a, b), 0.5) and expo[0] == 2


def test_assessment_raises_mastery_and_self_report_is_capped():
    a1, b1, _ = posterior([Evidence(0, 0.0, 1.0, "assessment")] * 3, 1, now=0.0)
    a2, b2, _ = posterior([Evidence(0, 0.0, 1.0, "self_report")] * 3, 1, now=0.0)
    assert mastery_mean(a1, b1)[0] > mastery_mean(a2, b2)[0] > 0.5


def test_decay_returns_toward_prior_and_widens_uncertainty():
    ev = [Evidence(0, 0.0, 1.0, "assessment")] * 5
    var = lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1))
    a0, b0, _ = posterior(ev, 1, now=0.0, decay=0.1)
    a1, b1, _ = posterior(ev, 1, now=50.0, decay=0.1)
    assert mastery_mean(a1, b1)[0] < mastery_mean(a0, b0)[0]
    assert var(a1, b1)[0] > var(a0, b0)[0]


@pytest.mark.parametrize("ev", [Evidence(0, 5.0, 1.0, "assessment"), Evidence(0, 0.0, 1.5, "assessment"),
                                Evidence(0, 0.0, 1.0, "vibes"), Evidence(9, 0.0, 1.0, "assessment")])
def test_bad_evidence_rejected(ev):
    with pytest.raises(PLNInputError):
        posterior([ev], 2, now=1.0)


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = np.round(rng.normal(size=60) + y, 1)
    pos, neg = s[y == 1], s[y == 0]
    ref = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(metrics.roc_auc(y, s) - ref) < 1e-12


def test_average_precision_known_values():
    assert metrics.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)
    assert metrics.average_precision([1, 1, 0], [0.1, 0.1, 0.1]) == pytest.approx(2 / 3)  # all tied: precision = prevalence
    assert np.isnan(metrics.average_precision([0, 0], [0.1, 0.2]))


def test_calibration_metrics():
    y = np.array([0, 0, 1, 1])
    assert metrics.brier(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0
    assert metrics.expected_calibration_error(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)
    assert metrics.expected_calibration_error(y, np.array([0.9, 0.9, 0.1, 0.1])) == pytest.approx(0.9)  # by hand: 0.5*|0-0.9| + 0.5*|1-0.1|
    assert metrics.precision_at_k(y, np.array([0.1, 0.2, 0.9, 0.3]), 1) == 1.0
    with pytest.raises(PLNInputError):
        metrics.brier(y, np.array([0.0, 0.0, 1.0, 2.0]))
    with pytest.raises(PLNInputError):
        metrics.roc_auc([0, 2], [0.1, 0.2])
