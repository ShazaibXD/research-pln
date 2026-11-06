"""Baseline mastery posterior: time-decayed Beta-Bernoulli (derivation.md section 4).

This is a deliberately simple Module 3 baseline, not a claimed contribution. Rules enforced:
  * course completion and raw activity carry reliability 0: they only update an exposure count, never mastery;
  * self-report carries a low, capped reliability;
  * evidence from the future relative to the query time is rejected;
  * nothing derived from speed or frequency of activity is used.
Decay shrinks both pseudo-counts toward the prior, so forgetting widens uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from pln.errors import PLNInputError

RELIABILITY: Dict[str, float] = {
    "assessment": 1.0,  # rubric-scored or proctored assessment
    "project": 1.0,  # evaluated project work
    "self_report": 0.2,  # capped
    "completion": 0.0,  # exposure only
    "activity": 0.0,  # exposure only
}


@dataclass(frozen=True)
class Evidence:
    skill: int
    time: float
    outcome: float  # scored outcome in [0, 1]
    kind: str


def posterior(evidence: Iterable[Evidence], n_skills: int, now: float, decay: Union[float, np.ndarray] = 0.01,
              prior: Tuple[float, float] = (1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, beta, exposure), each of shape (n_skills,).

    alpha_s = a0 + sum q y exp(-lambda (now - t)), beta_s = b0 + sum q (1-y) exp(-lambda (now - t)).
    mastery mean = alpha / (alpha + beta). `decay` is a scalar or per-skill array (>= 0).
    """
    a0, b0 = prior
    if a0 <= 0 or b0 <= 0:
        raise PLNInputError("prior pseudo-counts must be positive")
    lam = np.broadcast_to(np.asarray(decay, dtype=np.float64), (n_skills,))
    if (lam < 0).any():
        raise PLNInputError("decay must be non-negative")
    alpha = np.full(n_skills, a0, dtype=np.float64)
    beta = np.full(n_skills, b0, dtype=np.float64)
    exposure = np.zeros(n_skills)
    for ev in evidence:
        if ev.kind not in RELIABILITY:
            raise PLNInputError(f"unknown evidence kind {ev.kind!r}; expected one of {sorted(RELIABILITY)}")
        if not 0 <= ev.skill < n_skills:
            raise PLNInputError(f"skill index {ev.skill} out of range")
        if not 0.0 <= ev.outcome <= 1.0 or not math.isfinite(ev.outcome):
            raise PLNInputError("evidence outcome must be within [0, 1]")
        if ev.time > now:
            raise PLNInputError("evidence lies in the future relative to `now`")
        q = RELIABILITY[ev.kind]
        if q == 0.0:
            exposure[ev.skill] += 1.0
            continue
        w = q * math.exp(-lam[ev.skill] * (now - ev.time))
        alpha[ev.skill] += w * ev.outcome
        beta[ev.skill] += w * (1.0 - ev.outcome)
    # decay of the prior toward itself is a no-op; pseudo-counts above the prior shrink via w
    return alpha, beta, exposure


def mastery_mean(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return alpha / (alpha + beta)
