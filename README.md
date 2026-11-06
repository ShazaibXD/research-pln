# research-pln: Predictive Learning Network (PLN)

Research repository for PLN, the model of how skill combinations across domains support higher-order competencies, for the Polymath OS platform.

**Status: Stages 1 to 5 of 8.** The core mechanism (the Moebius Competency Network, MCN) is implemented and unit-tested. Modules for career alignment, the full planner, the API, real-data baselines and the manuscript are **not built yet**. The dataset PLN-CC does not exist yet, so there is no result on real data.

## What is and is not claimed

- Nothing here claims a novel, superior or validated algorithm. MCN builds on k-additive set functions, higher-order factorization machines and G-DINA. Its candidate contributions are unverified hypotheses with refutation criteria in `research/novelty_analysis/novelty_ledger.md`.
- "Original implementation" (code written for this repo) is not the same as "original algorithm" or "original research contribution". The ledger tracks the three separately.
- Everything measured so far is on **synthetic** data with a planted ground truth and says nothing about real learners. See `research/results/synthetic_planted.md`.
- Readiness is a prediction about whether a learner's current component skills form a sufficient foundation. It is not mastery of the competency and not a statement about ability.

## Layout

| Path | Content |
|---|---|
| `research/` | Stage 1 to 4 documents, literature review, derivation, numeric checks, novelty ledger, results |
| `pln/algorithm/` | `config.py`, `setfunc.py` (NumPy reference algebra), `interaction_model.py` (MCN), `explain.py`, `readiness.py`, `objective.py` |
| `pln/learner/mastery.py` | Baseline decayed Beta-Bernoulli mastery (Module 3 baseline) |
| `pln/data/synthetic.py` | **Synthetic** planted-structure generator (tests and sanity checks only) |
| `pln/training/` | `train.py` (loop, seeding, device handling), `checkpoint.py` |
| `pln/evaluation/metrics.py` | AP, AUC, Brier, log loss, ECE, precision@k |
| `experiments/` | Synthetic experiment scripts |
| `configs/mcn_default.yaml` | Default model and training configuration |
| `tests/` | 84+ tests: brute-force references, invariances, validation, reproducibility, checkpoint round-trip |

## Install and test

Python 3.9+ and PyTorch 2+. CPU works; CUDA is used if requested or available with `device: auto`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyyaml torch pytest
python -m pytest -q
```

The numeric identities of the derivation are checked separately:

```bash
python research/mathematical_derivation/numeric_checks.py
```

## Using the core model (synthetic or your own tensors)

```python
from pln.algorithm.config import MCNConfig
from pln.algorithm.interaction_model import MCN

model = MCN(MCNConfig(in_dim_skill=F, in_dim_comp=G, max_order=3, n_max=12))
# idx (B, L) long, mask (B, L) bool, skill_feats (N, F), comp_feats (M, G)
logits = model(idx, mask, skill_feats, comp_feats)               # support logits f_c(S)
logits, parts = model(idx, mask, skill_feats, comp_feats, return_parts=True)  # + order parts
coeffs = model.coefficients(idx, mask, skill_feats, comp_feats)  # explicit m_c(T) for explanations
```

Labels are `+1` annotated positive, `-1` expert-reviewed negative, `0` unknown. Unknown is never treated as negative by the default objective (`pln/algorithm/objective.py`).

## Next stages (not started)

6. Remaining PLN modules (ontology/graph layer, career alignment, planner, inference pipeline, API). 7. Baselines (logistic regression, MLP, GCN, GAT, HGT, hypergraph networks, HOFM, G-DINA-style, Set Transformer) and experiments on real data. 8. Documentation and manuscript.
