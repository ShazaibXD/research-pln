# Novelty ledger

Status: Stage 2 to 4 snapshot. Date: 2026-11-06.

Three things are tracked separately because they are not interchangeable. A new codebase does not make a new algorithm, and a new algorithm is not a research contribution until an experiment shows it adds value.

| Layer | Meaning | How it is established | Status today |
|---|---|---|---|
| **Original algorithm** | The mathematical mechanism itself is new | Credible prior-art search finds nothing equivalent | **Not established.** The algebraic core is borrowed (Grabisch 1997/1999; HOFM 2016; G-DINA 2011). |
| **Original implementation** | Code written independently, not copied | Provenance of code, review, licence audit | **Not started.** No implementation exists; none will be written before approval. |
| **Original research contribution** | The mechanism is shown to add value over existing approaches | Pre-specified experiments on real, expert-annotated data with baselines, ablations and confidence intervals | **Not started.** PLN-CC does not exist yet, so no such experiment can run. |

## Established foundations (borrowed, documented)

| Item | Source |
|---|---|
| Möbius representation, k-additive set functions, inversion | Grabisch, Fuzzy Sets and Systems 92(2), 1997 |
| Shapley and pair interaction indices from Möbius coefficients | Grabisch and Roubens, Int. J. Game Theory 28(4), 1999 |
| Higher-order interactions with the ANOVA kernel and O(d m) DP | Blondel et al., NeurIPS 2016; Rendle, ICDM 2010 |
| Attribute interactions for skill combinations on items (saturated model) | de la Torre, Psychometrika 76(2), 2011 (G-DINA) |
| Permutation-invariant set networks | Zaheer et al. 2017; Lee et al. 2019 |
| PU risk, non-negative estimator, SAR | Elkan and Noto 2008; Kiryo et al. 2017; Bekker and Davis 2020 |
| Pure-interaction (functional-ANOVA) identifiability | Lengerich et al. 2020 |
| Strong heredity | Bien, Taylor, Tibshirani 2013 |
| Beta-Bernoulli mastery with time decay; probit approximation; Efron-Stein expansion | Standard techniques |

## Candidate contributions (all unverified research hypotheses)

| ID | Claim | Closest prior art | What would support it | What would refute it |
|---|---|---|---|---|
| H1 | Embedding-generated, relevance-gated Möbius coefficients let one k-additive function score unseen skills, sets and competencies, with exact order-wise explanations | HOFM with competency-conditioned weights; G-DINA with parameters tied to embeddings; NEUR-HCI (order 2) | Group-held-out gains over a CP higher-order FM and an embedding G-DINA, with the same inputs | No gain over HOFM or over Set Transformer / AllSet; or a prior-art hit in the unsearched list |
| H2 | A single multilinear function serves discovery (binary sets) and learner readiness (mastery probabilities) with exact variance propagation and an order split | Multilinear extension (standard); knowledge tracing + separate prediction heads | Better calibrated readiness on the multi-skill item proxy than propagating point mastery | Calibration no better than sigma(f(mu)); or no gain over NeuralCD / G-DINA on items |
| H3 | Per-competency PU risk with reviewed negatives and a random audit sample avoids the bias of unknown-as-negative and yields valid precision and calibration | PU learning literature | Audit-sample precision of PU model above unknown-as-negative baseline; propensity audit near chance | Equal or worse performance than naive; or strong SAR sensitivity |
| H4 | Refit at k = 3 beats k = 2 on withheld combinations, which identifies a real higher-order effect in cross-domain competency formation | The brief's own question; none found for educational competencies | Statistically reliable improvement on expert-annotated PLN-CC, with purified energy E_{>2} agreeing | No improvement; or improvement explained by parameter count (matched-capacity control) |
| H5 | Cross-domain integration index J_c and cross-domain generalisation are informative about competencies spanning domains | None found | Held-out domain-pair performance above shortcut baselines | Performance at shortcut baseline levels |

## Evidence log (updated at Stage 5)

| Date | Item | Evidence | Effect on status |
|---|---|---|---|
| 2026-11-06 | H4 (order 3 beats order 2) | Synthetic planted task, 5 seeds: no reliable gain (80 epochs: about 0; 200 epochs: +0.009 AP unseen, 2/5 seeds positive). Order 2 beats order 1 by about +0.31 AP in 5/5 seeds. See `research/results/synthetic_planted.md`. | **Not supported.** Synthetic only, so it cannot refute H4 on real data either; derivation erratum E2 explains why the test may be uninformative on threshold-like tasks. |
| 2026-11-06 | H3 (PU risk versus unknown-as-negative) | Synthetic: nnPU ahead by +0.011 to +0.034 AP, 4/5 to 5/5 seeds; calibration not run. | **Weakly consistent, not established.** Small effect, rank metric, priors from a truth-labelled audit sample. |
| 2026-11-06 | Implementation | Core mechanism written for this repo; 84 tests including brute-force references and invariances; mutation checks caught injected errors in 5 of 5 tried. | Original *implementation* status only. Does not change algorithm or contribution status. |

H1 and H2 are the ones most exposed to being "an application of existing models". If they fail, the honest outcome is a negative or an engineering-only result (a useful explainable scorer, not a new algorithm). H4 and H5 are the scientifically interesting claims and depend entirely on PLN-CC annotation quality.

## Rules this ledger enforces

- No sentence in the README, paper draft or API docs may say "novel", "superior" or "validated" about an item above unless the right-hand column of its row has been satisfied by a recorded experiment.
- Synthetic-data results are labelled synthetic and cannot satisfy any "what would support it" cell that says real or expert-annotated.
- The prior-art search list in `literature_review/literature_review.md` section 5 must be completed before H1 to H5 are upgraded from "hypothesis".
