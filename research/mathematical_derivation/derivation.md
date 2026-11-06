# Stage 4: Mathematical specification of the selected algorithm (Candidate A)

Status: proposed, awaiting approval. Date: 2026-11-06.
Working name: **MCN** (Möbius Competency Network). The name is descriptive, not a claim.

Reading guide. Section 0 gives conventions. Sections 1 to 5 define the objects. Section 6 is the mechanism. Sections 7 and 8 define prediction and readiness. Section 9 states Problems A to D. Section 10 is a hand-worked example. Sections 11 to 15 give the objective, optimisation, inference, uncertainty and updating. Section 16 maps every claimed identity to a numeric check. Section 17 lists assumptions and failure modes.

**Borrowed foundations (documented per rule 15).** Möbius representation, k-additive set functions, Shapley and pair interaction indices: Grabisch 1997, Grabisch and Roubens 1999. ANOVA-kernel dynamic programme for CP-parameterised interactions: Blondel et al. 2016. Functional-ANOVA purification: Lengerich et al. 2020. Strong heredity: Bien et al. 2013. Unbiased and non-negative PU risk: du Plessis et al. and Kiryo et al. 2017 (the former cited from memory), Elkan and Noto 2008. Probit approximation of the logistic-Gaussian integral: MacKay 1992 (cited from memory). Efron-Stein / Hoeffding orthogonal expansion for independent inputs (standard). Beta-Bernoulli updating (standard). Everything specific to PLN (relevance gate, embedding-generated coefficients, three-state labels, integration index, readiness semantics, tiers) is a **design proposal**, not a verified contribution.

---

## 0. Conventions

[N] = {1,...,N}. For a set T, |T| is its size. Sums over "T within S" include only non-empty T unless stated. sigma(z) = 1/(1+e^{-z}). All quantities are real unless stated. "Support" is a modelled relation; "readiness" is a prediction; neither is demonstrated competence.

## 1. Input space

- Skills K = {s_1..s_N}, competencies C = {c_1..c_M}, domains D (initially CS, Mathematics and Statistics, Business, Design, Data Science), domain map delta: K -> 2^D (a skill may belong to more than one domain). For a set T, dom(T) = union of delta(s) over s in T.
- **Combination instance** i = (S_i, c_i, l_i, w_i, prov_i): S_i within K with 1 <= |S_i| <= n_max (default 12); label l_i in {+1, -1, 0}: +1 annotated positive, -1 expert-reviewed negative, 0 unknown; w_i in (0,1] evidence reliability; prov_i provenance (source ids, annotator ids, evidence type in {expert, occupational-association, curricular, learner-data}). An (S, c) pair with no record is unknown, never negative.
- Presence vector x_S in {0,1}^N with x_S[s] = 1 iff s in S. Mastery vector mu in [0,1]^N (section 4).
- The full dataset is unordered; split into train/validation/test by **groups** (section 12.3).

## 2. Skill representation

Each skill s has a record: name, description, domain(s), difficulty level, expected duration, assessment criteria, provenance. Embedding

  e_s = W_e [ tau(text_s) ; 1_{delta(s)} ; phi(difficulty_s) ] in R^d,

where tau is a **frozen** pre-trained sentence encoder (choice deferred), 1_delta the multi-hot domain vector, phi a small difficulty featuriser, W_e learned. **No free per-skill ID embedding** is used in any evaluation that includes unseen skills; an "ID" ablation is allowed but is excluded from those evaluations. Relationships between skills (prerequisite, part-of, applied-in, complements, ...) live in the ontology graph (Module 1) with a confidence and provenance each; they are inputs to the planner and to embedding features only in the ways restricted by section 11.6.

## 3. Competency representation

A competency c has a description text, a domain profile, and embedding eps_c = W_eps tau(text_c) in R^d. A competency embedding lets the model score a competency **not seen in training** from its description alone (an experimental question, not an assumption).

## 4. Learner-state representation

Learner u at time t has state L_u(t) = (mu_u(t), alpha_u(t), beta_u(t), exposure_u(t), goals_u, constraints_u).

- **Evidence** records epsilon = (s, t_e, y, q): skill s, time t_e, scored outcome y in [0,1], evidence reliability q in [0,1] by type (proctored or rubric-scored assessment and project work high; self-report low and capped; **course completion and activity counts have q = 0** and only update exposure_u). No feature derived from speed or frequency of activity enters mastery.
- **Mastery posterior** per skill (decayed Beta-Bernoulli, a baseline Module 3 model):
  alpha_s(t) = alpha_0 + sum_e q_e y_e exp(-lambda_s (t - t_e)),
  beta_s(t)  = beta_0  + sum_e q_e (1 - y_e) exp(-lambda_s (t - t_e)),
  mu_s(t) = alpha_s / (alpha_s + beta_s).
  Decay shrinks both counts toward the prior, so forgetting widens uncertainty. lambda_s is fitted per domain on held-out retention data, or fixed to a conservative value and flagged. This is not a claim of novelty; DKT/transformer KT and FKGRec-style forgetting are the Module 3 baselines.
- **Latent mastery.** X_s in {0,1}: "the learner can reliably demonstrate skill s". The model uses mu_s = P(X_s = 1 | evidence) and assumes the X_s are independent given evidence (assumption A2; prerequisite correlation is ignored in v1).

## 5. Cross-domain relationship representation

Typed directed relations R = {prerequisite, part_of, applied_in, required_for, complements, assessed_by, contributes_to}, each edge with (confidence, provenance, validation status in {expert-validated, model-suggested, source-asserted}). Only expert-validated prerequisite edges can be hard planner constraints; model-suggested edges are advisory. Co-occurrence in O\*NET or ESCO is recorded as `source-asserted association` and never as prerequisite.

**Cross-domain mass and integration index** (defined, section 7.3): the share of a score's absolute coefficient mass carried by interaction groups T with |dom(T)| >= 2.

## 6. The higher-order interaction mechanism

### 6.1 Support function

For competency c, the pre-activation (support logit) on a presence vector x in [0,1]^N is the degree-k multilinear polynomial

  f_c(x) = b_c + sum_{j=1..k} sum_{T within K, |T| = j} m_c(T) prod_{s in T} x_s.   (1)

For a binary set S, (1) is f_c(S) = b_c + sum over T within S, |T| <= k, of m_c(T), i.e. a **k-additive set function in Möbius form** (Grabisch 1997). Interpretation: m_c({s}) is the individual contribution of s; m_c({s,t}) is the extra (synergy if positive, redundancy if negative) from having s and t together beyond their individual contributions; m_c(T) with |T| = j is the extra from having all of T beyond everything that its proper subsets already explain. b_c is the support logit of the empty set, which acts as a base rate.

### 6.2 Coefficient generator (what is learned)

  m_c(T) = u_c(T) h_{|T|}(eps_c, {e_s}_{s in T}),   u_c(T) = prod_{s in T} a_{c,s},   a_{c,s} = sigma(<q_c, e_s> + beta_0).   (2)

- **Relevance gate** a_{c,s} in (0,1) with q_c = W_q eps_c: an interaction can be large only if every member skill is relevant to c (a soft strong-heredity prior; the gate is on skills, not on lower-order terms).
- **Variant CP** (default; scalable): h_j = sum_{r=1..R} lambda_{j,r}(eps_c) prod_{s in T} z_{s,r}, with z_s = W_z e_s in R^R and lambda_j(eps_c) = V_j eps_c in R^R. Gating can be absorbed as z~_{s,r} = a_{c,s} z_{s,r}.
- **Variant DS** (more expressive): h_j = rho_j( [eps_c ; sum_{s in T} psi_j(e_s)] ), a Deep-Sets coefficient (Zaheer et al. 2017) with MLP rho_j: R^{2d} -> R.
- Bias b_c = <w_b, eps_c> + b_0.

Because (2) depends on skills only through embeddings and on c only through eps_c, **unseen skills, sets and competencies get coefficients without retraining**.

Honest relation to prior art: with variant CP and no gate, (1) is a higher-order factorization machine with competency-conditioned weights (Blondel et al. 2016). The semantic layer in sections 6.3 to 8 (exact decomposition, mastery propagation, tiers, PU training) is what is new at most. Whether that is a research contribution is decided by the Stage 7 experiments, not by this document.

### 6.3 Exact properties (each verified numerically, section 16)

All identities below hold for the polynomial (1) with coefficients m. Write F_j(S) = sum over T within S with |T| = j of m(T).

**P1 (uniqueness, inversion).** Every v: 2^K -> R has a unique m with v(S) = sum_{T within S} m(T) (with m(empty) included), and m(T) = sum_{U within T} (-1)^{|T|-|U|} v(U). v is k-additive iff m(T) = 0 for |T| > k. Check: C1.

**P2 (exact order decomposition).** f_c(S) = b_c + F_1(S) + ... + F_k(S). Individual, pairwise and higher-order contributions are therefore exact and sum to the score. Check: C2. Computational use: report (F_1, F_2, F_{>=3}) with every prediction.

**P3 (Shapley and pair interaction from m).** For the game v(S) = f_c(S) - f_c(empty):
  phi_s(S) = sum_{T within S, T contains s} m(T)/|T|,
  I_{st}(S) = sum_{T within S, T contains {s,t}} m(T)/(|T| - 1).
Efficiency: sum_s phi_s(S) = f_c(S) - b_c. Check: C3 (brute-force Shapley and pair interaction match to 1e-15). These are the explanation quantities: per-skill credit and per-pair synergy, computed from the model's own coefficients, not a post-hoc fit.

**P4 (marginal completion value).** For s not in S: f_c(S ∪ {s}) - f_c(S) = sum_{T within S} m_c(T ∪ {s}) (with T = empty allowed, giving m_c({s})). Ranking s by this gives "which missing skill would add the most support". Check: C7, C7d.

**P5 (mastery propagation).** Let X_s ~ Bernoulli(mu_s) independent. Because (1) is multilinear in x,
  E[f_c(X)] = f_c(mu),   (3)
  Var[f_c(X)] = sum over U with 1 <= |U| <= k of (d_U f_c(mu))^2 prod_{s in U} mu_s (1 - mu_s),
  d_U f_c(mu) = sum over T containing U of m_c(T) prod_{s in T \ U} mu_s.   (4)
Proof sketch. A multilinear f equals its Taylor expansion at mu exactly: f(x) = sum_U d_U f(mu) prod_{s in U}(x_s - mu_s). For U != U' there is an index in the symmetric difference whose centred factor appears once, so the cross expectation is zero by independence; squares contribute prod mu_s(1 - mu_s). Splitting (4) by |U| attributes readiness variance to order-j skill groups. Checks: C4 (enumeration agrees to 1e-10).

**P6 (order limit; refit versus truncation).** A conjunction of k+1 skills is not in the span of degree <= k monomials under any presence distribution with full support, so the best degree-k least-squares fit has strictly positive residual. In the Möbius basis the monomials are not orthogonal under a non-uniform presence distribution, so refitting at lower k moves mass between orders (truncating coefficients is not the same as refitting). Checks: C5 (residual fractions for AND3: 51.6% at ρ = 0.2, 14.3% at ρ = 0.5, 1.6% at ρ = 0.8), C5b (lower-order refit coefficients are non-zero although true lower-order Möbius coefficients are zero). Consequence: "higher-order matters" must be tested by **refitting at k = 2 and k = 3 and comparing held-out performance**, and by a purified energy diagnostic (section 7.4), never by reading coefficient magnitudes.

**Erratum E2 (found while running the Stage 5 synthetic check).** P6 is a statement about the pre-activation f. Ranking and classification depend on sigma(f) and a threshold, and a conjunction of skills is a *threshold function* in logit space, so a lower-order f can still rank AND-like sets well. On the synthetic task, refitting at k = 3 did not reliably beat k = 2 (`research/results/synthetic_planted.md`). So the H4 test (refit k = 3 against k = 2 on withheld combinations) can be uninformative on tasks whose structure is threshold-like. A task in which order 3 is necessary for *ranking* needs non-threshold structure; whether real competencies have it is an empirical question for PLN-CC, and the purified energy E_{>2} (section 7.4) should be read as a property of the fitted f, not as proof that order 3 improves predictions.

**P7 (monomial cost and the CP dynamic programme).** The number of monomials is sum_{j<=k} C(n, j) (n = 10, k = 3: 175; n = 15, k = 3: 575; C10). For variant CP, sum over T within S, |T| = j, of prod z_{s,r} is the elementary symmetric polynomial e_j of {z_{s,r}}, obtained by the DP a_t += a_{t-1} z for t = k..1 in O(|S| k R). Check: C6.

**P8 (monotonicity).** f(A ∪ {i}) - f(A) = sum_{T within A} m(T ∪ {i}) (P4), so f is monotone in presence iff these sums are >= 0 for all A, i. Sufficient: all m >= 0 (C7b). Redundant substitutes need a negative m on their union yet remain monotone if the singles dominate (C7c). Monotonicity is an optional soft constraint (section 11.4), not an assumed truth.

## 7. Competency prediction function

### 7.1 Probability and tiers

  p_c(S) = sigma( f_c(S) / tau_c ),   tau_c > 0 a temperature fitted post hoc (section 14).

**Support tiers** for each (S, c):
- Tier 1 *documented*: an annotated positive (S', c) exists in training data with S' within S. Carries provenance.
- Tier 2 *model-supported*: not documented, p_c(S) >= theta_c, ensemble spread (section 14) <= gamma, and the pair is in-distribution (its domain set was observed with any competency in training).
- Tier 3 *hypothesis*: p_c(S) >= theta_c but spread > gamma or out-of-distribution domain combination.
- Tier 0 *unsupported*: p_c(S) < theta_c.
Unknown labels are not rendered as Tier 0 negatives in evaluation; they are excluded or scored by the audit sample (section 11.1).

### 7.2 Multi-label output

The model scores all M competencies for a set S: p(S) = (p_c(S))_c, no softmax over competencies (a set can support several competencies or none). Cost per set: O(M |S| k R) for CP.

### 7.3 Cross-domain integration index

  J_c(S) = ( sum over T within S, |dom(T)| >= 2 of |m_c(T)| ) / ( sum over T within S of |m_c(T)| ).   (5)

Interpretation: the fraction of absolute interaction mass carried by groups that span at least two domains. It is a *descriptive* statistic of the fitted function. It is not evidence that cross-domain integration is useful; usefulness is tested in Stage 7 (within-domain versus cross-domain experiments). Check: C9 (toy value 0.5357).

### 7.4 Purified higher-order energy (synergy diagnostic)

Fix a reference presence distribution P_rho on the relevant skills R_c(S) (skills with a_{c,s} above a threshold, at most 12): either the product measure with presence probability rho, or the empirical distribution of combinations. Let Pi_{<=2} be the P_rho-weighted L2 projection of f_c onto monomials of degree <= 2. Define

  E_{>2}(c) = E_P[ (f_c - Pi_{<=2} f_c)^2 ] / Var_P[ f_c ].   (6)

This is the functional-ANOVA "pure" interaction energy at order >= 3 under P (Lengerich et al. 2020 for the idea). It depends on P, so report it for several rho and for the empirical distribution. Check: C5 computes exactly this ratio for AND3. It is a **diagnostic of the fitted function**; the falsifiable claim remains the held-out refit comparison.

### 7.5 Explanation record

For each prediction return: p_c(S), (F_1, F_2, F_{>=3}), top-phi_s skills, top I_{st} pairs, J_c(S), E_{>2}, missing skills ranked by P4 with gain, the Tier, provenance of any Tier-1 evidence, ensemble spread, and the mastery-variance split by order (4) when a learner is given. All are functions of m or of ensemble members; none is a separate explanation model.

## 8. Learner-readiness function

Given a learner state, **readiness** is the predicted probability that the learner's current skills form a sufficient foundation for c:

  rho_c(u) = E_{X ~ Bern(mu_u)} [ sigma( f_c(X) / tau_c ) ].   (7)

Using (3) and (4) and the probit approximation of the logistic-Gaussian integral (MacKay 1992, cited from memory):

  rho_c(u) ≈ sigma( f_c(mu_u) / ( tau_c sqrt(1 + pi V_c / (8 tau_c^2)) ) ),   V_c = Var[f_c(X)].   (8)

Exact enumeration is used when the relevant set has at most 16 skills. A credible interval is obtained by sampling mu_s ~ Beta(alpha_s, beta_s) and propagating, if second-order uncertainty is needed.

Semantics, to avoid conflation: rho_c is **not** mastery of c and not a measure of ability; it is the model's belief about skill-set sufficiency given the mastery evidence of the *component skills*. Mastery of c itself is estimated only from assessments of c.

Why not simply evaluate sigma(f_c(mu))? Because of Jensen: on the toy example (section 10) the naive value is 0.1726 while the exact expectation is 0.2202. Ignoring mastery uncertainty biases readiness downward where the logistic is convex (negative logit) and upward where it is concave.

## 9. Problem statements

**Problem A (competency discovery).** Given a combination S and the trained theta, return the Tier-labelled set of competencies {c : p_c(S) >= theta_c} with explanation records. Solved by sections 6 to 7, trained by section 11.

**Problem B (learner readiness).** Given L_u(t), return rho_c(u) with variance split by order. Solved by (7) to (8), no new parameters. *Training signal*: PLN-CC itself has no learner outcomes, so rho_c is a **deduced** quantity from the Problem A function and independently estimated mastery. It can be *tested* on real data only through a proxy: multi-skill items, where the "competency" is "answers item q correctly" and the skill set is the item's tagged skills, using ASSISTments-style data (item correctness given prior mastery estimates). This is the cognitive-diagnosis setting, where G-DINA and NeuralCD are the baselines, and it is evidence about *skill-combination structure for items*, not about competencies. Anything else about B (for example learner-conditioned versus learner-independent competency formation) can only be run on synthetic learners and must be marked synthetic.

**Problem C (cross-domain generalisation).** Evaluate p_c on (S, c) whose domain-pair, whole combination family, or competency was withheld from training. Mechanism that can support it: coefficients are generated from frozen embeddings (6.2); whether it does is the experiment. Success metric: performance gap between seen and withheld groups, with confidence intervals; compare to baselines that receive the same embeddings.

**Problem D (adaptive pathway generation).** Specified in section 15. Shares the Problem A function; has no trained parameters of its own in v1.

**Shared representation or separate objectives?** Shared: the function f_c serves A (binary inputs), B (probabilistic inputs) and the planner gain (D). Separate: the trained objective exists only for A (and an optional external-data likelihood for B); mastery is a separate Bayesian model; the planner is deterministic. This keeps components independently testable (rule 14).

## 10. Hand-worked example (invented numbers, not data)

Competency "business analytics"; skills 1 = programming (CS), 2 = statistics (Mathematics and Statistics), 3 = business domain knowledge (Business). Coefficients chosen for illustration: b = -3.0; m(1) = 0.3, m(2) = 0.2, m(3) = 0.1; m(12) = 0.4, m(13) = 0.1, m(23) = 0.2; m(123) = 1.5.

- f(empty) = -3.0; singles: -2.7, -2.8, -2.9; pairs: f(12) = -2.1, f(13) = -2.5, f(23) = -2.5; f(123) = -0.2.
- Order parts for S = {1,2,3}: F_1 = 0.60, F_2 = 0.70, F_3 = 1.50. Most of the lift appears only when all three are present, and that lift is not reproducible by pairwise terms (P6).
- Learner with mu = (0.9, 0.8, 0.5): f(mu) = -1.5670, Var f(X) = 1.1028, exact readiness (tau = 1) = 0.2202, probit approximation = 0.2127, naive sigma(f(mu)) = 0.1726.
- Marginal completion (P4) for S = {1,2}: adding skill 3 gives f(123) - f(12) = 1.9.
All numbers reproduced by `numeric_checks.py` check C11.

## 11. Training objective

### 11.1 Per-competency PU risk

Per competency c, partition training instances into positives P_c (l = +1), reviewed negatives N_c (l = -1) and unknown (l = 0). The risk below uses a **pool** U_c = all candidate sets for c, labelled or not (see erratum E1). Use logistic loss l(f, +1) = log(1 + e^{-f}), l(f, -1) = log(1 + e^{f}). Define empirical terms with reliability weights w_i on positives:

  R_P^+ = weighted mean over P_c of l(f, +1),  R_P^- = weighted mean over P_c of l(f, -1),  R_U^- = mean over the pool U_c of l(f, -1).

With class prior pi_c = P(S supports c) in the pool's sampling distribution, the **unbiased PU risk** is pi_c R_P^+ + R_U^- - pi_c R_P^-, and the **non-negative** variant used for training is

  R_c = pi_c R_P^+ + max( 0, R_U^- - pi_c R_P^- ).   (9)

Justification: unknown pairs are a mixture of unlabelled positives and true negatives; treating them as negatives is biased, which check C8 shows numerically (|bias| 0.187 for unknown-as-negative against 0.0001 for the PU risk on a synthetic case with a known prior). The clipping prevents the empirical risk going negative with flexible models (Kiryo et al. 2017).

**Erratum E1 (found while implementing Stage 5).** The first version of this section defined R_U^- as the mean over the *unknown* (l = 0) sets only. That is the right reference only in the case-control setting, where unlabelled sets are an independent sample from the marginal. In the single-sample (censoring) setting that this project has (labelled positives are a subset of the candidate pool), the unknown subset under-represents positives, so the estimator would be biased. Two valid corrections: (a) take U_c to be the whole pool, labelled sets included, with pi_c the prevalence of support in the pool (implemented, default); or (b) keep only the l = 0 sets and use the adjusted prior pi' = (1 - e) pi / (1 - e pi), with e the labelling frequency. The numeric check C8 (case-control) remains valid for its own setting; the censoring setting is tested in `tests/test_objective.py::test_pu_risk_is_unbiased_in_censoring_setting_and_naive_is_not`.

Assumption A1: labelled positives are selected at random among positives within strata (stratify by domain set and set size). Annotation of popular combinations violates this (SAR, Bekker and Davis 2020); a sensitivity analysis varying the propensity is part of the evaluation.

**Class prior and unbiased evaluation come from a random audit sample**: experts label a stratified uniform random sample of unknown (S, c) pairs. This gives pi_c, unbiased precision estimates, and the only valid set for calibration. Precision, calibration and AUPRC on PU-labelled data are otherwise not interpretable.

### 11.2 Reviewed negatives

Reviewed negatives are true negatives but a **selection-biased** sample (experts review hard cases). They enter as a separate term with weight lambda_N:  L_neg = mean over N_c of l(f, -1). They are not mixed into (9).

### 11.3 Parsimony prior

  L_sparse = kappa * sum over instances, sum over j >= 2, of (j - 1) * sum over T within S_i, |T| = j, of |m_{c}(T)|,

an l1 penalty increasing with order, so that order-j terms enter only when they improve held-out risk. Together with the relevance gate (2), this is the soft heredity prior.

### 11.4 Optional monotonicity penalty

  L_mono = mean over sampled (A, i, c) of max(0, -(f_c(A ∪ {i}) - f_c(A)))^2.

Included as a switch; the Stage 7 ablation decides whether it helps (some competencies may legitimately have skills that do not help).

### 11.5 Total and weighting

  L = sum over c of R_c + lambda_N L_neg + L_sparse + lambda_m L_mono.   (10)

Weights (lambda_N, kappa, lambda_m) are chosen on group-held-out validation folds by a logarithmic grid, reporting the full sensitivity grid. Learned uncertainty weighting and gradient-norm balancing are **not** used in v1 because their interaction with PU risk is unstudied. Calibration is not in the loss; it is applied post hoc (section 14). Relation reconstruction for Module 1 and mastery estimation for Module 3 are separate objectives with their own losses; they connect to (10) only through frozen embeddings and the mastery vector. Each added objective must earn its place by an ablation (rule: no automatic inclusion).

### 11.6 Preventing shortcuts

1. Groups: connected components of sets with Jaccard >= 0.6 (and of competencies in the same family) are never split across train/validation/test.
2. Label-propensity audit: train a classifier on (domain set, |S|) alone to separate labelled positives from unknown; it must be near chance, otherwise the unknown pool is not matched and the PU assumption is violated. Unknown candidates are generated to match the positive size and domain distributions.
3. Report size-only, domain-only and (where allowed) ID-only baselines next to every result.
4. The ontology's `contributes_to` edges touching a test pair are removed from every input and feature path. Prerequisite, part-of and similarity edges may be used if validated independently of the test labels.
5. Text-leakage audit: skill descriptions that contain a withheld competency's name are flagged and excluded from the corresponding evaluation.
6. Frozen text encoder; no label-derived features.

## 12. Optimisation procedure

12.1 Adam-type optimiser (AdamW), gradient clipping, fixed seeds for Python/NumPy/PyTorch, deterministic flags, CPU and GPU parity checks.
12.2 Mini-batches of sets; the model scores all M competencies per set; each competency's PU terms use only the label states present in the batch (an unannotated (S, c) is unknown, not negative). pi_c is fixed from the audit sample (or an Elkan-Noto estimate if no audit exists yet, flagged).
12.3 Splits: group-based (section 11.6.1); additionally *leave-domain-pair-out*, *leave-competency-out* and *leave-combination-family-out* splits for Problem C. Chronological splits for learner-side tasks.
12.4 Optional *order curriculum*: train k = 1 first, then add orders, as a bias toward lower order; ablated against joint training.
12.5 Model selection on group-held-out validation AUPRC against the audit sample where available; ensembles of E = 5 seeds.

**Training pseudocode.**
```
for epoch:
  for batch of sets {S_b} with label-state matrix L[b, c] in {+1, -1, 0}:
    z[b, c] = f_c(S_b)            # (1)-(2); CP: ANOVA DP, O(|S| k R)
    for c in C:
      R_c = nnPU( z[:, c][L[:, c] == +1], z[:, c][L[:, c] == 0], pi_c )   # (9)
    L_neg = mean softplus(z) over L == -1
    loss = sum_c R_c + lambda_N * L_neg + L_sparse(m) + lambda_m * L_mono
    step(AdamW, clip=1.0)
  validate on held-out groups (audit-sample AUPRC); early stop
fit tau_c on calibration fold (audit sample only); save ensemble member
```

## 13. Inference procedure

```
discover(S):                       # Problem A
  for c: compute f_c(S), parts F_j, phi, I, J, completions; p = sigmoid(f/tau_c)
  attach tier (section 7.1); return explanation record
readiness(learner):                # Problem B
  mu <- mastery model; for c in goals ∪ candidates:
     mean = f_c(mu); V = (4) split by order; rho = (8); attach spread across ensemble
candidates for a learner:         # no subset enumeration needed
  f_c(mu) uses all skills at once; gates a_{c,s} prune to relevant skills (<= 12 for exact variance)
```

## 14. Uncertainty estimation

Three kinds are kept separate in the output:
1. **Mastery uncertainty**: exact variance (4) and the order split; optional Beta sampling for a credible interval.
2. **Model (epistemic) uncertainty**: spread of f_c across an ensemble of E = 5 independently seeded models, used for the tier rule (spread <= gamma).
3. **Label and annotation uncertainty**: through reliability weights w_i and the sensitivity analysis of A1.
Calibration: per-competency temperature tau_c (or a shared tau) fitted by minimising the log loss on the audit-sample calibration fold; evaluate ECE with reliability diagrams **on audit-sample labels only**. Out-of-distribution flag: unseen domain combination or Jaccard distance to every training set above a threshold.

## 15. Problem D: constrained pathway planning

Inputs: learner mastery mu; goals G within C with weights w_g; validated prerequisite DAG (K, E_pre); resources r with skill s(r), effort e_r (hours), expected mastery after completion mu_s^+(r) (from resource metadata; unknown values carry an interval and a conservative default) and availability; budget B.

A pathway is a sequence pi = (r_1..r_L). Planning projection: after r_l, mu_{s(r_l)} <- max(mu_{s(r_l)}, mu^+(r_l)). **Hard constraints** (checked by an independent validator, never by the model): (i) for every step teaching s, every validated prerequisite s' of s has projected mu_{s'} >= theta_pre at that time; (ii) the pathway is acyclic and refers only to available resources; (iii) sum of efforts <= B. Objective on feasible pathways:

  J(pi) = sum_{g in G} w_g rho_g(mu^{pi}) - lambda_e * effort(pi) - lambda_L * L.   (11)

Search: beam search over feasible next resources scored by the multilinear gain, which is exact: the immediate gain of raising mu_s by delta is delta * d_{s} f_g(mu), by (4). A\* with an admissible bound is available only when f_g is monotone on the relevant set (verified by exhaustive check for sets up to 12 skills); otherwise the search is a beam search with **no optimality claim**, while validity is still guaranteed by the validator. Replanning after new evidence: update mu, re-run from the current state; adaptation efficiency is measured by recomputation time and plan edit distance. Planned mastery gains are projections; actual mu changes only through assessment evidence (section 4).

## 16. Numeric verification map

| Claim | Check | Result |
|---|---|---|
| P1 Möbius inversion, k-additivity | C1 | pass, error 1e-15 |
| P2 order decomposition | C2 | pass |
| P3 Shapley, pair interaction from m | C3 | pass, 1e-15 |
| P5 mean, variance, order split | C4 | pass, 1e-10 |
| P6 order limit and refit versus truncation | C5, C5b | pass (AND3 residual 51.6% / 14.3% / 1.6% at rho = 0.2 / 0.5 / 0.8) |
| P7 CP dynamic programme | C6 | pass |
| P4, P8 completion, monotonicity, motifs | C7 to C7d | pass |
| (9) PU risk versus unknown-as-negative | C8 | pass (|bias| 0.0001 vs 0.1872) |
| (5) integration index | C9 | pass (toy) |
| P7 monomial counts | C10 | tabulated |
| Section 10 worked example | C11 | pass |

Run with `python numeric_checks.py`; output in `numeric_checks_output.txt`. These checks verify algebra and a synthetic PU toy. They are not model training, not data, and not evidence of predictive value or novelty.

## 17. Assumptions, failure modes and open points

Assumptions: A1 selected-at-random labelled positives within strata. A2 independent mastery across skills. A3 frozen text embeddings carry the semantics needed for unseen skills and competencies. A4 a polynomial of degree at most 3 suffices (tested by k = 2, 3, 4 refits). A5 logit-additive link adequate (tested by comparing with a monotone link). A6 expert annotations of PLN-CC are reliable (inter-annotator agreement to be measured).

Failure modes to look for: the model learns label propensity instead of support (audit 11.6.2); identifiability: orders trade mass (P6, report purified and refit); calibration under PU (audit sample only); embeddings collapse domains; reading co-occurrence in O\*NET/ESCO as dependency; independence assumption understates variance for prerequisite chains; the decomposition costs accuracy against attention baselines.

Open points needing a decision: the frozen text encoder; default n_max and k; whether the audit sample can be funded (it is what makes precision and calibration meaningful); PLN-CC annotation protocol and expert panel; the domain list (Engineering dropped, Data Science added, section 3 of the inspection note).
