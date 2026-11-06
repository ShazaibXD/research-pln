# Stage 3: Three candidate algorithms and the selection

Status: proposed, awaiting your approval. Date: 2026-11-06.

Common task (Problem A in the brief). Input: a variable-size set S of skills (|S| <= n_max, drawn from many domains) and, optionally, a competency c with a text description. Output: a calibrated score that S is a sufficient foundation for c, plus an explanation derived from the score's own computation. Labels per (S, c) are one of: annotated positive, expert-reviewed negative, unknown.

Notation used below (full definitions in `mathematical_derivation/derivation.md`): skill embedding e_s in R^d; competency embedding eps_c in R^d; x in [0,1]^N is a vector of skill presence or mastery (x_s = 1 if s is in S).

---

## Candidate A: Embedding-conditioned k-additive support function

**Mathematical idea.** Every set function over skills has a unique polynomial form (the Möbius representation): f_c(S) = b_c + sum over T within S of m_c(T). Restricting to |T| <= k gives a k-additive function. Make the coefficients m_c(T) *functions of the embeddings* of the skills in T and of the competency, so the same parameters cover skills, sets and competencies not seen in training.

**Core mechanism.**
- Order-j coefficient: m_c(T) = u_c(T) * g_j(eps_c, {e_s : s in T}) for |T| = j, with a permutation-invariant g_j (two variants: CP/low-rank, and Deep-Sets).
- Relevance gate u_c(T) = product over s in T of sigmoid(<q_c, e_s> + beta_c): an interaction can only be non-zero if all its member skills are relevant to c (soft strong heredity).
- Score f_c(S) is evaluated on the *multilinear extension*, so replacing the 0/1 presence vector by mastery probabilities mu gives learner readiness, with exact mean and variance under independent Bernoulli mastery.
- Exact explanation: order-wise parts, Shapley values, pair interaction indices and "marginal completion value" of a missing skill are closed-form functions of m.

**Expected advantages.**
- "Is it higher-order?" becomes an experiment on one model family: refit with k = 1, 2, 3 and compare on held-out combinations.
- Missing-prerequisite effects (conjunction), redundancy (substitutes, negative interaction) and complementarity are all representable and verified on toy cases (`numeric_checks.py` C7).
- One function serves discovery, readiness, uncertainty propagation and planner gain.

**Limitations.**
- Monomials in S grow as sum of C(|S|, j): 175 at |S| = 10, k = 3; 575 at |S| = 15 (C10). The CP variant avoids enumeration; the Deep-Sets variant does not.
- Logit-linear in monomials: strongly saturating or ordinal effects need the link function.
- Independent-mastery assumption for uncertainty propagation ignores prerequisite-induced correlation.
- Non-orthogonal basis: truncating to k does not equal refitting at k (C5b), so order claims must use refit and purified comparison.

**Complexity.** CP variant: O(|S| k R) per competency per set, by the ANOVA dynamic programme (C6). Deep-Sets variant: O(C(|S|, <=k) * cost of g). Variance: O(C(N_rel, <=k)) with N_rel the relevant skills.

**Relationship to existing methods.** The algebra is Grabisch's k-additive capacity (1997) with Shapley/interaction indices (1999). With CP coefficients and no gate, the model is a higher-order factorization machine (Blondel et al. 2016) with per-competency weights. The saturated version with free per-item parameters is G-DINA. Faith-Shap defines a similar polynomial but only post hoc.

**Potential novelty.** Low to moderate, and only in the *combination and use*: embedding-generated, relevance-gated coefficients; one function for discovery and mastery-propagated readiness; per-competency three-state PU training; cross-domain integration index. The algebraic core is borrowed. Status: unverified research hypothesis.

---

## Candidate B: Soft recipe logic (differentiable AND-of-bottlenecks, OR over recipes)

**Mathematical idea.** A competency is supported if *any* of several "recipes" is satisfied; a recipe is satisfied only if *all* its weighted skills are present. Non-compensatory by construction.

**Core mechanism.**
- Recipe activation: a_r(x) = product over s of (1 - w_{r,s} (1 - x_s)), where w_{r,s} in [0,1] is skill importance in recipe r.
- Competency support: p_c(x) = 1 - product over r of (1 - theta_r a_r(x)) (noisy-OR over recipes).
- w_{r,s} = sigmoid(<q_{c,r}, e_s>) so recipes generalise to unseen skills; R recipes per competency.

**Expected advantages.** Directly interpretable ("needs probability AND linear algebra AND code, or alternatively ..."); naturally handles missing prerequisites and alternative routes; cheap: O(R |S|).

**Limitations.**
- No decomposition into order-j effects, so "is it higher-order" cannot be tested by a dial.
- AND via product saturates; partial credit and pure complements (zero solo value, large joint value) are awkward to express without many recipes.
- Negative interactions beyond OR-redundancy are not expressible.
- Number of recipes R is a capacity knob that confounds the order question.

**Complexity.** O(R |S|) per competency.

**Relationship to existing methods.** DINA/DINO and noisy-OR networks in psychometrics and graphical models; differentiable rule learning in neuro-symbolic work (cited from memory, not verified here).

**Potential novelty.** Low. It is a re-parameterisation of established logical aggregation with embeddings.

**Formal relation to A.** On binary inputs, an AND recipe is a single monomial, and the noisy-OR of two recipes a1, a2 is a1 + a2 - a1 a2, which has a negative coefficient on the union monomial. So every candidate-B model is a (sparse, constrained) member of A's function class when k is at least the largest recipe size.

---

## Candidate C: Typed hypernode cross-attention

**Mathematical idea.** Treat a competency as a *latent hypernode* that attends over the elements of a skill set; typed incidence (skill domain, relation type) biases the attention. Pooling-by-multihead-attention with competency queries as seeds.

**Core mechanism.**
- alpha_{c,s} = softmax_s( <W_Q eps_c, W_K e_s> / sqrt(d) + b_{type(s), c} ).
- z_c = sum_s alpha_{c,s} W_V e_s, then an MLP head. Optionally stack a set-attention block over S before pooling.

**Expected advantages.** Likely the strongest raw predictive accuracy and the simplest to implement; handles unseen skills through embeddings; scales as O(|S| M d).

**Limitations.**
- No exact contribution decomposition: attention weights are not additive effects, and explanations built on them risk being unfaithful.
- "Higher-order versus pairwise" cannot be isolated: depth and width change interaction order implicitly.
- Interaction structure is entangled with parameter count, which the brief asks us to disentangle.

**Complexity.** O(|S| M d) per set.

**Relationship to existing methods.** Set Transformer (PMA with seed vectors), AllSet (multiset functions with Transformer blocks), Hyper-SAGNN (attention over node tuples).

**Potential novelty.** Very low as an algorithm. Its role is the strongest honest *control*.

---

## Comparison

| Criterion | A: k-additive | B: soft recipes | C: hypernode attention |
|---|---|---|---|
| Isolates higher-order effect (dial k) | **Yes** | No | No |
| Exact order-wise, Shapley, interaction explanations | **Yes (closed form)** | Partial (recipe view) | No (attention only) |
| Missing-prerequisite / redundancy / complement | **All three, verified on toy** | Conjunction, redundancy | Learned implicitly |
| Unseen skills / combos / competencies | Yes (embeddings) | Yes | Yes |
| Unknown labels supported by the objective | Yes (PU per competency) | Yes | Yes |
| Learner readiness with exact uncertainty propagation | **Yes (multilinear)** | Yes (product form, no order split) | No |
| Expected raw accuracy | Medium | Medium | **High** |
| Complexity | O(|S| k R) CP; O(C(|S|,<=k)) general | O(R|S|) | O(|S| M d) |
| Algorithmic novelty | Low to moderate (borrowed core, new use) | Low | Very low |
| Falsifiable "higher-order helps" claim | **Direct** | Indirect | Indirect |
| Main risk | Reads as HOFM/G-DINA applied to skills | Capacity confound | Not a contribution |

## Selection

**Select Candidate A.** The reasons are about testability, not sophistication:

1. It is the only candidate where the question the brief centres on, "does a higher-order mechanism contribute beyond pairwise?", is answered inside one model family by changing k and refitting.
2. It strictly contains Candidate B on binary inputs, so B's structure can be recovered from A with sparsity, and A's explanations are exact.
3. It gives one function for Problems A, B and D without a separate learner network for readiness.

**What would make me abandon or revise A.**
- Refit at k = 3 does not beat k = 2 on group-held-out combinations beyond confidence intervals on PLN-CC (then there is no higher-order contribution to claim).
- A does not beat a CP-only higher-order FM baseline (then the contribution is not beyond HOFM).
- C (Set Transformer / AllSet) beats A by a large margin on held-out combinations (then the decomposability costs too much accuracy; report this and consider A only as an explanation layer).
- A prior-art check in the "not yet searched" list of `literature_review.md` finds the combination already published.

**Baselines carried forward from this stage (Stage 7):** logistic regression, MLP, GCN, GAT, HGT, HGNN/HyperGCN/AllSet/Hyper-SAGNN, a CP higher-order FM, a G-DINA-style model with embedding-generated parameters, Set Transformer (Candidate C), and size-only and domain-only shortcut baselines.
