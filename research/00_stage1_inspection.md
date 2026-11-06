# Stage 1: Inspection of the existing project

Status: complete. Date: 2026-11-06.

## 1. What exists

| Source | Finding |
|---|---|
| `github.com/ShazaibXD/research-pln` | Empty repository (no commits, no default branch). Nothing was overwritten. |
| `~/Downloads/Neural network for polymathos.docx` | The PLN research proposal (about 6,400 words). This is the only PLN-specific source document found. |
| `~/Downloads/PolymathOS Curriculum Architecture.md/.pdf` | A curated learning-resource curriculum (Coding, AI/ML, Speaking, Design, ...). Contains no PLN model content. It may later serve as a source of *resources and step sequences*, never as ground truth for prerequisites or competencies. |

A local search of Downloads, Documents, Desktop and the working folders found no other PLN material.

## 2. What the existing proposal says

The proposal defines PLN as a hybrid system: a heterogeneous skill graph (HGT or relation-aware GAT), a transformer or DKT learner-state encoder, a multi-task head (next skill, mastery forecast, relation discovery), and a deterministic planner (topological sort plus A\*/beam search with hard prerequisite constraints). It adds four research questions on pathway quality, five hypotheses, a 24-week schedule, a three-layer data plan (public data, a PLN skill graph, synthetic learners) and a risk list.

Points that carry over unchanged into the new brief:

- Neural models learn and predict; explicit graph and planning algorithms construct and validate pathways. An LLM does not write the roadmap.
- Typed, directed, provenance-carrying relations. Do not collapse them into one `CONNECTED_TO`.
- Absent edges are not negatives. Use an explicit unknown state.
- Mastery means demonstrated evidence, not exposure or course completion. No inference of aptitude from speed or activity.
- Synthetic learner data may support controlled experiments but cannot prove educational effectiveness.
- Establish baselines first; do not promise arbitrary percentage gains.

The proposal itself names the closest competing work (KnowLP, FKGRec, hierarchical graph-attention KT) and says that GNN + Transformer + RL is not a sufficient novelty claim. The new brief agrees and goes further.

## 3. Gaps and inconsistencies to resolve

1. **Equations were lost.** The `.docx` stores its formulas as images, so a text export drops them. Only the surrounding prose and a few LaTeX fragments survive (for example the relation-aware attention and the combined loss). The exact original formulas for the learner state, forgetting decay and pathway objective are not recoverable from text. I did not guess them. If you want them preserved verbatim, please share the formulas or a PDF export.
2. **Different core mechanism.** The proposal's core is a heterogeneous GNN with a relation head. That is *pairwise* by construction. The new brief makes *higher-order, cross-domain competency discovery* the central contribution. The proposal's own research questions ("combination of competencies creates the foundation for another competency, even when not represented in the learning graphs") match the brief, but its proposed architecture does not yet implement that.
3. **Domain lists differ.**
   - Proposal: Computer Science, Mathematics, Business and Economics, Engineering, Design and Creative Technology.
   - Brief (PLN-CC): Computer Science, Mathematics and Statistics, Business, Design, Data Science.
   I treat the brief as current and PLN-CC as the first experiment. Engineering is dropped from the first experiment; Data Science is added. Please confirm.
4. **Dataset sources differ.** Proposal: EdNet, MOOCCube, ASSISTments. Brief: O\*NET, ESCO, EdNet, ASSISTments, OULAD, expert annotations. MOOCCube is not in the brief; I keep it as an optional concept-relation source.
5. **Unverified citations in the proposal.** Verified this session: KnowLP (AAAI 2026, DOI 10.1609/aaai.v40i17.38479) and FKGRec (Information Processing & Management 63(2), 104393, 2026, per DBLP). Not verified by me: "Deep Knowledge Tracing and Cognitive Load Estimation (2025)" and "Hierarchical Graph Attention Knowledge Tracing (2026)", both attributed to Scientific Reports. They stay flagged until checked.
6. **PLN-CC does not exist yet.** The brief describes it (canonical skills, multi-label annotations, provenance, expert validation, unknown labels) but no data, schema or annotation protocol is in the repo. This is the single largest risk to any claim of *original research contribution*; see `novelty_analysis/novelty_ledger.md`.
7. **Learner-outcome data for "readiness" is missing.** None of the listed public datasets records competency *formation* from skill combinations. The one real-data proxy is multi-skill item correctness (see `derivation.md` section 9). Anything beyond that must be marked synthetic.

## 4. Terminology fixed for this repository

- **Skill**: atomic, assessable capability. **Subskill**: part-of a skill. **Competency**: higher-order capability that may be supported by a *set* of skills. **Milestone**: a learning objective or deliverable. **Resource**: course, project, lesson or exercise. These are distinct node types.
- **Support** (or **foundation**) of a competency by a skill set: a modelled, annotated or predicted relation. It is **not** a demonstrated competence.
- **Readiness** of a learner for a competency: predicted foundation given the learner's mastery estimates. It is **not** mastery of the competency.
- **Mastery**: posterior belief, from assessment evidence, that the learner can reliably demonstrate a skill, with uncertainty and decay.
- **Unknown**: a label or relation nobody has judged. It is never a negative.

## 5. Stage gate

Stages 1 to 4 are delivered in this repository. No model, training or API code has been written, by design. Implementation (Stages 5 to 8) waits for your explicit approval of the selected algorithm in `algorithm_specification/candidate_algorithms.md` and the specification in `mathematical_derivation/derivation.md`.
