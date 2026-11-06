"""SYNTHETIC data generator with a planted ground truth. NOT REAL DATA.

Purpose: controlled unit tests, debugging, and a well-specified recovery sanity check of the
Moebius Competency Network. Anything measured on this data says NOTHING about real learners,
real competencies, or real-world performance, and must always be labelled synthetic.

World model (invented): skills belong to clusters, clusters belong to domains, and a skill's
feature vector is its cluster centroid plus noise. A competency has one or more "recipes", each
a list of role clusters, spanning at least two domains overall. A skill set S truly supports the
competency iff some recipe has every role filled by at least one skill of that role's cluster.
Types: "conj2" (one recipe of 2 roles), "conj3" (one recipe of 3 roles) and "alt3" (two alternative
recipes of 3 roles each, i.e. two routes to the same competency). Annotators label a random fraction of true positives (+1, selected completely at
random); everything else is unknown (0), optionally with a fraction of true negatives
marked as reviewed negatives (-1). The generator is well-specified for the model class by
construction, so recovery here is a necessary sanity check, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

IS_SYNTHETIC = True


@dataclass
class SyntheticSpec:
    n_domains: int = 4
    clusters_per_domain: int = 3
    skills_per_cluster: int = 8
    feat_dim: int = 16
    feat_noise: float = 0.35
    n_comp3: int = 4
    n_comp2: int = 4
    n_alt3: int = 4
    n_sets: int = 8000
    size_min: int = 3
    size_max: int = 7
    p_planted: float = 0.5
    annotate_prob: float = 0.5
    reviewed_neg_frac: float = 0.0
    holdout_skill_frac: float = 0.15
    jaccard_group_threshold: float = 0.6
    audit_size: int = 600
    seed: int = 0


@dataclass
class SyntheticData:
    spec: SyntheticSpec
    skill_feats: torch.Tensor  # (N, F)
    comp_feats: torch.Tensor  # (M, F)
    idx: torch.Tensor  # (n, Lmax) long, padded with 0
    mask: torch.Tensor  # (n, Lmax) bool
    y_true: torch.Tensor  # (n, M) in {0,1}: planted ground truth
    labels: torch.Tensor  # (n, M) in {+1,-1,0}: what the learner is allowed to see
    skill_domains: torch.Tensor  # (N, D) bool
    skill_cluster: np.ndarray
    comp_recipes: List[List[List[int]]]  # per competency: list of recipes, each a list of role clusters
    comp_type: List[str]
    groups: np.ndarray
    split: Dict[str, np.ndarray]  # name -> row indices
    heldout_skills: np.ndarray
    priors: torch.Tensor  # (M,) audit-sample estimate of P(support) in the pool
    is_synthetic: bool = field(default=True)


def _roles_for(rng: np.random.Generator, n_roles: int, cluster_domain: np.ndarray) -> List[int]:
    n_clusters = len(cluster_domain)
    while True:
        roles = rng.choice(n_clusters, size=n_roles, replace=False).tolist()
        if len({int(cluster_domain[r]) for r in roles}) >= 2:
            return roles


def _group_components(sets01: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph with an edge when Jaccard(S_i, S_j) >= threshold."""
    n = sets01.shape[0]
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sizes = sets01.sum(1)
    A = sets01.astype(np.float32)
    for start in range(0, n, 1000):
        inter = A[start : start + 1000] @ A.T
        union = sizes[start : start + 1000, None] + sizes[None, :] - inter
        jac = inter / np.maximum(union, 1)
        rows, cols = np.nonzero(jac >= threshold)
        for r, c in zip(rows, cols):
            r += start
            if r < c:
                ra, rb = find(int(r)), find(int(c))
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def generate(spec: SyntheticSpec) -> SyntheticData:
    rng = np.random.default_rng(spec.seed)
    n_clusters = spec.n_domains * spec.clusters_per_domain
    N = n_clusters * spec.skills_per_cluster
    cluster_domain = np.repeat(np.arange(spec.n_domains), spec.clusters_per_domain)
    skill_cluster = np.repeat(np.arange(n_clusters), spec.skills_per_cluster)
    centroids = rng.normal(size=(n_clusters, spec.feat_dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    noise = rng.normal(scale=spec.feat_noise / np.sqrt(spec.feat_dim), size=(N, spec.feat_dim))
    dom_onehot = np.eye(spec.n_domains)[cluster_domain[skill_cluster]]
    skill_feats = np.concatenate([centroids[skill_cluster] + noise, dom_onehot], axis=1)

    comp_type = ["conj3"] * spec.n_comp3 + ["conj2"] * spec.n_comp2 + ["alt3"] * spec.n_alt3
    comp_recipes: List[List[List[int]]] = []
    for t in comp_type:
        if t == "alt3":
            six = _roles_for(rng, 6, cluster_domain)
            comp_recipes.append([six[:3], six[3:]])
        else:
            comp_recipes.append([_roles_for(rng, 3 if t == "conj3" else 2, cluster_domain)])
    M = len(comp_type)
    all_clusters = [sorted({r for rec in recs for r in rec}) for recs in comp_recipes]
    comp_feats = np.stack([
        np.concatenate([centroids[r].sum(0), np.eye(spec.n_domains)[cluster_domain[r]].mean(0)]) for r in all_clusters])

    members_by_cluster = [np.nonzero(skill_cluster == c)[0] for c in range(n_clusters)]
    sets: List[List[int]] = []
    for _ in range(spec.n_sets):
        size = int(rng.integers(spec.size_min, spec.size_max + 1))
        chosen: List[int] = []
        if rng.random() < spec.p_planted:
            recipes = comp_recipes[int(rng.integers(M))]
            roles = recipes[int(rng.integers(len(recipes)))]
            chosen = [int(rng.choice(members_by_cluster[r])) for r in roles]
        while len(chosen) < size:
            s = int(rng.integers(N))
            if s not in chosen:
                chosen.append(s)
        rng.shuffle(chosen)
        sets.append(chosen)

    Lmax = spec.size_max
    idx = np.zeros((spec.n_sets, Lmax), dtype=np.int64)
    mask = np.zeros((spec.n_sets, Lmax), dtype=bool)
    sets01 = np.zeros((spec.n_sets, N), dtype=np.float32)
    for i, s in enumerate(sets):
        idx[i, : len(s)] = s
        mask[i, : len(s)] = True
        sets01[i, s] = 1.0

    y_true = np.zeros((spec.n_sets, M), dtype=np.float32)
    for c, recipes in enumerate(comp_recipes):
        sat = [np.stack([sets01[:, members_by_cluster[r]].sum(1) > 0 for r in roles], axis=1).all(axis=1) for roles in recipes]
        y_true[:, c] = np.any(np.stack(sat, axis=1), axis=1)

    annotated = (y_true == 1) & (rng.random(y_true.shape) < spec.annotate_prob)
    reviewed = (y_true == 0) & (rng.random(y_true.shape) < spec.reviewed_neg_frac)
    labels = np.where(annotated, 1, np.where(reviewed, -1, 0)).astype(np.float32)

    groups = _group_components(sets01, spec.jaccard_group_threshold)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_tr, n_va = int(0.6 * len(uniq)), int(0.2 * len(uniq))
    which = {g: ("train" if k < n_tr else "val" if k < n_tr + n_va else "test") for k, g in enumerate(uniq)}
    part = np.array([which[g] for g in groups])
    heldout = rng.choice(N, size=int(spec.holdout_skill_frac * N), replace=False)
    has_held = sets01[:, heldout].sum(1) > 0
    split = {
        "train": np.nonzero((part == "train") & ~has_held)[0],
        "val": np.nonzero((part == "val") & ~has_held)[0],
        "test_seen": np.nonzero((part == "test") & ~has_held)[0],
        "test_unseen": np.nonzero((part == "test") & has_held)[0],
    }
    pool = np.concatenate([split["train"], split["val"]])
    audit = rng.choice(pool, size=min(spec.audit_size, len(pool)), replace=False)
    priors = np.clip(y_true[audit].mean(0), 0.01, 0.99)

    domains_bool = np.eye(spec.n_domains, dtype=bool)[cluster_domain[skill_cluster]]
    return SyntheticData(
        spec=spec, skill_feats=torch.tensor(skill_feats, dtype=torch.float32), comp_feats=torch.tensor(comp_feats, dtype=torch.float32),
        idx=torch.tensor(idx), mask=torch.tensor(mask), y_true=torch.tensor(y_true), labels=torch.tensor(labels),
        skill_domains=torch.tensor(domains_bool), skill_cluster=skill_cluster, comp_recipes=comp_recipes, comp_type=comp_type,
        groups=groups, split=split, heldout_skills=heldout, priors=torch.tensor(priors, dtype=torch.float32))
