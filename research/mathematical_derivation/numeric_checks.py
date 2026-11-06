"""Numerical sanity checks for the Stage-4 derivation (derivation.md).

Scope: these checks verify *algebraic identities and small worked examples*
claimed in derivation.md. They are NOT experiments, NOT model training and
NOT evidence for novelty or predictive performance. Everything here uses
synthetic toy numbers invented for the check.

Run:  python numeric_checks.py          (needs only numpy)
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import itertools
import math
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

RNG = np.random.default_rng(20261106)
RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------------------
# Set-function helpers. A subset of {0..n-1} is a bitmask int.
# ----------------------------------------------------------------------------

def popcount(x: int) -> int:
    return bin(x).count("1")


def submasks(mask: int):
    """All submasks of mask, including 0 and mask."""
    s = mask
    while True:
        yield s
        if s == 0:
            break
        s = (s - 1) & mask


def members(mask: int) -> List[int]:
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


def random_kadditive(n: int, k: int, scale: float = 1.0) -> Dict[int, float]:
    """Random Moebius coefficients m(T) for 1 <= |T| <= k."""
    return {T: float(RNG.normal(0, scale)) for T in range(1, 1 << n) if popcount(T) <= k}


def f_set(m: Dict[int, float], b: float, S: int) -> float:
    """k-additive set function: f(S) = b + sum_{T subset S} m(T)."""
    return b + sum(m.get(T, 0.0) for T in submasks(S))


def moebius_from_values(v: np.ndarray, n: int) -> np.ndarray:
    """m(T) = sum_{U subset T} (-1)^{|T|-|U|} v(U)."""
    m = np.zeros(1 << n)
    for T in range(1 << n):
        m[T] = sum((-1) ** (popcount(T) - popcount(U)) * v[U] for U in submasks(T))
    return m


def f_multilinear(m: Dict[int, float], b: float, x: np.ndarray) -> float:
    """Multilinear extension: f(x) = b + sum_T m(T) prod_{s in T} x_s."""
    return b + sum(c * np.prod([x[s] for s in members(T)]) for T, c in m.items())


# ----------------------------------------------------------------------------
# C1  Moebius inversion round trip and k-additivity recovery
# ----------------------------------------------------------------------------

def check_c1() -> None:
    n, k = 5, 3
    m = random_kadditive(n, k)
    b = 0.37
    v = np.array([f_set(m, b, S) for S in range(1 << n)])
    m_rec = moebius_from_values(v - v[0], n)  # remove constant: m(empty)=0 for the shifted game
    err_in = max(abs(m_rec[T] - m[T]) for T in m)
    err_hi = max((abs(m_rec[T]) for T in range(1, 1 << n) if popcount(T) > k), default=0.0)
    record("C1 Moebius inversion recovers coefficients", err_in < 1e-10 and err_hi < 1e-10,
           f"max|m_rec-m|={err_in:.2e}, max|m_rec(|T|>k)|={err_hi:.2e}")


# ----------------------------------------------------------------------------
# C2  Exact decomposition of f(S) into order-j parts
# ----------------------------------------------------------------------------

def check_c2() -> None:
    n, k = 6, 3
    m = random_kadditive(n, k)
    b = -0.2
    worst = 0.0
    for S in range(1 << n):
        parts = [sum(m[T] for T in submasks(S) if T and popcount(T) == j) for j in range(1, k + 1)]
        worst = max(worst, abs(b + sum(parts) - f_set(m, b, S)))
    record("C2 f(S) = b + sum_j f_j(S) exactly", worst < 1e-12, f"max abs error {worst:.2e}")


# ----------------------------------------------------------------------------
# C3  Shapley value and pair interaction index from Moebius coefficients
# ----------------------------------------------------------------------------

def shapley_bruteforce(v: Callable[[int], float], n: int, i: int) -> float:
    others = [j for j in range(n) if j != i]
    total = 0.0
    for r in range(len(others) + 1):
        for K in itertools.combinations(others, r):
            mask = sum(1 << j for j in K)
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            total += w * (v(mask | (1 << i)) - v(mask))
    return total


def interaction_bruteforce(v: Callable[[int], float], n: int, i: int, j: int) -> float:
    others = [t for t in range(n) if t not in (i, j)]
    total = 0.0
    for r in range(len(others) + 1):
        for K in itertools.combinations(others, r):
            mask = sum(1 << t for t in K)
            w = math.factorial(n - r - 2) * math.factorial(r) / math.factorial(n - 1)
            d = (v(mask | (1 << i) | (1 << j)) - v(mask | (1 << i)) - v(mask | (1 << j)) + v(mask))
            total += w * d
    return total


def check_c3() -> None:
    n, k = 5, 3
    m = random_kadditive(n, k)
    v = lambda S: f_set(m, 0.0, S)
    e1 = 0.0
    for i in range(n):
        formula = sum(m[T] / popcount(T) for T in m if (T >> i) & 1)
        e1 = max(e1, abs(formula - shapley_bruteforce(v, n, i)))
    e2 = 0.0
    for i, j in itertools.combinations(range(n), 2):
        formula = sum(m[T] / (popcount(T) - 1) for T in m if (T >> i) & 1 and (T >> j) & 1)
        e2 = max(e2, abs(formula - interaction_bruteforce(v, n, i, j)))
    record("C3 Shapley + pair interaction index from Moebius", e1 < 1e-10 and e2 < 1e-10,
           f"max err Shapley={e1:.2e}, pair-interaction={e2:.2e}")


# ----------------------------------------------------------------------------
# C4  Multilinear extension: exact mean and variance under Bernoulli mastery
# ----------------------------------------------------------------------------

def check_c4() -> None:
    n, k = 5, 3
    m = random_kadditive(n, k)
    b = 0.1
    mu = RNG.uniform(0.1, 0.9, n)
    ex = 0.0
    ex2 = 0.0
    for X in range(1 << n):
        p = np.prod([mu[i] if (X >> i) & 1 else 1 - mu[i] for i in range(n)])
        fx = f_set(m, b, X)
        ex += p * fx
        ex2 += p * fx * fx
    var_enum = ex2 - ex * ex
    mean_formula = f_multilinear(m, b, mu)

    def dU(U: int) -> float:
        return sum(c * np.prod([mu[s] for s in members(T & ~U)]) for T, c in m.items() if (T & U) == U)

    var_by_order = {j: 0.0 for j in range(1, k + 1)}
    for U in range(1, 1 << n):
        if popcount(U) <= k:
            var_by_order[popcount(U)] += dU(U) ** 2 * np.prod([mu[s] * (1 - mu[s]) for s in members(U)])
    var_formula = sum(var_by_order.values())
    record("C4 E[f(X)] = f(mu) (multilinear extension)", abs(ex - mean_formula) < 1e-10,
           f"enumerated {ex:.8f} vs f(mu) {mean_formula:.8f}")
    record("C4 Var[f(X)] closed form (Efron-Stein on multilinear f)", abs(var_enum - var_formula) < 1e-10,
           f"enumerated {var_enum:.8f} vs formula {var_formula:.8f}; by order "
           + ", ".join(f"j={j}:{v:.4f}" for j, v in var_by_order.items()))


# ----------------------------------------------------------------------------
# C5  A 3-way AND is not representable at order <= 2; refit vs truncation
# ----------------------------------------------------------------------------

def fit_order(vals: np.ndarray, n: int, p: float, k: int):
    cols, names = [], []
    for T in range(0, 1 << n):
        if popcount(T) <= k:
            cols.append([1.0 if (S & T) == T else 0.0 for S in range(1 << n)])
            names.append(T)
    A = np.array(cols).T
    w = np.array([np.prod([p if (S >> i) & 1 else 1 - p for i in range(n)]) for S in range(1 << n)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], vals * sw, rcond=None)
    resid = vals - A @ coef
    return dict(zip(names, coef)), float((w * resid ** 2).sum()), float(w @ vals ** 2 - (w @ vals) ** 2)


def check_c5() -> None:
    n = 3
    vals = np.array([1.0 if S == 0b111 else 0.0 for S in range(1 << n)])
    lines = []
    ok = True
    for p in (0.2, 0.5, 0.8):
        _, r2, var = fit_order(vals, n, p, 2)
        _, r3, _ = fit_order(vals, n, p, 3)
        frac = r2 / var
        ok &= (r2 > 1e-6) and (r3 < 1e-12)
        lines.append(f"p={p}: Var={var:.4f}, resid(k=2)={r2:.4f} ({frac:.1%}), resid(k=3)={r3:.1e}")
    coef2, _, _ = fit_order(vals, n, 0.5, 2)
    leak = max(abs(coef2[T]) for T in coef2 if popcount(T) in (1, 2))
    record("C5 AND3 residual under order<=2 fit > 0; order-3 fit exact", ok, " | ".join(lines))
    record("C5b order-2 refit puts mass on lower orders (truncation != refit)", leak > 1e-3,
           f"true lower-order Moebius coefficients are 0; refit max |coef| = {leak:.3f}")


# ----------------------------------------------------------------------------
# C6  CP (ANOVA-kernel) parameterisation: DP equals brute force
# ----------------------------------------------------------------------------

def elementary_symmetric(z: np.ndarray, kmax: int) -> np.ndarray:
    """e_0..e_kmax of the rows of z (shape [|S|, R]) via O(|S| k R) DP."""
    a = np.zeros((kmax + 1, z.shape[1]))
    a[0] = 1.0
    for row in z:
        for t in range(kmax, 0, -1):
            a[t] += a[t - 1] * row
    return a


def check_c6() -> None:
    n, R, k = 7, 4, 3
    z = RNG.normal(size=(n, R))
    lam = RNG.normal(size=(k + 1, R))
    S = [0, 2, 3, 5, 6]
    dp = elementary_symmetric(z[S], k)
    worst = 0.0
    for j in range(1, k + 1):
        brute = sum(lam[j] @ np.prod(z[list(T)], axis=0) for T in itertools.combinations(S, j))
        worst = max(worst, abs(brute - lam[j] @ dp[j]))
    record("C6 CP coefficients: ANOVA DP = brute-force subset sum", worst < 1e-10,
           f"max abs error {worst:.2e} (cost O(|S| k R) vs O(C(|S|,<=k) k R))")


# ----------------------------------------------------------------------------
# C7  Monotonicity identity and the semantics of the three toy motifs
# ----------------------------------------------------------------------------

def check_c7() -> None:
    n, k = 5, 3
    m = random_kadditive(n, k)
    worst = 0.0
    for A in range(1 << n):
        for i in range(n):
            if not (A >> i) & 1:
                lhs = f_set(m, 0, A | (1 << i)) - f_set(m, 0, A)
                rhs = sum(m.get(T | (1 << i), 0.0) for T in submasks(A))
                worst = max(worst, abs(lhs - rhs))
    record("C7 marginal gain f(A+i)-f(A) = sum_{T subset A} m(T+i)", worst < 1e-12, f"max err {worst:.2e}")

    mon = {T: abs(c) for T, c in random_kadditive(n, k).items()}
    ok = all(f_set(mon, 0, A | (1 << i)) >= f_set(mon, 0, A) - 1e-12
             for A in range(1 << n) for i in range(n) if not (A >> i) & 1)
    record("C7b all m(T) >= 0 implies monotone", ok, "exhaustive over n=5")

    # Motifs. Skills 0,1,2.
    conj = {0b111: 2.0}                       # missing-prerequisite / conjunctive
    redun = {0b001: 1.0, 0b010: 1.0, 0b011: -1.0}   # two substitutes
    comp = {0b011: 1.5}                       # pure complement, no solo value
    f_conj = [f_set(conj, 0, S) for S in (0b011, 0b101, 0b110, 0b111)]
    ok_conj = f_conj[:3] == [0.0, 0.0, 0.0] and f_conj[3] == 2.0
    ok_red = (f_set(redun, 0, 0b011) == f_set(redun, 0, 0b001) == 1.0)
    ok_comp = (f_set(comp, 0, 0b001) == 0 and f_set(comp, 0, 0b010) == 0 and f_set(comp, 0, 0b011) == 1.5)
    record("C7c motifs: conjunctive / redundant / complementary", ok_conj and ok_red and ok_comp,
           "AND3 gives 0 unless all 3 present; substitutes saturate at 1.0; complement pair 0+0 -> 1.5")
    # Marginal completion: which single skill to add to S={0,1} under conj?
    gains = {s: f_set(conj, 0, 0b011 | (1 << s)) - f_set(conj, 0, 0b011) for s in (2,)}
    record("C7d marginal completion value identifies the missing skill", gains[2] == 2.0, f"gain(add skill 2)={gains[2]}")


# ----------------------------------------------------------------------------
# C8  PU risk: unknown-as-negative is biased, unbiased PU risk is not (SCAR)
# ----------------------------------------------------------------------------

def check_c8() -> None:
    pi, N = 0.3, 400_000
    w, b0 = np.array([1.2, -0.4]), -0.3
    logistic = lambda z, y: np.log1p(np.exp(-y * z))

    def g(x):
        return x @ w + b0

    pos = RNG.normal([1.0, 0.5], 1.0, size=(int(N * pi), 2))
    neg = RNG.normal([-1.0, -0.5], 1.0, size=(N - int(N * pi), 2))
    true_risk = pi * logistic(g(pos), 1).mean() + (1 - pi) * logistic(g(neg), -1).mean()

    # Case-control PU: labelled positives P ~ p(x|y=1); unlabelled U ~ p(x) (marginal).
    P = RNG.normal([1.0, 0.5], 1.0, size=(N // 4, 2))
    comp = RNG.random(N) < pi
    U = np.where(comp[:, None], RNG.normal([1.0, 0.5], 1.0, size=(N, 2)), RNG.normal([-1.0, -0.5], 1.0, size=(N, 2)))
    r_pu = pi * logistic(g(P), 1).mean() - pi * logistic(g(P), -1).mean() + logistic(g(U), -1).mean()
    # Naive baseline: every unlabelled item is treated as a negative, with the same
    # class weights (pi for positives, 1-pi for negatives) as the true PN risk.
    r_naive = pi * logistic(g(P), 1).mean() + (1 - pi) * logistic(g(U), -1).mean()
    bias_pu, bias_naive = abs(r_pu - true_risk), abs(r_naive - true_risk)
    record("C8 unbiased PU risk matches PN risk; unknown-as-negative does not",
           bias_pu < 0.01 and bias_naive > 0.05,
           f"true={true_risk:.4f}, PU={r_pu:.4f} (|bias|={bias_pu:.4f}), unknown-as-negative={r_naive:.4f} (|bias|={bias_naive:.4f})")


# ----------------------------------------------------------------------------
# C9  Cross-domain integration share: toy example
# ----------------------------------------------------------------------------

def check_c9() -> None:
    domain = {0: "CS", 1: "CS", 2: "STAT", 3: "BUS"}
    m = {0b0001: 0.5, 0b0010: 0.4, 0b0100: 0.3, 0b0011: 0.1,   # within CS pair
         0b0101: 0.9, 0b0111: 0.6}                              # CS+STAT, CS+CS+STAT
    S = 0b0111
    total = sum(abs(c) for T, c in m.items() if (T & S) == T)
    cross = sum(abs(c) for T, c in m.items() if (T & S) == T and len({domain[i] for i in members(T)}) >= 2)
    share = cross / total
    expected = (0.9 + 0.6) / (0.5 + 0.4 + 0.3 + 0.1 + 0.9 + 0.6)
    record("C9 cross-domain integration share (toy)", abs(share - expected) < 1e-12, f"share={share:.4f}")


# ----------------------------------------------------------------------------
# C10  Size of the coefficient space (justifies k <= 3 as the default)
# ----------------------------------------------------------------------------

def check_c10() -> None:
    rows = []
    for n in (6, 10, 15):
        rows.append(f"n={n}: " + ", ".join(f"k={k}:{sum(math.comb(n, j) for j in range(1, k + 1))}" for k in (1, 2, 3, 4)))
    record("C10 monomial counts sum_{j<=k} C(n,j)", True, " | ".join(rows) + " | full 2^n for n=15: 32768")


# ----------------------------------------------------------------------------
# C11  Worked toy example used in derivation.md section 10 (invented numbers)
# ----------------------------------------------------------------------------

def check_c11() -> None:
    # Skills: 0 = programming (CS), 1 = statistics (MS), 2 = business domain knowledge (BUS).
    # Hypothetical competency "business analytics". Coefficients are INVENTED for illustration.
    b = -3.0
    m = {0b001: 0.3, 0b010: 0.2, 0b100: 0.1, 0b011: 0.4, 0b101: 0.1, 0b110: 0.2, 0b111: 1.5}
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    f = {S: f_set(m, b, S) for S in range(8)}
    parts = [sum(c for T, c in m.items() if popcount(T) == j) for j in (1, 2, 3)]
    mu = np.array([0.9, 0.8, 0.5])
    mean = f_multilinear(m, b, mu)
    ex_f, ex_f2, ex_sig = 0.0, 0.0, 0.0
    for X in range(8):
        p = np.prod([mu[i] if (X >> i) & 1 else 1 - mu[i] for i in range(3)])
        ex_f += p * f[X]
        ex_f2 += p * f[X] ** 2
        ex_sig += p * sig(f[X])
    var = ex_f2 - ex_f ** 2
    probit = sig(mean / math.sqrt(1 + math.pi * var / 8))
    print("    f(S) by set:", {format(S, '03b'): round(v, 3) for S, v in f.items()})
    print(f"    order parts for S={{0,1,2}}: F1={parts[0]:.2f}, F2={parts[1]:.2f}, F3={parts[2]:.2f}")
    print(f"    mastery mu={mu.tolist()}: f(mu)={mean:.4f}, Var f(X)={var:.4f}, "
          f"exact E[sigmoid f(X)]={ex_sig:.4f}, probit approx={probit:.4f}, naive sigmoid(f(mu))={sig(mean):.4f}")
    record("C11 worked example is self-consistent", abs(ex_f - mean) < 1e-12 and abs(parts[0] + parts[1] + parts[2] - (f[7] - b)) < 1e-12,
           f"E[f(X)]={ex_f:.4f} equals f(mu)={mean:.4f}; probit approx error {abs(probit - ex_sig):.4f}")


def main() -> int:
    for fn in (check_c1, check_c2, check_c3, check_c4, check_c5, check_c6, check_c7, check_c8, check_c9, check_c10, check_c11):
        fn()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
