# PHASE 14.7 — SMALL-CLUSTER ($K=11$) FAMILY-LEVEL STATISTICAL AUDIT REPORT

## 1. Executive Summary & Audit Context

In Phase 14 and Phase 14.6, cluster bootstrapping was performed using 10,000 resamples to account for clustering within document families. However, classical cluster bootstrap theory relies on asymptotic arguments where the number of clusters $K \to \infty$. With $K = 11$ document families (`fam_auth_42` through `fam_auth_52`), small-sample cluster properties, finite degrees of freedom ($df = 10$), and individual cluster leverage become critical concerns for publication defense.

This forensic statistical audit independently evaluated:
1. **True Independent Cluster Units**: Aggregated all 1,100 trace records into exactly 11 family-level outcome vectors per condition ($N = 55$ unique queries, 5 queries per family, evaluated across 5 seeds).
2. **Exact Paired Permutation Tests**: Enumerated all $2^{11} = 2,048$ possible sign-flip permutations, eliminating random Monte Carlo sampling error.
3. **Nonparametric Tests**: Exact Wilcoxon signed-rank and exact two-sided binomial sign tests.
4. **Finite-Sample Parametric Bounds**: Paired Student's $t$-distribution with $df = 10$.
5. **Leave-One-Family-Out (LOFO) Sensitivity**: Jackknife influence analysis across each of the 11 individual families.

---

## 2. Cluster Geometry and Aggregation

- **Total Clusters ($K$)**: 11 unique document families.
- **Queries per Cluster**: Exactly 5 queries per family.
- **Trace Evaluations per Cluster**: 25 evaluations per family per condition (5 queries $\times$ 5 seeds).
- **Paired Design**: Perfect 1-to-1 family matching across baseline and proposed conditions.

---

## 3. Statistical Test Results ($K=11$ Clusters)

| Hypothesis | Comparison | Metric | Cluster Mean $\Delta$ | 95% CI ($df=10$) | Exact Permutation $p$ ($2^{11}=2,048$) | Wilcoxon $p$ (Exact) | Sign Test $p$ ($n=11$) | LOFO Min $\Delta$ | LOFO Max $\Delta$ | Sign Preserved |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H14-3** | B14-D vs B14-A | Grounded SUC ($\tau=0.50$) | **+70.91 pp** | [+65.26, +76.56] | **0.000977** | **0.000977** | **0.000977** | +69.20 pp | +72.40 pp | **100% (11/11)** |
| **H14-4** | B14-D vs B14-A | Unconditional UAR | **-21.82 pp** | [-26.04, -17.59] | **0.000977** | **0.000977** | **0.000977** | -22.80 pp | -20.80 pp | **100% (11/11)** |

*Note: The minimum achievable two-sided $p$-value for $K=11$ in an exact permutation or sign test is $\frac{2}{2^{11}} = \frac{2}{2048} \approx 0.0009765625$. Both hypotheses achieved the theoretical mathematical limit of significance.*

---

## 4. In-Depth Analysis of Hypotheses

### 4.1 H14-3: Grounded SUC ($\tau = 0.50$)
- **Family Performance Direction**: All 11 families exhibited strictly positive deltas ($11/11 > 0$).
- **Cluster Mean $\Delta$**: $+70.91\text{ pp}$ ($95\%\text{ CI } [65.26\%, 76.56\%]$, $t = 27.97, df = 10, p < 10^{-9}$).
- **Permutation Test**: Exactly 2 out of 2,048 permutations produced a mean magnitude $\ge 70.91\text{ pp}$ ($p = 0.000977$).
- **LOFO Sensitivity**: Omitting any single family yields mean deltas between $+69.20\text{ pp}$ and $+72.40\text{ pp}$. The effect is not driven by any outlier document family.

### 4.2 H14-4: Hallucination / Unsupported Answer Rate (UAR)
- **Family Performance Direction**: All 11 families exhibited strictly negative deltas ($11/11 < 0$).
- **Cluster Mean $\Delta$**: $-21.82\text{ pp}$ ($95\%\text{ CI } [-26.04\%, -17.59\%]$, $t = -11.50, df = 10, p < 10^{-6}$).
- **Permutation Test**: Exactly 2 out of 2,048 permutations produced a mean magnitude $\ge 21.82\text{ pp}$ ($p = 0.000977$).
- **LOFO Sensitivity**: Omitting any single family yields mean deltas between $-22.80\text{ pp}$ and $-20.80\text{ pp}$.

---

## 5. Auditor's Verdict on Small-Cluster Limitations

> [!NOTE] Small-Cluster Validity Finding
> The reported effects for H14-3 and H14-4 are **statistically robust to finite-sample cluster adjustments ($K=11, df=10$)**.
> Because the effect sizes are large and directionally unanimous across all 11 families, the tests achieve the lowest possible discrete $p$-value ($p = 2/2048 \approx 0.00098$).
> 
> However, an IEEE paper MUST NOT cite asymptotic normal approximations without disclosing that $K=11$ clusters limits the minimum achievable $p$-value to $\approx 10^{-3}$ and restricts broader external generalizability to unseen document layouts.
