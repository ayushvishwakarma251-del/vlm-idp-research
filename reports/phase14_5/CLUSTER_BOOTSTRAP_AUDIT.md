# Cluster Bootstrap Methodology Audit

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Implementation Source**: `src/phase14/statistics.py` (`cluster_bootstrap_paired`)  
**Sample Clusters**: 11 Document Families (`fam_auth_42` .. `fam_auth_52`)  
**Resampling Intensity**: $B = 10,000$ iterations  

---

## 1. Executive Summary

This forensic audit evaluates the mathematical formulation, implementation, and assumptions of the document-family-level cluster bootstrap algorithm used to compute empirical confidence intervals and p-values in Phase 14.

### Audit Verdict: `VERIFIED_WITH_QUALIFICATIONS`
- **Algorithm Correctness**: The implementation in `src/phase14/statistics.py` correctly resamples document family clusters with replacement ($N = 11$, size = 11), concatenates all observations within selected clusters, recomputes paired mean differences, and determines 2.5th and 97.5th percentiles.
- **Reproducibility**: Under fixed seed 42 and $B = 10,000$, confidence intervals and p-values reproduce with zero error.
- **Cluster Count Qualification**: The number of independent clusters in the test set is $N = 11$. In econometric and biostatistical literature (Cameron & Miller, 2015), cluster bootstrap inference with fewer than 30–50 clusters can exhibit minor finite-sample distortion. However, because effect sizes here are large ($\Delta = +8.36\%$ and $\Delta = +21.09\%$) and confidence intervals are bounded far from zero ($[+3.64\%, +13.45\%]$ and $[+17.82\%, +24.73\%]$), the conclusions are statistically robust.

---

## 2. Mathematical Formalism

Let $C = \{c_1, c_2, \dots, c_K\}$ denote the set of $K = 11$ document family clusters. For each bootstrap iteration $b \in \{1, \dots, B\}$ ($B = 10,000$):
1. Sample $K$ clusters with replacement: $C^{*(b)} = \{c_1^{*(b)}, \dots, c_K^{*(b)}\}$.
2. Assemble all observations belonging to the resampled clusters:
   $$D^{*(b)} = \bigcup_{k=1}^K D_{c_k^{*(b)}}$$
3. Compute the paired mean difference on $D^{*(b)}$:
   $$\Delta^{*(b)} = \frac{1}{|D^{*(b)}|} \sum_{i \in D^{*(b)}} (y_{i, \text{Condition B}} - y_{i, \text{Condition A}})$$
4. The two-sided empirical percentile 95% confidence interval is:
   $$\text{CI}_{95\%} = [\text{Percentile}_{2.5\%}(\vec{\Delta}^*), \text{Percentile}_{97.5\%}(\vec{\Delta}^*)]$$
5. The empirical two-sided p-value testing $H_0: \Delta = 0$ is:
   $$p = 2 \times \min\left(\frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta^{*(b)} \le 0), \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta^{*(b)} \ge 0)\right)$$

---

## 3. Comparison with Alternative Statistical Tests

To evaluate sensitivity to the cluster bootstrap specification, parametric paired t-tests and Wilcoxon signed-rank tests were computed across the 11 cluster means:

| Test Method | H14-3 (Accuracy Gain) | H14-4 (UAR Reduction) | Agreement with Bootstrap |
|:---|:---:|:---:|:---:|
| **Cluster Bootstrap (10,000 resamples)** | $p = 0.0002$ | $p = 0.0000$ | Baseline |
| **Paired Student's t-test ($N=11$)** | $p = 0.0075$ | $p < 0.0001$ | Full Agreement ($p < 0.01$) |
| **Wilcoxon Signed-Rank Test ($N=11$)** | $p = 0.0098$ | $p = 0.0010$ | Full Agreement ($p < 0.01$) |

Regardless of whether cluster bootstrapping, paired t-tests, or non-parametric rank tests are employed, both hypotheses are strongly confirmed ($p < 0.01$).

---

## 4. Scientific Qualification

- **Finding Classification**: **`P2 (Methodological Qualification)`**.
- **Manuscript Text Requirement**: Explicitly state that $K = 11$ independent document family clusters were evaluated in the authentic test set, and report that both cluster bootstrap and exact non-parametric tests confirm statistical significance.
