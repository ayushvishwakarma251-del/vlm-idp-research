# Sensitivity and Robustness Analysis Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Associated Hypotheses**: H14-3 (Accuracy Gain) & H14-4 (UAR Reduction)  

---

## 1. Executive Summary

To stress-test the statistical conclusions of Phase 14 against potential vulnerabilities in cluster count, random seeds, bootstrap resampling intensity, and distributional assumptions, four orthogonal sensitivity analyses were executed:
1. **Leave-One-Out (LOO) Cluster Stability**: Evaluates whether any single document family drives the statistical effect.
2. **Bootstrap Seed Sensitivity**: Evaluates random seed stability across 5 independent RNG seeds ($B = 10,000$).
3. **Resampling Intensity Scaling**: Compares bootstrap convergence from $B = 1,000$ to $B = 20,000$ resamples.
4. **Non-Parametric & Parametric Alternative Tests**: Compares cluster bootstrap with paired Student's t-tests and Wilcoxon signed-rank tests.

| Sensitivity Axis | Tested Variation | Observed Impact on H14-3 ($\Delta_{\text{acc}}$) | Observed Impact on H14-4 ($\Delta_{\text{UAR}}$) | Decision Robustness |
|:---|:---|:---:|:---:|:---:|
| **Leave-One-Out Clusters** | Exclude each of 11 families | Mean $\Delta \in [+6.40\%, +10.00\%]$ | Mean $\Delta \in [+20.00\%, +22.00\%]$ | **STABLE** (100% positive) |
| **Random Seeds** | Seeds 42, 123, 456, 789, 2024 | $p \le 0.0012$ across all seeds | $p \le 0.0001$ across all seeds | **STABLE** (All $p \ll 0.05$) |
| **Resampling Intensity** | $B \in [1k, 5k, 10k, 20k]$ | CI bounded $[+3.27\%, +13.82\%]$ | CI bounded $[+17.82\%, +24.73\%]$ | **STABLE** |
| **Alternative Statistical Tests**| t-test & Wilcoxon signed-rank | $p = 0.0118$ (t), $p = 0.0137$ (W) | $p < 0.0001$ (t), $p = 0.0010$ (W) | **STABLE** (All $p < 0.02$) |

---

## 2. Granular Results

### A. Leave-One-Out (LOO) Analysis
- **H14-3**: When omitting the most favorable family (`fam_auth_49`, which gained $+28\%$), the remaining 10 families still exhibit a mean gain of $+6.40\%$. When omitting the single negative family (`fam_auth_50`, $-8\%$), the mean gain rises to $+10.00\%$. The effect is not driven by an outlier family.
- **H14-4**: The mean UAR reduction remains tightly constrained between $+20.00\%$ and $+22.00\%$ regardless of which family is removed.

### B. Alternative Hypothesis Testing
- Paired Student's t-test on family means:
  - H14-3: $t(10) = 3.074, p = 0.0118$
  - H14-4: $t(10) = 10.808, p = 3.8 \times 10^{-7}$
- Wilcoxon signed-rank test:
  - H14-3: $W = 4.0, p = 0.0137$
  - H14-4: $W = 0.0, p = 0.00098$

---

## 3. Scientific Verdict

The statistical significance of Phase 14 hypotheses H14-3 and H14-4 is **exceptionally robust** and invariant to choice of random seed, bootstrap resample size, test family subset, or statistical test paradigm.
