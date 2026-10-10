# Accuracy Statistical Reanalysis Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Comparison**: Condition B14-B (Retrieval-Pruned VLM) vs. B14-A (Full-Document VLM)  
**Registered Hypothesis**: H14-3 (Retrieval Efficiency Accuracy Gain)  
**Primary Tables**: `experiments/phase14/tables/table_13_statistical_tests.csv`, `experiments/phase14_5/tables/recomputed_statistical_tests.csv`  

---

## 1. Executive Summary

This forensic report independently recomputes and evaluates the accuracy improvement achieved by multimodal retrieval-based page pruning (B14-B) over full-document context feeding (B14-A).

- **Sample Size**: 11 clusters (document families), 55 unique queries $\times$ 5 seeds = 275 paired observations per condition.
- **B14-A Accuracy Mean**: $72.73\%$ ($200 / 275$)
- **B14-B Accuracy Mean**: $81.09\%$ ($223 / 275$)
- **Observed Mean Delta**: $+8.36\%$ ($+0.083636$)
- **Family-Clustered Bootstrap 95% CI** ($B = 10,000$): $[+3.64\%, +13.45\%]$
- **Empirical P-Value**: $p = 0.0002$
- **Holm-Bonferroni Adjusted P-Value**: $p_{\text{adj}} = 0.0002$
- **Reproducibility Status**: **`VERIFIED`** (Exact numerical concordance with historical Table 13).

---

## 2. Family-Level Paired Distribution

At the independent cluster level ($N = 11$ document families), paired accuracy comparisons demonstrate positive or non-negative gains across 10 out of 11 families:

| Family ID | B14-A Accuracy | B14-B Accuracy | Paired Delta ($\Delta$) | Direction |
|:---|:---:|:---:|:---:|:---:|
| `fam_auth_42` | 72.0% | 84.0% | +12.0% | Positive |
| `fam_auth_43` | 80.0% | 80.0% | 0.0% | Neutral |
| `fam_auth_44` | 68.0% | 80.0% | +12.0% | Positive |
| `fam_auth_45` | 68.0% | 76.0% | +8.0% | Positive |
| `fam_auth_46` | 80.0% | 84.0% | +4.0% | Positive |
| `fam_auth_47` | 68.0% | 80.0% | +12.0% | Positive |
| `fam_auth_48` | 72.0% | 76.0% | +4.0% | Positive |
| `fam_auth_49` | 64.0% | 92.0% | +28.0% | Positive |
| `fam_auth_50` | 76.0% | 68.0% | -8.0% | Negative |
| `fam_auth_51` | 76.0% | 88.0% | +12.0% | Positive |
| `fam_auth_52` | 76.0% | 84.0% | +8.0% | Positive |

Only 1 family (`fam_auth_50`) experienced an accuracy drop, while 9 families improved by up to $+28.0\%$.

---

## 3. Parametric vs. Empirical Caveat

While the recomputed bootstrap mathematics, confidence intervals, and p-values match historical Table 13 bit-for-bit, forensic code auditing (see `TRACE_INTEGRITY_REPORT.md`) revealed that the underlying query-level accuracies were generated via parametric sampling (`rng.rand() < threshold`) calibrated to experimental targets rather than physical model forward passes.

**Conclusion**: The statistical calculation is **mathematically verified**, but claims regarding real model accuracy must be qualified as reflecting a calibrated parametric simulation.
