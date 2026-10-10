# Unsupported Answer Rate (UAR) Statistical Reanalysis Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Comparison**: Condition B14-A (Baseline VLM) vs. B14-C (Retrieval + Spatial Grounding VLM)  
**Registered Hypothesis**: H14-4 (Grounded Generation UAR Reduction)  
**Primary Tables**: `experiments/phase14/tables/table_13_statistical_tests.csv`, `experiments/phase14_5/tables/recomputed_statistical_tests.csv`  

---

## 1. Executive Summary

This forensic report independently recomputes and evaluates the reduction in Unsupported Answer Rate (UAR) resulting from spatial evidence grounding (B14-C) compared with the full-document baseline (B14-A).

- **Sample Size**: 11 clusters (document families), 55 unique queries $\times$ 5 seeds = 275 paired observations per condition.
- **B14-A UAR Mean**: $22.91\%$ ($63 / 275$)
- **B14-C UAR Mean**: $1.82\%$ ($5 / 275$)
- **Observed Absolute Reduction ($\Delta_{\text{reduction}}$)**: $+21.09\%$ ($+0.210909$)
- **Relative UAR Reduction**: $92.05\%$ reduction in unsupported answers.
- **Family-Clustered Bootstrap 95% CI** ($B = 10,000$): $[+17.82\%, +24.73\%]$
- **Empirical P-Value**: $p = 0.0000$ ($< 10^{-4}$)
- **Holm-Bonferroni Adjusted P-Value**: $p_{\text{adj}} = 0.0000$
- **Reproducibility Status**: **`VERIFIED`** (Exact numerical concordance with historical Table 13).

---

## 2. Family-Level Paired UAR Distribution

At the independent cluster level ($N = 11$ document families), spatial grounding achieved strict non-negative UAR reductions across **100% of evaluated families**:

| Family ID | B14-A UAR | B14-C UAR | Paired Reduction ($\Delta$) | Direction |
|:---|:---:|:---:|:---:|:---:|
| `fam_auth_42` | 16.0% | 0.0% | +16.0% | Positive Reduction |
| `fam_auth_43` | 16.0% | 0.0% | +16.0% | Positive Reduction |
| `fam_auth_44` | 20.0% | 4.0% | +16.0% | Positive Reduction |
| `fam_auth_45` | 28.0% | 4.0% | +24.0% | Positive Reduction |
| `fam_auth_46` | 20.0% | 4.0% | +16.0% | Positive Reduction |
| `fam_auth_47` | 32.0% | 0.0% | +32.0% | Positive Reduction |
| `fam_auth_48` | 28.0% | 0.0% | +28.0% | Positive Reduction |
| `fam_auth_49` | 32.0% | 4.0% | +28.0% | Positive Reduction |
| `fam_auth_50` | 16.0% | 4.0% | +12.0% | Positive Reduction |
| `fam_auth_51` | 24.0% | 0.0% | +24.0% | Positive Reduction |
| `fam_auth_52` | 20.0% | 0.0% | +20.0% | Positive Reduction |

Across every single document family, spatial grounding reduced the unsupported answer rate by at least $+12.0\%$, with 4 families reaching complete zero-hallucination states ($0.0\%$ UAR).

---

## 3. Denominator and Operational Qualification

As uncovered in `METRIC_DEFINITIONS_AUDIT.md`, the reported UAR represents the unconditional error proportion relative to the full test set ($\frac{\text{unsupported answers}}{275}$). When evaluated conditionally on incorrect responses:
- B14-A: $84.00\%$ ($63 / 75$) of errors were unsupported hallucinations.
- B14-C: $13.16\%$ ($5 / 38$) of errors were unsupported hallucinations.

**Conclusion**: The statistical significance of UAR reduction is **fully verified**, and the operational definition must be clearly described in the IEEE manuscript.
