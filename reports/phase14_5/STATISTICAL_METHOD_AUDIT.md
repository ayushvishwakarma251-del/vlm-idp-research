# Statistical Methods and Multiple Testing Audit

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Implementation Source**: `src/phase14/statistics.py`  
**Execution Source**: `src/phase14/experiments.py`  
**Evaluation Table**: `experiments/phase14/tables/table_13_statistical_tests.csv`  

---

## 1. Executive Summary

This forensic audit evaluates the statistical testing methodology used in Phase 14 to validate hypotheses H14-3 and H14-4, focusing on multiple testing adjustments, family-wise error rate control, and test formulation.

### Key Audit Findings:
1. **Holm-Bonferroni Implementation vs. Execution**:
   - The function `apply_holm_bonferroni(p_values)` is correctly implemented in `src/phase14/statistics.py` (lines 48–60).
   - However, in `src/phase14/experiments.py` (lines 375–398), the function was imported but **never called**. The raw bootstrap p-values ($0.0002$ and $0.0000$) were directly written to `table_13_statistical_tests.csv`.
   - **Multiple Testing Impact Assessment**:
     - Raw $p_1 = 0.0000$, Raw $p_2 = 0.0002$.
     - Ranked for $m = 2$ hypotheses:
       - Rank 1: $p_{(1)} = 0.0000 \times 2 = 0.0000 \le 0.05$ (Significant)
       - Rank 2: $p_{(2)} = 0.0002 \times 1 = 0.0002 \le 0.05$ (Significant)
     - Both hypotheses remain statistically significant at $\alpha = 0.05$ and $\alpha = 0.001$.
2. **Paired Structure Integrity**:
   - Observations between conditions B14-A, B14-B, and B14-C were evaluated across identical queries and document instances. The paired design correctly controls for inter-document baseline difficulty variation.
3. **P-Value Reporting Standards**:
   - Both empirical 95% confidence intervals and p-values are reported together in Table 13, adhering to IEEE standards against p-value-only reporting.

---

## 2. P-Value Comparison Table

| Hypothesis | Comparison | Raw Empirical P-Value | Holm-Bonferroni Adjusted P-Value | Significance at $\alpha = 0.05$ | Significance at $\alpha = 0.01$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **H14-3** | B14-B vs B14-A (Accuracy Gain) | 0.0002 | 0.0002 | Significant | Significant |
| **H14-4** | B14-A vs B14-C (UAR Reduction) | 0.0000 | 0.0000 | Significant | Significant |

---

## 3. Scientific Recommendation

- **Finding Classification**: **`P3 (Minor Implementation Defect)`**.
- **Action**: In the manuscript and reproducibility notes, explicitly document that Table 13 displays raw empirical bootstrap p-values, which remain identical after Holm-Bonferroni correction across the 2 pre-registered tests.
