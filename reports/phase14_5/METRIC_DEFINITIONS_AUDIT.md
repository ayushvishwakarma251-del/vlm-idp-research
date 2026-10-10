# Metric Definitions and Calculation Audit

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Source Code**: `src/phase14/experiments.py`  
**Denominator Table**: `experiments/phase14_5/tables/metric_denominator_audit.csv`  

---

## 1. Executive Summary

This forensic audit investigates the operational definitions, mathematical formulations, numerators, and denominators of all reported Phase 14 evaluation metrics across conditions B14-A through B14-D.

### Primary Audit Findings:
1. **Mathematical Collapse of Safe Useful Coverage (SUC)**: In Phase 14 tables (Table 10 and Table 12), SUC is reported as numerically identical to Exact Match Accuracy across all four conditions ($0.727273$, $0.810909$, $0.861818$, and $0.890909$). This is caused by an implementation artifact in `src/phase14/experiments.py` where `suc = acc if unsupported == 0.0 else 0.0`, combined with conditioning `unsupported` strictly on `acc == 0.0`. Consequently, whenever `acc == 1.0`, `suc` is forced to `1.0`; and whenever `acc == 0.0`, `suc` is forced to `0.0`. Thus:
$$\text{SUC} \equiv \text{Exact Match Accuracy}$$
2. **Unsupported Answer Rate (UAR) Denominator**: The reported UAR uses the **total query population** ($N = 275$) as its denominator, rather than conditioning exclusively on incorrect answers:
   - Condition B14-A: $\frac{63}{275} = 22.91\%$ (unconditional) vs. $\frac{63}{75} = 84.00\%$ (conditional on incorrect)
   - Condition B14-B: $\frac{13}{275} = 4.73\%$ (unconditional) vs. $\frac{13}{52} = 25.00\%$ (conditional on incorrect)
   - Condition B14-C: $\frac{5}{275} = 1.82\%$ (unconditional) vs. $\frac{5}{38} = 13.16\%$ (conditional on incorrect)
   - Condition B14-D: $\frac{3}{275} = 1.09\%$ (unconditional) vs. $\frac{3}{30} = 10.00\%$ (conditional on incorrect)
3. **Exact Match Accuracy**: Correctly defined as $\frac{N_{\text{correct}}}{N_{\text{total}}}$ where $N_{\text{total}} = 275$.

---

## 2. Metric Denominator and Counting Audit

| Condition | Evaluated Queries ($N$) | Exact Match Numerator | Exact Match Rate | Incorrect Answers | Unsupported Numerator | Reported UAR ($\frac{\text{Unsup}}{N}$) | Conditional UAR ($\frac{\text{Unsup}}{\text{Incorrect}}$) | SUC Numerator | SUC Rate | $\Delta(\text{SUC} - \text{Acc})$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B14-A** | 275 | 200 | 72.73% | 75 | 63 | 22.91% | 84.00% | 200 | 72.73% | 0.0000 |
| **B14-B** | 275 | 223 | 81.09% | 52 | 13 | 4.73% | 25.00% | 223 | 81.09% | 0.0000 |
| **B14-C** | 275 | 237 | 86.18% | 38 | 5 | 1.82% | 13.16% | 237 | 86.18% | 0.0000 |
| **B14-D** | 275 | 245 | 89.09% | 30 | 3 | 1.09% | 10.00% | 245 | 89.09% | 0.0000 |

---

## 3. Grounding IoU Metric Formulation

- **Metric**: Mean Intersection-over-Union between retrieved/predicted bounding box and ground-truth supporting evidence coordinates.
- **Observed Means**:
  - B14-A: $0.4527 \pm 0.0586$
  - B14-B: $0.6477 \pm 0.0588$
  - B14-C: $0.7553 \pm 0.0388$
  - B14-D: $0.7994 \pm 0.0477$
- **Findings**: IoU values are well-bounded in $[0, 1]$, finite, and monotonically increasing with pipeline sophistication.

---

## 4. Scientific Recommendations and Manuscript Qualification

1. **SUC Discrepancy (P2 Finding)**: The manuscript text must disclose that SUC in Table 10 and Table 12 was numerically identical to Exact Match Accuracy because abstention coverage was not decoupled from answer correctness in the evaluation script.
2. **UAR Clarification (P2 Finding)**: The manuscript must explicitly declare whether UAR refers to the **overall corpus hallucination rate** ($\frac{\text{unsupported answers}}{\text{all queries}}$) or the **error hallucination rate** ($\frac{\text{unsupported answers}}{\text{incorrect answers}}$) to prevent ambiguity during peer review.
