# PHASE 14.7 — MULTIPLE TESTING AND CONFIRMATORY FAMILY AUDIT REPORT

## 1. Multiple-Testing Governance & Partition Architecture

A critical vulnerability in machine learning empirical claims is post-hoc hypothesis generation and uncorrected multi-threshold fishing. If an investigator tests 6 different IoU thresholds, 4 architectural conditions, and multiple metrics simultaneously, testing dozens of combinations without multiple-testing adjustment guarantees Type I error inflation.

To uphold IEEE Transactions-level statistical rigor, Phase 14.7 establishes a strict, preregistered partition:

```
                          TOTAL EVALUATED HYPOTHESES
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
      CONFIRMATORY FAMILY (m = 2)               EXPLORATORY SENSITIVITY FAMILIES
      Strict Family-Wise Error Rate (FWER)       Threshold & Intermediate Path Exploration
      Step-Down Holm-Bonferroni (α = 0.05)       Reported for Robustness Profiling Only
              │                                               │
      ┌───────┴───────┐                       ┌───────────────┴───────────────┐
      │               │                       │                               │
    H14-3           H14-4               EXPL-TAU (m = 6)              EXPL-ARCH (m = 4)
  Grounded SUC      UAR              Grid τ ∈ {0.30..0.80}       Pairwise Paths (B-A, C-B, etc.)
  (τ = 0.50)      Reduction
```

---

## 2. Confirmatory Hypothesis Family Results ($m = 2$)

The primary scientific claim of the paper rests solely on the Confirmatory Hypothesis Family evaluated under exact small-sample permutation tests ($K = 11$, 2,048 permutations):

| Hypothesis ID | Scientific Statement | Metric | Raw Exact Permutation $p$ | Holm Multiplier $(m - k + 1)$ | Adjusted $p$-value | Significance ($\alpha = 0.05$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **H14-3** | Grounded SUC is significantly higher in proposed B14-D than baseline B14-A | `grounded_suc_tau50` | **0.000977** | 2 | **0.001954** | **CONFIRMED** |
| **H14-4** | Unsupported Answer Rate (UAR) is significantly reduced in B14-D vs B14-A | `uar_unconditional` | **0.000977** | 1 | **0.001954** | **CONFIRMED** |

Both confirmatory hypotheses retain rigorous statistical significance ($p_{\text{adj}} \approx 0.00195 < 0.05$) after step-down Holm-Bonferroni control.

---

## 3. Exploratory Sensitivity Family Results ($m = 6$)

Sweeping the preregistered spatial overlap threshold $\tau \in \{0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}$ represents a sensitivity analysis to assess metric robustness.

| Threshold $\tau$ | Condition Delta $\Delta(D - A)$ | Raw Permutation $p$ | Adjusted $p$ ($m=6$) | Exploratory Status |
| :---: | :---: | :---: | :---: | :---: |
| **0.30** | +16.36 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |
| **0.40** | +34.54 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |
| **0.50** | +70.91 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |
| **0.60** | +89.09 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |
| **0.70** | +89.09 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |
| **0.80** | +43.64 pp | 0.000977 | 0.005862 | ROBUST_EFFECT |

Even if treated as an unpartitioned 6-test exploratory family, every single threshold retains significance under Holm-Bonferroni adjustment ($p_{\text{adj}} \le 0.00586 < 0.05$).

---

## 4. Auditor's Verdict on Multiplicity

> [!NOTE] Multiplicity Partition Verdict
> The statistical separation between Confirmatory ($m=2$) and Exploratory sensitivity tests is **fully defensible**.
> The primary scientific conclusions of Phase 14 / Phase 14.6 are not an artifact of data snooping or multi-testing inflation.
