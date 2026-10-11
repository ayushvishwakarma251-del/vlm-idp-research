# PHASE 14.7 — MASTER RESEARCH AND FORENSIC AUDIT REPORT

## Title: Independent Grounded-Metric Challenge, Small-Cluster Statistical Audit & Robustness Verification

**Author**: Senior ML Research Auditor, Biostatistical Specialist & IEEE Independent Reviewer  
**Repository**: `vlm-idp-research`  
**Base Lineage**: Phase 14 (`583e8a8f`) $\to$ Phase 14.5 (`6d95f8d7`) $\to$ Phase 14.6 (`c1b2fbe2`)  
**Audit Phase**: Phase 14.7  

---

## 1. Executive Summary

Phase 14.7 subjected the Phase 14.6 remediated codebase, metrics, and experimental claims to an adversarial reproducibility and biostatistical audit.

The primary questions evaluated were:
1. **Metric Construct & Threshold Stability**: Is the newly introduced `SUC_grounded` metric sound, fairly evaluated against identical query denominators, and invariant to the spatial overlap threshold $\tau$?
2. **Small-Cluster ($K=11$) Inference**: Do statistical conclusions hold when accounting for the true independent sampling unit (the 11 document families), finite degrees of freedom ($df = 10$), and individual cluster leverage?
3. **Multiple Testing Control**: Are conclusions protected against Type I error inflation when sweeping multiple thresholds and comparisons?
4. **Adversarial Peer Review**: Which manuscript claims survive an IEEE TPAMI-level critical cross-examination?

### Key Audit Findings:
- **Strict Rank-Order Invariance**: Across all evaluated IoU thresholds $\tau \in \{0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}$, condition ordering is strictly preserved:
  $$\text{B14-D (Proposed)} \ge \text{B14-C (Grounded)} \ge \text{B14-B (Retrieval)} \ge \text{B14-A (Baseline)}$$
  Zero rank reversals occurred across the entire parameter space.
- **Finite-Cluster Statistical Confirmation**: Clustered exact permutation tests across all $2^{11} = 2,048$ combinations confirm both primary hypotheses at the discrete theoretical limit:
  - **H14-3 (Grounded SUC $\Delta$)**: Mean delta $+70.91\text{ pp}$ ($95\%\text{ CI } [65.26\%, 76.56\%]$, $p_{\text{perm}} = 0.000977$).
  - **H14-4 (UAR Reduction $\Delta$)**: Mean delta $-21.82\text{ pp}$ ($95\%\text{ CI } [-26.04\%, -17.59\%]$, $p_{\text{perm}} = 0.000977$).
- **Jackknife Stability**: Leave-One-Family-Out (LOFO) across all 11 individual families revealed zero sign flips, with minimal delta fluctuations (minimum SUC gain was $+69.20\text{ pp}$).
- **Multiple Testing Protection**: Under step-down Holm-Bonferroni correction within the confirmatory family ($m=2$), both hypotheses retain significance ($p_{\text{adj}} \approx 0.00195 < 0.05$).
- **100% Regression Suite Pass**: All 50 tests across Phase 14, 14.5, 14.6, and 14.7 pass without any failures.

---

## 2. Table Summary of Phase 14.7 Experimental Deliverables

1. `experiments/phase14_7/tables/threshold_sensitivity.csv`: Full factorial metric grid across 6 thresholds $\times$ 4 conditions.
2. `experiments/phase14_7/tables/threshold_family_effects.csv`: Family-specific Grounded SUC across all thresholds.
3. `experiments/phase14_7/tables/family_level_outcomes.csv`: Clustered performance profiles for all $K=11$ document families.
4. `experiments/phase14_7/tables/family_level_statistical_results.csv`: Exact permutation, Wilcoxon, sign, and LOFO test statistics.
5. `experiments/phase14_7/figures/grounded_coverage_threshold_sensitivity.svg`: Standalone vector visualization of threshold invariance.
6. `experiments/phase14_7/audit/claim_evidence_ledger.csv`: Formal evidentiary classification for 5 primary manuscript claims.
7. `experiments/phase14_7/checksums/phase14_7_sha256_manifest.json`: Cryptographic integrity manifest for all Phase 14.7 artifacts.

---

## 3. Final Recommendation for IEEE Manuscript

The empirical evidence solidly justifies the architectural advantages of multimodal retrieval, evidence grounding, and selective abstention for degraded multi-page document intelligence. To achieve publication integrity, the manuscript must adopt the remediated provenance declarations established across Phase 14.6 and Phase 14.7.
