# Statistical Revalidation and Multiplicity Correction Report

**Phase**: Phase 14.6 — Remediation of Discrepancies P3-01 & Cluster Sampling Design  
**Target Code**: `src/phase14/statistics.py`, `src/phase14/experiments.py:378-398`  
**Remediated Module**: `src/phase14_6/statistics.py`  
**Test Suite**: `tests/test_phase14_6_statistics.py`  
**Date**: 2026-10-11  
**Status**: **RESOLVED**  

---

## 1. Context and Original Discrepancies

### 1.1 Discrepancy P3-01: Holm-Bonferroni Implementation vs Calling
In Phase 14, `src/phase14/experiments.py:36` imported `apply_holm_bonferroni`, and Section VII of the Phase 14 research report stated:
> *"Statistical significance across multiple comparisons was confirmed via Holm-Bonferroni correction."*

However, in lines 378–398 of `experiments.py`, the code wrote `res_h14_3["p_value"]` ($0.0002$) and `res_h14_4["p_value"]` ($0.0000$) directly into `table_13_statistical_tests.csv`. The `apply_holm_bonferroni` function was never called in the table-export pipeline.

### 1.2 Statistical Sampling Hierarchy: 11 Document Families
Phase 14 generated 1,100 traces across 5 seeds, 4 conditions, and 55 queries. These 55 queries were derived from only **11 distinct document families**. 
Treating 1,100 traces as 1,100 independent degrees of freedom would commit pseudoreplication. The true independent sampling unit is the **document family cluster ($K = 11$)**.

---

## 2. Independent Sampling Unit and Data Hierarchy

The experimental sampling hierarchy is formally modeled as:

$$\text{Document Family } (K = 11) \longrightarrow \text{Document Pages} \longrightarrow \text{Query } (N = 55) \longrightarrow \text{Condition } (4) \times \text{Seed } (5) \longrightarrow \text{Trace } (1,100)$$

### Cluster Structure:
- **Total Clusters**: $K = 11$ authentic document families.
- **Queries per Cluster**: Mean of 5 queries per family.
- **Replicates per Query**: $4 \text{ conditions} \times 5 \text{ seeds} = 20 \text{ traces per query}$.
- **Independence Principle**: All statistical inference must resample or aggregate at the **family level ($K = 11$)**, ensuring that intra-family correlations across queries and seeds do not artificially deflate standard errors.

---

## 3. Cluster Bootstrap Audit and Finite-Cluster Limitations

Phase 14 implemented a cluster bootstrap with $B = 10,000$ resamples. The bootstrap resampled document families with replacement, maintaining condition pairing within each sampled family.

### Methodological Audit Findings:
1. **Paired Structure**: Correctly preserved. When a family was drawn, both Condition A and Condition B outcomes for that family were drawn together.
2. **Empirical Percentile CIs**: Two-sided 95% confidence intervals were obtained from the 2.5th and 97.5th percentiles of the 10,000 bootstrap replicates.
3. **Finite Cluster Limitations ($K = 11$)**:
   - In cluster-robust inference, asymptotic normality and bootstrap consistency typically require $K \ge 30$ to $50$ clusters (Cameron, Gelbach, & Miller, 2008).
   - With $K = 11$, bootstrap distributions exhibit finite-sample discrete massing and sensitivity to individual cluster leverage.
   - **Scientific Disclosure**: The 10,000 resamples simulate combinations of the 11 observed families; they do **not** generate 10,000 independent real-world document families. Inferential claims must disclose the small cluster degrees of freedom ($df \approx 10$).

---

## 4. Multiplicity Correction: Tested Holm-Bonferroni Application

Phase 14.6 implemented and validated a robust Holm-Bonferroni step-down algorithm in `src/phase14_6/statistics.py`.

### Pre-Registered Hypothesis Family ($m = 2$):
1. **$H_{14-3}$ (Retrieval Pruning Accuracy Gain)**:
   - Null Hypothesis: $\mathbb{E}[\text{Acc}_{B14-B} - \text{Acc}_{B14-A}] \le 0$
   - Observed Mean Delta: $+8.36$ percentage points
   - Family Cluster 95% CI: $[+3.64\%, +13.45\%]$
   - Raw Empirical $p$-value: $p_{(2)} = 0.0002$
2. **$H_{14-4}$ (Grounding Hallucination Reduction)**:
   - Null Hypothesis: $\mathbb{E}[\text{UAR}_{B14-A} - \text{UAR}_{B14-C}] \le 0$
   - Observed Mean Delta: $-21.09$ percentage points
   - Family Cluster 95% CI: $[-28.00\%, -14.18\%]$
   - Raw Empirical $p$-value: $p_{(1)} = 0.0000$ (reported boundedly as $p < 0.0001$)

### Step-Down Holm Adjustment ($m = 2$):
- **Rank 1**: Test $H_{14-4}$ ($p_{(1)} < 0.0001$)
  $$\tilde{p}_{(1)} = \min(1.0, 2 \times p_{(1)}) < 0.0002$$
- **Rank 2**: Test $H_{14-3}$ ($p_{(2)} = 0.0002$)
  $$\tilde{p}_{(2)} = \min(1.0, \max(\tilde{p}_{(1)}, 1 \times 0.0002)) = 0.0002$$

### Multiplicity Verdict:
Both hypotheses retain strong statistical significance after Holm-Bonferroni multiplicity correction:
- $H_{14-4}$: $p_{\text{adj}} < 0.0002 \ll 0.05$ $\implies$ **REJECT NULL (SUPPORTED)**
- $H_{14-3}$: $p_{\text{adj}} = 0.0002 \ll 0.05$ $\implies$ **REJECT NULL (SUPPORTED)**

---

## 5. Test Evidence
`tests/test_phase14_6_statistics.py` verified:
- `test_holm_bonferroni_empty_and_single`: PASS
- `test_holm_bonferroni_known_case`: PASS (Verified textbook example)
- `test_holm_bonferroni_order_preservation`: PASS (Preserves original hypothesis ordering)
- `test_holm_bonferroni_monotonicity_and_bounds`: PASS (Monotonicity and $[0, 1]$ bounds verified)
- `test_holm_bonferroni_phase14_hypothesis_family`: PASS (Both adjusted p-values confirmed $< 0.01$)
- `test_cluster_bootstrap_family_paired`: PASS (Cluster pairing logic confirmed)

---

## 6. Remediation Status
- **Discrepancy P3-01**: **RESOLVED**
- **Residual Limitation**: While Holm-Bonferroni correction is verified and both hypotheses remain supported, inferential conclusions remain subject to the finite cluster degrees of freedom ($K = 11$ document families).
