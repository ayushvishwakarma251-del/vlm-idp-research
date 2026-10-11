# Phase 14.7 Initial Forensic Reconnaissance Report

**Audit Phase**: Phase 14.7 — Independent Grounded-Metric Challenge, Small-Cluster Statistical Audit & Robustness Verification  
**Date**: 2026-10-11  
**Author**: Independent Senior ML Researcher & Biostatistical Reviewer  
**Repository**: `vlm-idp-research`  
**Current HEAD**: `c1b2fbe2` (`remediate(phase14.6): discrepancy remediation, metric repair, statistical revalidation & manuscript alignment`)  

---

## 1. Verified Baseline Commits and Git Ancestry

### 1.1 Baseline Commits
- **Phase 14 Baseline**: `583e8a8fa93967a15ddfbdb97d247cfcdbaad922` (`583e8a8f`) $\rightarrow$ **VERIFIED**
- **Phase 14.5 Baseline**: `6d95f8d7b3ca9691c94b79c3e9818820c78a1bc7` (`6d95f8d7`) $\rightarrow$ **VERIFIED**
- **Phase 14.6 Baseline**: `c1b2fbe25041731cbc85bfd2f0b3195b097633e2` (`c1b2fbe2`) $\rightarrow$ **VERIFIED**

### 1.2 Ancestry Chain
The commit ancestry is confirmed strictly linear:
`583e8a8f` (Freeze Phase 14) $\rightarrow$ ... $\rightarrow$ `6d95f8d7` (Phase 14.5 Audit) $\rightarrow$ `db0c994c` $\rightarrow$ `6bf89300` $\rightarrow$ `c1b2fbe2` (Phase 14.6 Remediation, HEAD).

### 1.3 Working Tree Status
- Working tree: **CLEAN** (0 unstaged changes, 0 untracked files).
- Phase 14 trace file (`physical_inference_traces.jsonl`): SHA-256 `973ebd3af67e3fee0c401f39b280790f06b05df6ed45adf18e52d3bf8d7bf9cc` (Identical, 0 modifications).
- Phase 14.6 manifest (`phase14_6_sha256_manifest.json`): 26 / 26 files verified with **0 mismatches**.

---

## 2. Dataset Structure and Sampling Hierarchy

Forensic audit of `experiments/phase14/traces/physical_inference_traces.jsonl` establishes the exact physical structure:

- **Total Trace Records**: 1,100 JSONL records.
- **Independent Document Families ($K$)**: **11** (`fam_auth_42` through `fam_auth_52`).
- **Unique Queries ($N$)**: **55** (exactly 5 queries per document family across all 11 families).
- **Evaluation Seeds ($S$)**: **5** (`42, 123, 456, 789, 101112`).
- **Experimental Conditions ($C$)**: **4** (`B14-A`: Full-doc VLM, `B14-B`: Retrieval-pruned, `B14-C`: Grounded VLM, `B14-D`: Proposed architecture).
- **Design Balance**: Exactly $55 \text{ queries} \times 5 \text{ seeds} = 275 \text{ traces per condition}$. $275 \times 4 = 1,100 \text{ traces}$.
- **Pairing Integrity**: Every single query ID evaluated in `B14-A` is present under the identical seed in `B14-B`, `B14-C`, and `B14-D`. Query eligibility is 100% matched across all four conditions ($N_{\text{eligible}} = 275$ per condition).

---

## 3. Findings on Metric & Statistical Implementation in Phase 14.6

### 3.1 Metric Implementation (`src/phase14_6/metrics.py`)
In Phase 14.6, `RemediatedMetricEngine.compute_grounded_safe_useful_coverage` was defined as:
$$\text{SUC}_{\text{grounded}}(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{Accuracy}_i = 1.0 \land \text{Grounding\_IoU}_i \ge \tau \land \text{Abstained}_i = 0)$$
with fixed threshold $\tau = 0.50$.
- **Adversarial Question 1**: Is $\tau = 0.50$ a fragile decision boundary? In `B14-A`, mean IoU is $0.4533$, which is just below $0.50$. Does this create a sharp threshold cliff where `B14-A` is severely penalized at $\tau = 0.50$ but would recover at $\tau = 0.40$ or $0.30$?
- **Adversarial Question 2**: Is `SUC_grounded` measuring "safety" or strictly "spatial evidence compliance of correct answers"?

### 3.2 Statistical Implementation (`src/phase14_6/statistics.py`)
- Cluster bootstrapping is implemented with $B = 10,000$ resamples over the 11 document families.
- **Adversarial Question 3**: With only $K = 11$ clusters, bootstrap estimates can exhibit substantial Monte Carlo and sample instability. Does the reported statistical significance survive non-parametric exact permutation tests, Wilcoxon signed-rank tests, and Leave-One-Family-Out (LOFO) deletion?

---

## 4. Planned Adversarial Challenge Program

1. **Phase 1: Metric Formalization & Eligibility Audit**:
   Formally separate Accuracy, Grounding IoU Gate, Abstention Gate, and Eligibility.
2. **Phase 2: IoU Threshold Sensitivity Grid**:
   Evaluate $\tau \in \{0.30, 0.40, 0.50, 0.60, 0.70, 0.80\}$ across all 4 conditions. Determine whether condition rankings reverse or if the effect cliff is localized.
3. **Phase 3: Small-Cluster ($K=11$) Family-Level Statistical Audit**:
   Aggregate to the family level; compute Wilcoxon, exact sign tests, exact permutation tests, and Leave-One-Family-Out influence.
4. **Phase 4: Multiple Testing & Confirmatory Family Declarations**:
   Separate the confirmatory $m = 2$ family from the exploratory threshold grid to prevent multiplicity creep.
5. **Phase 5: UAR and Abstention Decomposition**:
   Unpack the exact trade-off between coverage, correctness, spatial localization, and abstention.
6. **Phase 6: Adversarial Scientific Review**:
   Address the 20 adversarial reviewer questions and assign honest evidentiary classifications.
7. **Phase 7 & 8: Reproducibility Pipeline & Regression Tests**:
   Deliver clean, tested Phase 14.7 modules with zero historical regressions.
