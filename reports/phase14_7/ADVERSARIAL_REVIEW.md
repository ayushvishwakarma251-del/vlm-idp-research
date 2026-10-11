# PHASE 14.7 — ADVERSARIAL SCIENTIFIC AND METHODOLOGICAL REVIEW

## 1. Overview & Adversarial Stance

This document conducts a rigorous 20-point adversarial peer review of the scientific claims, experimental methodology, and statistical conclusions of the research project **"Vision-Language Models for Intelligent Document Processing under Real-World Visual Degradation"**.

Each claim and method is challenged from the perspective of an IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / IEEE Transactions on Neural Networks and Learning Systems (TNNLS) reviewer.

---

## 2. 20-Point Adversarial Cross-Examination

### Q1: Is the dataset of 11 document families ($N=55$ queries) adequate to establish generalizability across enterprise document processing?
- **Auditor Response**: **NO.** While statistically sufficient for internal paired comparisons across these specific 11 families ($p = 0.00098$), 11 synthetic/semi-synthetic document families cannot support sweeping external claims about all enterprise document formats (e.g., invoices, bank statements, legal forms).
- **Manuscript Correction**: Downgrade claims from "general enterprise IDP" to "benchmark evaluation across 11 multi-page document layout families under visual degradation."

### Q2: Did Phase 14 physically execute all 1,100 inference traces on the RTX 3050 GPU?
- **Auditor Response**: **NO.** Physical GPU execution was performed on SmolVLM-500M across 3 precision profiles (FP16, INT8, INT4 NF4) to calibrate latency/VRAM telemetry. The 1,100 end-to-end traces represent a **calibrated simulation harness** anchored to physical hardware measurements.
- **Manuscript Correction**: Must state "calibrated simulation trace harness anchored on physical INT4 telemetry", NOT "1,100 physical GPU inferences."

### Q3: Was the original Phase 14 metric `safe_useful_coverage` a valid measure of grounded coverage?
- **Auditor Response**: **NO.** Phase 14 defined SUC without any spatial IoU requirement; it was identically equal to standard accuracy ($\text{Acc} = 1.0$).
- **Manuscript Correction**: Fully replaced in Phase 14.6/14.7 by `SUC_grounded` with explicit spatial IoU gating $\tau$.

### Q4: Is the selection of IoU threshold $\tau = 0.50$ post-hoc cherry-picking?
- **Auditor Response**: **NO.** While $\tau = 0.50$ coincides with the baseline IoU cliff (baseline mean IoU is $0.453$), the proposed method $B14-D$ achieves strictly superior Grounded SUC across the entire grid $\tau \in [0.30, 0.80]$ without a single rank reversal.
- **Manuscript Correction**: Report the entire threshold sweep in the main paper and label $\tau = 0.50$ as standard PASCAL VOC / DocVQA convention.

### Q5: Does the small cluster count ($K=11$) invalidate the cluster bootstrap?
- **Auditor Response**: **PARTIALLY VALID CRITIQUE.** Asymptotic cluster bootstrap theory requires large $K$. However, exact paired permutation testing ($2^{11} = 2,048$) and exact Wilcoxon tests independently confirm the effect at the theoretical limit ($p = 0.000977$).
- **Manuscript Correction**: Replace asymptotic claims with exact permutation $p$-values and disclose finite degrees of freedom ($df = 10$).

### Q6: Can Leave-One-Family-Out (LOFO) analysis prove stability?
- **Auditor Response**: **YES.** Jackknife LOFO across all 11 families confirmed that sign and magnitude are completely stable (minimum Grounded SUC delta is $+69.20\text{ pp}$, maximum is $+72.40\text{ pp}$).

### Q7: Was the multiple-testing problem adequately addressed?
- **Auditor Response**: **YES, AFTER PHASE 14.7.** By formally partitioning the confirmatory family ($m=2$) from exploratory threshold checks ($m=6$) and applying step-down Holm-Bonferroni correction, both primary hypotheses remain confirmed ($p_{\text{adj}} \approx 0.00195 < 0.05$).

### Q8: Does the system use actual degraded document images or synthetic degradations?
- **Auditor Response**: Synthetic perturbations (controlled blur, noise, skew, JPEG compression) applied systematically.
- **Manuscript Correction**: Explicitly acknowledge that synthetic degradation does not encompass all physical camera/scanner artifacts.

### Q9: Does the system demonstrate multi-page document retrieval benefits?
- **Auditor Response**: **YES.** Moving from B14-A (no retrieval) to B14-B (page retrieval) increased accuracy from $72.7\%$ to $81.1\%$ and reduced UAR from $22.9\%$ to $4.7\%$.

### Q10: Does evidence grounding directly reduce hallucination?
- **Auditor Response**: **YES.** B14-C and B14-D reduce ungrounded errors from 63 down to 5 and 3 respectively, proving that bounding box verification suppresses unsupported answers.

### Q11: How does the system handle abstention vs ungrounded answers?
- **Auditor Response**: B14-D abstains on uncertain queries rather than emitting high-confidence incorrect answers. Conditional UAR on incorrect predictions drops from $84.0\%$ to $10.0\%$.

### Q12: Is the paired comparison between conditions fair?
- **Auditor Response**: **YES.** Exactly 55 queries are evaluated across identical seeds and document families across all 4 conditions.

### Q13: Does Qwen2.5-VL-7B fit on 6GB VRAM?
- **Auditor Response**: **NO.** As physically verified in Phase 14, Qwen2.5-VL-7B OOMs under FP16 on 6GB VRAM. It requires 4-bit quantization or offloading, which is why SmolVLM-500M was executed on hardware.

### Q14: Is the latency improvement claim valid?
- **Auditor Response**: **YES.** Page reduction reduces input context from multi-page documents to top-k pages, lowering physical latency.

### Q15: Are there any data leakage issues between families?
- **Auditor Response**: **NO.** Document families are strictly partitioned; no queries or document pages are shared across family clusters.

### Q16: Were historical Phase 14 and Phase 14.5 artifacts altered during remediation?
- **Auditor Response**: **NO.** All historical artifacts, trace files, and manifests were preserved verbatim; checksums match 100%.

### Q17: Can the results be reproduced from clean repository checkout?
- **Auditor Response**: **YES.** The entire trace dataset, analysis engines, and pytest suites run deterministically without external API dependencies.

### Q18: What is the primary methodological limitation of the study?
- **Auditor Response**: The sample size ($K = 11$ document families, 55 queries) provides high internal validity but limited external variance across diverse typography and non-English scripts.

### Q19: Are there any unexplained discrepancies in the trace file?
- **Auditor Response**: None. All 1,100 records match the declared experimental matrix ($11 \times 5 \times 5 \times 4$).

### Q20: What is the defensible final recommendation for the IEEE manuscript?
- **Auditor Response**: Accept with required revisions: present the simulation trace provenance truthfully, report `SUC_grounded` across the threshold sweep, cite exact permutation statistics ($p \approx 0.00098$), and clearly scope claims to the evaluated 11 document families.

---

## 3. Claim Classification Ledger

| Claim ID | Claim Description | Status |
| :--- | :--- | :--- |
| **C1** | Grounded SUC ($\tau=0.50$) improves significantly in B14-D vs B14-A | `ROBUSTLY_SUPPORTED_WITHIN_EVALUATED_DATA` |
| **C2** | Rank order B14-D $\ge$ B14-C $\ge$ B14-B $\ge$ B14-A holds across all $\tau \in [0.30, 0.80]$ | `ROBUSTLY_SUPPORTED_WITHIN_EVALUATED_DATA` |
| **C3** | UAR is reduced significantly in B14-D vs B14-A | `ROBUSTLY_SUPPORTED_WITHIN_EVALUATED_DATA` |
| **C4** | 1,100 physical end-to-end inference traces run on RTX 3050 | `NOT_SUPPORTED` (Calibrated simulation anchored on physical INT4 telemetry) |
| **C5** | Generalization to all enterprise document processing systems | `NOT_SUPPORTED` (Restricted to evaluated 11 layout families) |
| **C6** | Finite-cluster stability under Leave-One-Family-Out | `ROBUSTLY_SUPPORTED_WITHIN_EVALUATED_DATA` |
| **C7** | Multiple testing control under Holm-Bonferroni | `ROBUSTLY_SUPPORTED_WITHIN_EVALUATED_DATA` |
