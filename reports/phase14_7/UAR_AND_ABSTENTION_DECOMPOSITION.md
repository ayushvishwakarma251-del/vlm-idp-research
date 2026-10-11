# PHASE 14.7 — UNSUPPORTED ANSWER RATE (UAR) AND ABSTENTION DECOMPOSITION AUDIT

## 1. Executive Summary & Objective

The Unsupported Answer Rate (UAR) measures the proportion of model answers that lack faithful grounding in the visual document evidence (hallucinations). In Intelligent Document Processing (IDP), an incorrect answer that is accompanied by fabricated or absent bounding boxes poses a high operational risk.

This audit executes a complete forensic breakdown of:
1. **Unconditional UAR**: Hallucinations evaluated over the entire dataset population ($N = 275$ queries per condition).
2. **Conditional UAR**: Hallucinations evaluated strictly over incorrect answers ($\text{Acc} = 0$).
3. **Safe Abstention Interaction**: Distinguishing between silent, ungrounded failures vs safe model abstentions or ungrounded non-hallucinatory errors.

---

## 2. Population Breakdown Across Conditions ($N = 275$ per Condition)

| Condition | Description | Total Queries ($N$) | Correct Answers ($\text{Acc}=1$) | Incorrect Answers ($\text{Acc}=0$) | Hallucinated Answers ($\text{UAR}=1$) | Unconditional UAR | Conditional UAR ($\text{UAR} \mid \text{Acc}=0$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B14-A** | Vanilla VLM Baseline | 275 | 200 (72.7%) | 75 (27.3%) | 63 | **22.91%** | **84.00%** (63 / 75) |
| **B14-B** | Page Retrieval Only | 275 | 223 (81.1%) | 52 (18.9%) | 13 | **4.73%** | **25.00%** (13 / 52) |
| **B14-C** | Page + Spatial Grounding | 275 | 237 (86.2%) | 38 (13.8%) | 5 | **1.82%** | **13.16%** (5 / 38) |
| **B14-D** | Grounding + Uncertainty/Abstain | 275 | 245 (89.1%) | 30 (10.9%) | 3 | **1.09%** | **10.00%** (3 / 30) |

---

## 3. Key Forensic Insights

### 3.1 Unconditional vs Conditional Hallucination Dynamics
1. **Baseline Vulnerability (B14-A)**:
   - When B14-A fails (75 incorrect queries), **84.00% of those failures** (63/75) produce completely ungrounded, hallucinated answers.
   - Overall population hallucination rate is $22.91\%$.
2. **Mitigation by Grounding (B14-C)**:
   - Adding spatial evidence verification collapses ungrounded errors from 63 down to 5 ($1.82\%$ unconditional UAR), representing a **92.1% reduction in total hallucinations**.
3. **Synergy in Proposed System (B14-D)**:
   - Integrating uncertainty estimation and selective abstention further suppresses hallucinations down to 3 cases ($1.09\%$ unconditional UAR).
   - Among the remaining 30 incorrect predictions in B14-D, 27 are non-hallucinatory incorrect values (e.g., misread digits where bounding boxes correctly identified the target zone), leaving only 3 genuinely ungrounded outputs ($10.00\%$ conditional UAR).

### 3.2 True Safe and Useful Coverage (SUC) Interaction
- Historically in Phase 14, `safe_useful_coverage` was computed identically to accuracy ($\text{Acc} = 1.0$) because no spatial IoU gate was enforced.
- Under the Phase 14.6/14.7 definition:
  $$\text{SUC}_{\text{grounded}}(\tau) = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\big(\text{Acc}_i = 1 \land \text{IoU}_i \ge \tau \land \text{Abstain}_i = 0\big)$$
- At standard $\tau = 0.50$:
  - B14-A Grounded SUC: $18.18\%$ (despite $72.73\%$ accuracy, only 50 correct answers had $\text{IoU} \ge 0.50$).
  - B14-D Grounded SUC: $89.09\%$ (all 245 correct answers achieved $\text{IoU} \ge 0.50$, with mean IoU $0.799$).
  - Delta: **+70.91 pp improvement**.

---

## 4. Auditor's Verdict on UAR and Abstention Claims

> [!NOTE] UAR and Abstention Audit Verdict
> The claims regarding hallucination reduction and evidence verification are **empirically substantiated**:
> - Baseline B14-A generates 63 ungrounded answers out of 75 failures (84.0% failure hallucination rate).
> - Proposed B14-D reduces ungrounded answers to 3 (a 95.2% reduction in ungrounded errors).
> - The distinction between unconditional UAR ($22.91\% \to 1.09\%$) and error-conditional UAR ($84.00\% \to 10.00\%$) must be explicitly stated in the IEEE manuscript to prevent reviewer confusion regarding the denominator.
