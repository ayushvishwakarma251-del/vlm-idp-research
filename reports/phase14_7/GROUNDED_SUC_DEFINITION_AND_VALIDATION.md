# Grounded SUC Construct Definition and Denominator Validation Report

**Phase**: Phase 14.7 — Metric Formalization & Denominator Fairness Audit  
**Target Code**: `src/phase14_7/metric_validation.py`  
**Test Suite**: `tests/test_phase14_7_metrics.py`  
**Date**: 2026-10-11  
**Status**: **VALIDATED**  

---

## 1. Deconstruction of the "Safe and Useful Coverage" Construct

In Phase 14, the metric labeled *"Safe and Useful Coverage"* (SUC) was mathematically degenerate because it was identical to Exact Match Accuracy. In Phase 14.6, an independent spatial grounding requirement was added. Phase 14.7 formally dissects this construct into distinct mathematical dimensions to prevent semantic conflation:

| Dimension | Formal Mathematical Definition | Interpretation |
|:---|:---|:---|
| **Raw Coverage ($C$)** | $\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{1}(A_i = 1)$ | Fraction of eligible queries where the model attempts an answer rather than abstaining. |
| **Exact Match Accuracy ($\text{Acc}$)** | $\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{1}(E_i = 1)$ | Unconditional accuracy across all eligible queries. |
| **Conditional Accuracy ($\text{Acc}_{\text{cond}}$)** | $\frac{\sum \mathbf{1}(E_i = 1 \land A_i = 1)}{\sum \mathbf{1}(A_i = 1)}$ | Accuracy strictly among answered queries. |
| **Spatial Grounded Coverage ($G_{\text{spatial}}(\tau)$)** | $\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{1}(I_i \ge \tau \land A_i = 1)$ | Fraction of eligible queries yielding an answer with spatial $\text{IoU} \ge \tau$, regardless of semantic correctness. |
| **Correct Grounded Coverage ($\text{SUC}_{\text{grounded}}(\tau)$)** | $\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{1}(E_i = 1 \land I_i \ge \tau \land A_i = 1)$ | Fraction of eligible queries yielding a **simultaneously correct, non-abstained, and spatially grounded answer**. |
| **Unconditional UAR ($\text{UAR}_{\text{uncond}}$)** | $\frac{1}{|\mathcal{E}|} \sum_{i \in \mathcal{E}} \mathbf{1}(\text{Unsupported}_i = 1)$ | Unsupported/hallucinated answers relative to the entire query corpus. |
| **Error-Conditional UAR ($\text{UAR}_{\text{err}}$)** | $\frac{\sum \mathbf{1}(E_i = 0 \land \text{Unsupported}_i = 1)}{\sum \mathbf{1}(E_i = 0)}$ | Frequency with which an incorrect prediction is an unsupported hallucination. |

---

## 2. Denominator Fairness and Query Eligibility Audit

A critical flaw in benchmark evaluations is comparing conditions across differing or filtered query populations without disclosure.

### Audit Findings from `experiments/phase14/traces/physical_inference_traces.jsonl`:
- **Query Universe**: Exactly 55 authentic document queries ($11 \text{ families} \times 5 \text{ queries/family}$).
- **Seeds**: Exactly 5 evaluation seeds ($42, 123, 456, 789, 101112$).
- **Eligible Evaluated Records**: Exactly $N = 275$ queries per condition.
- **Fairness Check**: 
  - `B14-A`: 55 unique queries (275 evaluations)
  - `B14-B`: 55 unique queries (275 evaluations)
  - `B14-C`: 55 unique queries (275 evaluations)
  - `B14-D`: 55 unique queries (275 evaluations)
- **Verdict**: **100% PAIRING IDENTITY**. All conditions are evaluated against the exact same query population. Zero selective dropout or population bias was detected.

---

## 3. Empirical Decomposition at Baseline Threshold ($\tau = 0.50$)

| Metric Dimension | B14-A (Full-Doc) | B14-B (Retrieval) | B14-C (Grounded) | B14-D (Proposed) | Net Proposed Gain |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Eligible Queries ($|\mathcal{E}|$)** | 275 | 275 | 275 | 275 | — |
| **Coverage ($A_i=1$)** | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 pp |
| **Exact Match Accuracy** | 72.73% | 81.09% | 86.18% | 89.09% | **+16.36 pp** |
| **Spatial Grounded Coverage ($I_i \ge 0.5$)**| 25.45% | 100.0% | 100.0% | 100.0% | **+74.55 pp** |
| **Correct Grounded Coverage ($\text{SUC}_{0.5}$)**| **18.18%** | **81.09%** | **86.18%** | **89.09%** | **+70.91 pp** |
| **Mean Grounding IoU** | 0.4527 | 0.6477 | 0.7553 | 0.7994 | **+0.3467** |
| **Unconditional UAR** | 22.91% | 4.73% | 1.82% | 1.09% | **-21.82 pp** |
| **Error-Conditional UAR** | 84.00% | 25.00% | 13.16% | 10.00% | **-74.00 pp** |

### Key Scientific Observation:
In condition `B14-A`, although raw accuracy is $72.73\%$, mean grounding IoU is only $0.4527$. Because the majority of predictions in `B14-A` fail the spatial verification gate ($\text{IoU} \ge 0.50$), only $18.18\%$ of queries produce an answer that is simultaneously correct and verifiably grounded. In `B14-D`, $100\%$ of correct predictions pass the spatial gate ($\text{IoU} = 0.7994$), resulting in an authentic $\text{SUC}_{\text{grounded}}$ of $89.09\%$.

---

## 4. Test Verification Summary
Automated tests in `tests/test_phase14_7_metrics.py` verified:
- `test_iou_threshold_exact_boundary`: PASS (Strict boundary adherence verified)
- `test_separation_of_grounding_and_correctness`: PASS (Grounded Coverage vs Correct Grounded Coverage independently checked)
- `test_empty_dataframe_handling`: PASS (Safe zero handling)
- `test_trace_denominator_fairness_on_corpus`: PASS (All conditions share exact identical 55 queries)
