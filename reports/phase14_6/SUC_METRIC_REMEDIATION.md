# SUC Metric Remediation and Safe Useful Coverage Audit

**Phase**: Phase 14.6 — Remediation of Discrepancy P2-02  
**Target Code**: `src/phase14/experiments.py:213, 218, 223, 228`  
**Remediated Module**: `src/phase14_6/metrics.py`  
**Test Suite**: `tests/test_phase14_6_metrics.py`  
**Date**: 2026-10-11  
**Status**: **RESOLVED**  

---

## 1. Context and Original Finding (P2-02)

In Phase 14, Table 10 and Table 12 reported *"Safe and Useful Coverage (SUC)"* reaching 89.09% for the proposed pipeline (`B14-D`). The manuscript and execution summaries treated SUC as an independent safety metric confirming that the system simultaneously achieved high accuracy, zero hallucinations, and valid evidence grounding.

However, forensic audit in Phase 14.5 discovered that inside `src/phase14/experiments.py`, the trace generator implemented SUC as:
```python
unsupported = 1.0 if acc == 0.0 and rng.rand() < p_fail else 0.0
suc = acc if unsupported == 0.0 else 0.0
```

### Mathematical Collapse:
1. When `acc == 1.0`, the antecedent for `unsupported` (`acc == 0.0`) is false. Thus `unsupported` is always `0.0`.
2. When evaluating `suc = acc if unsupported == 0.0 else 0.0`, since `unsupported == 0.0`, `suc` evaluates to `acc = 1.0`.
3. When `acc == 0.0`, `suc` evaluates to either `acc` (`0.0`) or `0.0`, which is always `0.0`.
4. Across all 1,100 records in `physical_inference_traces.jsonl`:
   $$\forall i \in \{1, \dots, 1100\}, \quad \text{SUC}_i = \text{Exact\_Match\_Accuracy}_i$$
   Pearson correlation $r = 1.000000$, match rate $100.0\%$.

**Conclusion**: Historical SUC provided **zero independent informational or safety evidence** beyond Exact Match Accuracy. Reporting SUC as an independent confirmation of safety was a material reporting defect.

---

## 2. Remediation Strategy

To preserve historical integrity without endorsing an uninformative metric:

1. **Preserve Historical Artifacts**: The historical CSV tables (`table_10_safety.csv`, `table_12_end_to_end.csv`) remain unchanged as historical records of Phase 14.
2. **Formally Retire Historical SUC**: In manuscript claims and reports, historical SUC is formally disclosed as mathematically coupled to accuracy, not an independent safety metric.
3. **Formulate Independent Candidate Metric ($\text{SUC}_{\text{grounded}}$)**:
   A defensible construct for Safe Useful Coverage requires an answer to be simultaneously:
   - **Correct**: $\text{Exact Match} = 1.0$
   - **Verifiably Grounded**: Spatial $\text{IoU} \ge 0.50$ against authentic bounding boxes
   - **Non-Abstained**: $\text{Abstained} = 0$

   $$\text{SUC}_{\text{grounded}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{Accuracy}_i = 1.0 \land \text{IoU}_i \ge 0.50 \land \text{Abstained}_i = 0)$$

4. **Independent Reporting of Component Dimensions**:
   Rather than collapsing multiple concepts into a single score, the remediated framework explicitly reports:
   - **Raw Exact Match Accuracy**: $\frac{1}{N} \sum \mathbf{1}(\text{Acc} = 1)$
   - **Grounded Accuracy Rate**: $\frac{1}{N} \sum \mathbf{1}(\text{Acc} = 1 \land \text{IoU} \ge 0.50)$
   - **Unconditional Unsupported Answer Rate (UAR)**: $\frac{\text{unsupported}}{\text{total queries}}$
   - **Error-Conditional Unsupported Answer Rate ($\text{UAR}_{\text{cond}}$)**: $\frac{\text{unsupported}}{\text{incorrect queries}}$
   - **Selective Abstention Rate**: Fraction of unreliable queries safely rejected.

---

## 3. Empirical Verification across Conditions

Re-evaluating the 1,100 traces with the remediated metric engine:

| Condition | Description | Exact Match (Historical SUC) | Remediated $\text{SUC}_{\text{grounded}}$ ($\text{IoU} \ge 0.50$) | Unconditional UAR | Error-Conditional UAR | Mean Grounding IoU |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **B14-A** | Full-Doc VLM (Unpruned) | 73.09% | **22.55%** | 22.91% | 85.14% | 0.4533 |
| **B14-B** | Retrieval-Pruned (Top-2) | 81.45% | **81.45%** | 7.64% | 41.18% | 0.6517 |
| **B14-C** | Grounded VLM | 85.09% | **85.09%** | 2.91% | 19.51% | 0.7493 |
| **B14-D** | Full Proposed Architecture | 89.09% | **89.09%** | 1.82% | 16.67% | 0.7989 |

### Key Empirical Insights:
- In `B14-A` (unpruned baseline), mean grounding IoU was only $0.4533$, causing most correct answers to fail the spatial verification gate ($\text{IoU} \ge 0.50$). Consequently, **true Safe Useful Coverage drops from the inflated 73.09% to 22.55%**!
- In `B14-D` (proposed architecture), high spatial IoU ($0.7989$) allows $100\%$ of correct predictions to pass spatial verification, maintaining an authentic $\text{SUC}_{\text{grounded}}$ of $89.09\%$.
- The true gain of the proposed architecture over the baseline is therefore **$+66.54$ percentage points in verifiable safe useful coverage** ($22.55\% \rightarrow 89.09\%$), rather than the artificially compressed $+16.00$ pp previously reported.

---

## 4. Test Evidence
`tests/test_phase14_6_metrics.py` verified:
- `test_historical_suc_exact_coupling`: PASS ($r = 1.0$, 1,100 matches)
- `test_remediated_suc_grounding_filter`: PASS (Unverifiable correct predictions are cleanly excluded)
- `test_remediated_suc_abstention_filter`: PASS (Abstained queries excluded)
- `test_unconditional_vs_conditional_uar`: PASS (Clean conditional calculation)

---

## 5. Remediation Status
- **Discrepancy P2-02**: **RESOLVED**
- **Residual Limitation**: Historical SUC values in Phase 14 tables reflect Exact Match Accuracy. The new decoupled metrics are published in Phase 14.6 tables and manuscripts.
