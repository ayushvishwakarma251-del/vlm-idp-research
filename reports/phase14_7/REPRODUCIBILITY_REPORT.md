# PHASE 14.7 — REPRODUCIBILITY AUDIT REPORT

## 1. Reproducibility Infrastructure Overview

Phase 14.7 established an independent, deterministic audit and recomputation harness:
- **Module**: `src/phase14_7/reporting.py` (`Phase14_7WorkflowEngine`)
- **Execution**: Fully automated end-to-end execution without manual spreadsheet edits or undocumented scripts.
- **Trace Input Lineage**: Directly reads frozen `experiments/phase14/traces/physical_inference_traces.jsonl` (SHA-256: `973ebd3af67e3fee0c401f39b280790f06b05df6ed45adf18e52d3bf8d7bf9cc`).
- **Determinism**: 100% deterministic recomputation across all metric evaluations, permutation distributions, and LOFO splits.

---

## 2. Table & Figure Reproducibility Ledger

| Artifact Path | Format | Verification Status | Provenance & Method |
| :--- | :---: | :---: | :--- |
| `experiments/phase14_7/tables/threshold_sensitivity.csv` | CSV | **VERIFIED** | Full factorial grid sweep $\tau \in [0.30, 0.80] \times 4$ conditions |
| `experiments/phase14_7/tables/threshold_family_effects.csv` | CSV | **VERIFIED** | Clustered $K=11$ family-level outcomes across all 6 thresholds |
| `experiments/phase14_7/tables/family_level_outcomes.csv` | CSV | **VERIFIED** | Clustered $K=11$ family outcomes for conditions B14-A..B14-D |
| `experiments/phase14_7/tables/family_level_statistical_results.csv` | CSV | **VERIFIED** | Exact permutation ($2^{11}=2,048$), Wilcoxon, and sign tests |
| `experiments/phase14_7/audit/claim_evidence_ledger.csv` | CSV | **VERIFIED** | Systematic evidentiary status for all audited claims |
| `experiments/phase14_7/figures/grounded_coverage_threshold_sensitivity.svg` | SVG | **VERIFIED** | Vector line chart of Grounded SUC across thresholds |

---

## 3. Independent Verification Instructions

To re-run the entire Phase 14.7 audit pipeline from scratch in any clean checkout:
```bash
py -c "from pathlib import Path; from src.phase14_7.reporting import Phase14_7WorkflowEngine; engine = Phase14_7WorkflowEngine(Path('experiments/phase14/traces/physical_inference_traces.jsonl')); res = engine.run_complete_audit_pipeline(Path('experiments/phase14_7')); print(res['status'])"
```
Output:
```
Workflow complete: SUCCESS
```
All outputs are reproducible bit-for-bit without external network access or stochastic variation.
