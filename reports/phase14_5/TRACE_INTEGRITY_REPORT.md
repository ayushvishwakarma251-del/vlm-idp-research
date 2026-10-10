# Raw Trace Forensic Integrity Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Target File**: `experiments/phase14/traces/physical_inference_traces.jsonl`  
**Associated Source**: `src/phase14/experiments.py`  

---

## 1. Executive Summary

A comprehensive forensic audit of all raw inference traces was executed. The file contains **1,100 JSON Lines records**, spanning 4 experimental conditions and 5 predefined evaluation seeds across authentic document queries.

| Forensic Attribute | Expected Claim | Observed Reality | Audit Status |
|:---|:---:|:---:|:---|
| **Total Records** | 1,100 | 1,100 | `VERIFIED` |
| **JSON Parse Success Rate** | 100% | 1,100 / 1,100 (100%) | `VERIFIED` |
| **Unique Experiment IDs** | 1,100 | 1,100 (0 duplicates) | `VERIFIED` |
| **Conditions Evaluated** | B14-A .. B14-D | 4 (275 records each) | `VERIFIED` |
| **Evaluation Seeds** | 5 seeds | 5 (220 records each) | `VERIFIED` |
| **Document Families** | 11 families | 11 families (100 records each) | `VERIFIED` |
| **Unique Queries** | 55 queries | 55 queries | `VERIFIED` |
| **Negative / NaN / Inf Values** | 0 | 0 | `VERIFIED` |
| **SUC vs Accuracy Concordance** | Independent | 1,100 / 1,100 (100.0% identical) | `DISCREPANCY_FOUND` |
| **Execution Trace Provenance** | Physical Forward Pass | Parametric Anchored Simulation | `QUALIFICATION_REQUIRED` |

---

## 2. Schema and Field Breakdown

Each record in `physical_inference_traces.jsonl` contains 18 fields:

1. `experiment_id` (str): Unique identifier format `EXP14_{condition}_S{seed}_{query_id}`
2. `condition` (str): `B14-A`, `B14-B`, `B14-C`, or `B14-D`
3. `condition_name` (str): Descriptive condition name
4. `seed` (int): 42, 123, 456, 789, or 101112
5. `query_id` (str): Query identifier from authentic document corpus
6. `document_id` (str): Document identifier
7. `family_id` (str): Document family cluster identifier
8. `modality` (str): Document layout/modality category
9. `pages_fed` (int): 5 (for B14-A) or 2 (for B14-B..D)
10. `page_reduction_pct` (float): 0.0% or 60.0%
11. `input_tokens` (int): Context token count (390 or 156)
12. `output_tokens` (int): Output generation token count (24)
13. `physical_latency_s` (float): Total end-to-end response time (s)
14. `peak_vram_mb` (float): VRAM allocation telemetry (MB)
15. `exact_match_accuracy` (float): 0.0 or 1.0
16. `grounding_iou` (float): Spatial grounding intersection-over-union
17. `unsupported_answer_rate` (float): 0.0 or 1.0
18. `safe_useful_coverage` (float): Safe and useful coverage score

---

## 3. Forensic Source Code Analysis (`src/phase14/experiments.py`)

Inspection of the generation routines in `src/phase14/experiments.py` (lines 174–252) establishes the exact relationship between physical hardware measurements and end-to-end traces:

### A. Physical vs. Simulated Components
1. **Physical GPU Telemetry Anchor**: The model `SmolVLM-500M-Instruct` was physically benchmarked on the NVIDIA RTX 3050 Laptop GPU under FP16, INT8, and INT4 (NF4) quantization regimes. Peak VRAM telemetry (`530.95 MB`) was captured from real CUDA hardware via `torch.cuda.max_memory_allocated()`.
2. **Trace Generation Engine**: While the INT4 model was physically loaded onto CUDA prior to trace generation (line 179), the per-query evaluation loop (lines 192–229) generated `exact_match_accuracy`, `grounding_iou`, `unsupported_answer_rate`, and `physical_latency_s` using stochastic parametric functions calibrated to condition-specific expectations, rather than dispatching 1,100 physical multimodal forward passes through PyTorch.

### B. Mathematical Collapse of Safe Useful Coverage (SUC)
In lines 212–228:
```python
unsupported = 1.0 if acc == 0.0 and rng.rand() < threshold else 0.0
suc = acc if unsupported == 0.0 else 0.0
```
- If `acc == 1.0`, then `unsupported` is always `0.0`, resulting in `suc = 1.0`.
- If `acc == 0.0`, then regardless of whether `unsupported` is `1.0` or `0.0`, `suc = 0.0`.
Consequently:
$$\text{SUC} \equiv \text{Exact Match Accuracy}$$
for every single trace. In Table 10 and Table 12, SUC is reported as identical to Exact Match Accuracy because of this implementation coupling, not because abstention or safety filtering was inactive.

---

## 4. Trace Exclusions and Audit Status

- Total trace records: **1,100**
- Excluded records: **0** (All records conform to schema and range constraints)
- `trace_validation.json` and `trace_exclusions.csv` preserved in `experiments/phase14_5/audit/`.
- Scientific Verdict on Raw Traces: **`VERIFIED_WITH_QUALIFICATIONS`** (Structurally valid, internally consistent, anchored to physical GPU memory telemetry, but per-query outputs were parametrically sampled rather than run through 1,100 forward passes).
