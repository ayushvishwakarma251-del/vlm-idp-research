# Figure-to-Table Consistency Audit Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Evaluated Figures**: 12 Figures (`experiments/phase14/figures/fig_01_*.png` to `fig_12_*.png`)  
**Evaluated Tables**: 15 Tables (`experiments/phase14/tables/table_01_*.csv` to `table_15_*.csv`)  
**Mapping Registry**: `experiments/phase14_5/audit/figure_to_table_mapping.json`  

---

## 1. Executive Summary

This forensic audit cross-referenced every visual representation generated in Phase 14 against its underlying data source in the tabular and trace registries.

- **Total Figures Audited**: 12 publication-grade figures.
- **Total Tables Audited**: 15 CSV tables.
- **Trace Source**: `experiments/phase14/traces/physical_inference_traces.jsonl`.
- **Consistency Verification**: **100% Concordance** across all 12 figures. Every plotted bar height, scatter point, error bar, and Pareto curve corresponds without discrepancy to its counterpart in the CSV tables and raw traces.

---

## 2. Cross-Reference Ledger

| Figure File | Plotted Concept | Direct Source Tables | Underlying Values | Verification Status |
|:---|:---|:---|:---|:---:|
| `fig_01_cuda_vram_by_precision.png` | Peak CUDA VRAM | `table_04`, `table_05` | FP16: 1111.64MB, INT8: 702.01MB, INT4: 530.95MB | `VERIFIED` |
| `fig_02_e2e_latency_by_precision.png` | Forward Latency | `table_04`, `table_06` | FP16: 0.863s, INT8: 2.802s, INT4: 1.455s | `VERIFIED` |
| `fig_03_throughput_by_precision.png` | Generation Throughput | `table_04`, `table_07` | FP16: 27.81 tok/s, INT8: 8.57, INT4: 16.49 | `VERIFIED` |
| `fig_04_model_loading_time.png` | Model Loading Time | `table_04` | FP16: 2.154s, INT8: 3.382s, INT4: 1.382s | `VERIFIED` |
| `fig_05_pareto_vram_vs_latency.png` | VRAM vs Latency Pareto | `table_04` | 3 precision points mapped onto 2D frontier | `VERIFIED` |
| `fig_06_context_scaling_latency.png`| Latency Scaling | `table_11`, `table_12` | 5 pages: 7.485s; 2 pages: 3.750s | `VERIFIED` |
| `fig_07_accuracy_by_condition.png` | Exact Match Accuracy | `table_08`, `table_12` | B14-A: 72.7%, B14-B: 81.1%, B14-C: 86.2%, B14-D: 89.1% | `VERIFIED` |
| `fig_08_grounding_iou_by_condition.png`| Grounding IoU | `table_09`, `table_12` | B14-A: 0.453, B14-B: 0.648, B14-C: 0.755, B14-D: 0.799 | `VERIFIED` |
| `fig_09_unsupported_answer_rate.png`| Unsupported Rate (UAR)| `table_10`, `table_12` | B14-A: 22.9%, B14-B: 4.7%, B14-C: 1.8%, B14-D: 1.1% | `VERIFIED` |
| `fig_10_safe_useful_coverage.png` | Safe Useful Coverage | `table_10`, `table_12` | B14-A: 72.7%, B14-B: 81.1%, B14-C: 86.2%, B14-D: 89.1% | `VERIFIED` |
| `fig_11_latency_by_modality.png` | Modality Latency | `table_12`, traces | Modality-stratified boxplots (3.5s to 7.8s) | `VERIFIED` |
| `fig_12_precision_quality_tradeoff.png`| Footprint vs Quality | `table_04`, `table_12` | Combined INT4 footprint with condition accuracies | `VERIFIED` |

---

## 3. Visual Artifact Integrity Note

In `fig_10_safe_useful_coverage.png`, the plotted bars are identical to `fig_07_accuracy_by_condition.png`. As documented in `METRIC_DEFINITIONS_AUDIT.md`, this is faithful to `table_10_safety.csv` and `table_12_end_to_end.csv` where SUC was identically equal to Exact Match Accuracy due to coupling in `src/phase14/experiments.py`.

---

## 4. Scientific Verdict

- **Figure-Table Consistency Status**: **`VERIFIED`** (No discrepancies between visual figures and empirical CSV data).
