# Figure and Table Regeneration Report

**Phase**: Phase 14.6 — Versioned Table and Figure Pipeline  
**Target Tables**: `experiments/phase14_6/tables/`  
**Target Figures**: `experiments/phase14_6/figures/`  
**Date**: 2026-10-11  
**Status**: **RESOLVED**  

---

## 1. Executive Summary

Phase 14.6 established an independent, deterministic reporting pipeline (`src/phase14_6/reporting.py`) that loads frozen source traces, computes both historical and remediated metrics, applies multiplicity-corrected statistical inference, and exports versioned tables and figures.

Historical Phase 14 tables in `experiments/phase14/tables/` remain **strictly unchanged**. All new reporting artifacts reside in `experiments/phase14_6/`.

---

## 2. Inventory of Regenerated Tables

| Table File | Core Subject | Execution Class | Key Remediation Applied |
|:---|:---|:---|:---|
| **`table_04_quantization_matrix_remediated.csv`** | Quantization Benchmark | `PHYSICAL_HARDWARE_BENCHMARK` | Explicitly attributes physical latency and memory to SmolVLM-500M on RTX 3050. |
| **`table_12_end_to_end_remediated.csv`** | Multi-Condition E2E Summary | `SIMULATED_END_TO_END` | Discloses historical SUC coupling to Accuracy; adds Remediated Grounded SUC ($\text{IoU} \ge 0.50$); distinguishes conditional vs unconditional UAR. |
| **`table_13_statistical_tests_remediated.csv`** | Hypothesis Testing | `SIMULATED_END_TO_END` | Reports independent cluster units ($K = 11$); computes and displays both raw and Holm-Bonferroni adjusted p-values. |
| **`table_14_failures_remediated.csv`** | Feasibility & Failure Modes | `ANALYTICAL_NEGATIVE_CONTROL` / `PHYSICAL_MODEL_INFERENCE` | Accurately describes 16 GB allocation probe as a synthetic negative control rather than checkpoint loading. |

---

## 3. Inventory of Regenerated Figures

| Figure File | Subject | Execution Class | Provenance Disclosure Banner |
|:---|:---|:---|:---|
| **`figure_01_remediated_suc_comparison.svg`** | Remediated Grounded SUC vs Accuracy | `SIMULATED_END_TO_END` | Embedded header clearly displays: *"Execution Class: SIMULATED_END_TO_END (Anchored on CUDA INT4 Telemetry)"*. |

---

## 4. Remediation Status
- **Figure & Table Pipeline**: **RESOLVED**
- **Residual Limitation**: Outputs reflect deterministic recomputation from frozen traces. No historical files were modified or overwritten.
