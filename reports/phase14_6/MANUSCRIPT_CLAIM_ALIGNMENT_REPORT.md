# Manuscript Claim Alignment Report

**Phase**: Phase 14.6 — Manuscript Claim Alignment and Ledger Verification  
**Reference Ledger**: `experiments/phase14_6/audit/claim_evidence_ledger.csv`  
**Date**: 2026-10-11  
**Status**: **RESOLVED**  

---

## 1. Executive Summary

This report establishes total alignment between experimental evidence, machine-readable registers, and manuscript claims. Every numerical claim in the research manuscript and reports has been audited for:
1. **Percentage vs Percentage-Point Language**:
   - `+8.36 percentage points` (from $73.09\%$ to $81.45\%$), not $+8.36\%$.
   - `-21.09 percentage points` (from $22.91\%$ to $1.82\%$).
   - `52.24% relative reduction` in peak memory (from $1,111.64\text{ MB}$ to $530.95\text{ MB}$).
2. **Explicit Provenance Labels**:
   - Physical GPU telemetry labeled `PHYSICAL_HARDWARE_BENCHMARK`.
   - 1,100 traces labeled `SIMULATED_END_TO_END`.
   - 16.0 GB OOM probe labeled `ANALYTICAL_NEGATIVE_CONTROL`.
3. **Multiplicity Correction**:
   - Both pre-registered hypothesis tests report both raw empirical p-values ($p = 0.0002$ and $p < 0.0001$) and Holm-adjusted p-values ($p_{\text{adj}} \le 0.0002$).

---

## 2. Complete Claim-to-Evidence Matrix

| Claim ID | Manuscript Location | Numerical Value | Evidence Classification | Statistical Method | Claim Alignment Status |
|:---|:---|:---|:---|:---|:---:|
| **CLM-01** | Section V-B | FP16 Peak VRAM: $1,111.64\text{ MB}$ | `PHYSICAL_HARDWARE_BENCHMARK` | Direct PyTorch CUDA memory profiling | **VERIFIED** |
| **CLM-02** | Section V-B | INT4 Peak VRAM: $530.95\text{ MB}$ ($-52.24\%$), $16.49\text{ tok/s}$ | `PHYSICAL_HARDWARE_BENCHMARK` | Direct PyTorch CUDA memory profiling | **VERIFIED** |
| **CLM-03** | Section V-A | FP16 7B OOM: $>16.0\text{ GiB}$ required | `ANALYTICAL_NEGATIVE_CONTROL` | Analytical model + 16 GB allocation probe | **VERIFIED** |
| **CLM-04** | Section VI-A | Context pruning: 5 pages $\rightarrow$ 2 pages (60% cut), $7.485\text{s} \rightarrow 3.750\text{s}$ | `SIMULATED_END_TO_END` | Context-token scaled parametric formula | **VERIFIED** |
| **CLM-05** | Section VI-A | Exact Match gain: $+8.36\text{ pp}$ ($73.09\% \rightarrow 81.45\%$) | `SIMULATED_END_TO_END` | Cluster bootstrap ($B=10k$, $p_{\text{adj}}=0.0002$) | **VERIFIED** |
| **CLM-06** | Section VI-B | Grounding IoU: $0.4533 \rightarrow 0.7989$ ($+0.3456$) | `SIMULATED_END_TO_END` | Multi-condition parametric simulation | **VERIFIED** |
| **CLM-07** | Section VI-C | UAR reduction: $22.91\% \rightarrow 1.82\%$ ($-21.09\text{ pp}$) | `SIMULATED_END_TO_END` | Cluster bootstrap ($B=10k$, $p_{\text{adj}}<0.0002$) | **VERIFIED** |
| **CLM-08** | Section VI-D | Remediated Grounded SUC ($\text{IoU} \ge 0.50$): $22.55\% \rightarrow 89.09\%$ ($+66.54\text{ pp}$) | `DERIVED_METRIC` | Independent multi-criteria computation | **VERIFIED** |

---

## 3. Alignment Status Summary
- **Total Claims Audited**: 8 major numerical claims
- **Physically Verified**: 2 (Quantization VRAM & throughput)
- **Analytically Established**: 1 (7B FP16 memory barrier)
- **Simulation Supported**: 4 (End-to-end multi-page retrieval, accuracy, IoU, UAR)
- **Derived / Remediated Metric**: 1 (Safe Useful Coverage)
- **Contradictions Remaining**: **0**

---

## 4. Remediation Status
- **Claim Alignment**: **RESOLVED**
- **Residual Limitation**: Historical claims are now fully qualified with strict evidence boundaries in the updated publication register.
