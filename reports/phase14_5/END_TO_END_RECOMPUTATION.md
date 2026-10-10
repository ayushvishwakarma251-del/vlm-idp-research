# End-to-End Metrics Independent Recomputation Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Source Data**: `experiments/phase14/traces/physical_inference_traces.jsonl` ($N = 1,100$)  
**Baseline Comparison**: `experiments/phase14/tables/table_12_end_to_end.csv`  
**Output Table**: `experiments/phase14_5/tables/recomputed_end_to_end.csv`  

---

## 1. Executive Summary

An independent, clean-room recomputation of all Phase 14 end-to-end evaluation metrics was executed across conditions B14-A, B14-B, B14-C, and B14-D. Every metric was recomputed directly from the 1,100 raw JSON Lines trace records and compared with historical Table 12.

| Metric | Condition | Recomputed Mean | Historical Table 12 | Absolute Difference | Relative Error | Reproducibility Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Exact Match** | B14-A | 0.727273 | 0.727273 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-B | 0.810909 | 0.810909 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-C | 0.861818 | 0.861818 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-D | 0.890909 | 0.890909 | 0.000000 | 0.00% | `VERIFIED` |
| **Grounding IoU** | B14-A | 0.452744 | 0.452744 | $< 10^{-16}$ | 0.00% | `VERIFIED` |
| | B14-B | 0.647733 | 0.647733 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-C | 0.755301 | 0.755301 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-D | 0.799357 | 0.799357 | $< 10^{-16}$ | 0.00% | `VERIFIED` |
| **Unsupported Rate (UAR)** | B14-A | 0.229091 | 0.229091 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-B | 0.047273 | 0.047273 | $< 10^{-16}$ | 0.00% | `VERIFIED` |
| | B14-C | 0.018182 | 0.018182 | $< 10^{-16}$ | 0.00% | `VERIFIED` |
| | B14-D | 0.010909 | 0.010909 | $< 10^{-16}$ | 0.00% | `VERIFIED` |
| **Safe Useful Coverage (SUC)** | B14-A | 0.727273 | 0.727273 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-B | 0.810909 | 0.810909 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-C | 0.861818 | 0.861818 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-D | 0.890909 | 0.890909 | 0.000000 | 0.00% | `VERIFIED` |
| **Latency (s)** | B14-A | 7.485123 | 7.485123 | 0.000000 | 0.00% | `VERIFIED` |
| | B14-B | 3.751716 | 3.751716 | $< 10^{-15}$ | 0.00% | `VERIFIED` |
| | B14-C | 3.750525 | 3.750525 | $< 10^{-15}$ | 0.00% | `VERIFIED` |
| | B14-D | 3.749405 | 3.749405 | $< 10^{-15}$ | 0.00% | `VERIFIED` |
| **Page Reduction (%)** | B14-A | 0.0% | 0.0% | 0.000000 | 0.00% | `VERIFIED` |
| | B14-B..D | 60.0% | 60.0% | 0.000000 | 0.00% | `VERIFIED` |

---

## 2. Statistical Dispersion and Uncertainty

Recomputed standard deviations and standard errors of the mean (SEM, $N = 275$ per condition):

| Condition | Metric | Mean | Standard Deviation (Std) | Standard Error (SEM) | 95% Confidence Interval |
|:---|:---|:---:|:---:|:---:|:---:|
| **B14-A** | Exact Match | 0.7273 | 0.4462 | 0.0269 | $[0.6745, 0.7801]$ |
| | Grounding IoU | 0.4527 | 0.0586 | 0.0035 | $[0.4458, 0.4597]$ |
| | UAR | 0.2291 | 0.4210 | 0.0254 | $[0.1793, 0.2789]$ |
| | Latency (s) | 7.4851 | 0.0760 | 0.0046 | $[7.4761, 7.4941]$ |
| **B14-B** | Exact Match | 0.8109 | 0.3923 | 0.0237 | $[0.7645, 0.8573]$ |
| | Grounding IoU | 0.6477 | 0.0588 | 0.0035 | $[0.6408, 0.6547]$ |
| | UAR | 0.0473 | 0.2126 | 0.0128 | $[0.0221, 0.0725]$ |
| | Latency (s) | 3.7517 | 0.0760 | 0.0046 | $[3.7427, 3.7607]$ |
| **B14-C** | Exact Match | 0.8618 | 0.3457 | 0.0209 | $[0.8209, 0.9027]$ |
| | Grounding IoU | 0.7553 | 0.0388 | 0.0023 | $[0.7507, 0.7599]$ |
| | UAR | 0.0182 | 0.1338 | 0.0081 | $[0.0023, 0.0340]$ |
| | Latency (s) | 3.7505 | 0.0827 | 0.0050 | $[3.7407, 3.7603]$ |
| **B14-D** | Exact Match | 0.8909 | 0.3123 | 0.0188 | $[0.8539, 0.9279]$ |
| | Grounding IoU | 0.7994 | 0.0477 | 0.0029 | $[0.7937, 0.8050]$ |
| | UAR | 0.0109 | 0.1041 | 0.0063 | $[-0.0014, 0.0232]$ |
| | Latency (s) | 3.7494 | 0.0827 | 0.0050 | $[3.7396, 3.7592]$ |

---

## 3. Scientific Verdict

All numbers in Table 12 are **100% reproducible** from `physical_inference_traces.jsonl`.
Scientific status: **`VERIFIED`**.
