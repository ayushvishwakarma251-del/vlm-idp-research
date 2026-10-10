# Quantization Benchmark Audit Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Model Benchmarked**: `HuggingFaceTB/SmolVLM-500M-Instruct`  
**Hardware Device**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.6, SM 8.6)  
**Primary Tables**: `table_04_quantization_matrix.csv`, `table_05_memory.csv`, `table_06_latency.csv`, `table_07_throughput.csv`  

---

## 1. Executive Summary

This forensic audit evaluates the physical quantization benchmark reported in Phase 14, comparing FP16 (Q0), INT8 (Q1), and INT4 NF4 (Q2) configurations on physical CUDA hardware.

| Precision Mode | Backend Framework | Measured Peak VRAM (MB) | VRAM Reduction vs FP16 | Measured Latency (s) | Throughput (tok/s) | Physical Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Q0 (FP16)** | PyTorch Native (`torch.float16`) | 1,111.64 MB | 0.0% (Baseline) | 0.8629 s | 27.81 tok/s | `VERIFIED` |
| **Q1 (INT8)** | `bitsandbytes` (`load_in_8bit`) | 702.01 MB | 36.85% reduction | 2.8021 s | 8.57 tok/s | `VERIFIED` |
| **Q2 (INT4 NF4)** | `bitsandbytes` (`load_in_4bit`, NF4) | 530.95 MB | 52.24% reduction | 1.4551 s | 16.49 tok/s | `VERIFIED` |

---

## 2. Telemetry and Performance Trade-Off Analysis

1. **Memory Compression**:
   - Compressing weights from FP16 to INT8 reduced peak allocated VRAM from $1,111.64\text{ MB}$ to $702.01\text{ MB}$ ($36.9\%$ savings).
   - Compressing to INT4 NF4 reduced peak VRAM to $530.95\text{ MB}$ ($52.2\%$ savings), well within the 6 GB physical ceiling.
2. **De-quantization Overhead and Latency Inversion**:
   - **FP16** exhibits the fastest token generation latency ($0.8629\text{ s}$, $27.81\text{ tok/s}$) because native Tensor Cores compute directly in FP16 without runtime de-quantization.
   - **INT8** is the slowest ($2.8021\text{ s}$, $8.57\text{ tok/s}$), demonstrating the known `bitsandbytes` 8-bit dynamic dequantization latency penalty on consumer laptop architectures.
   - **INT4 (NF4)** achieves a balanced trade-off ($1.4551\text{ s}$, $16.49\text{ tok/s}$), providing $1.9\times$ higher throughput than INT8 while reducing memory by another $171\text{ MB}$.

---

## 3. Physical Hardware Provenance

- All three configurations were physically benchmarked on CUDA device 0 with explicit CUDA synchronization (`torch.cuda.synchronize()`) and peak memory tracking (`torch.cuda.max_memory_allocated()`).
- Source code in `src/phase14/experiments.py` (lines 120–172) executes warmup and multi-iteration averaging across 5 measured inference runs per precision level.
- Recomputed values in `table_04_quantization_matrix.csv`, `table_05_memory.csv`, `table_06_latency.csv`, and `table_07_throughput.csv` match code variables exactly.

---

## 4. Scientific Verdict

- **Quantization Benchmark Status**: **`VERIFIED`** (Legitimate physical GPU benchmark with valid BitsAndBytes integration).
