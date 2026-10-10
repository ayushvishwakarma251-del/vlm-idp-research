# Target 7B Model Feasibility and OOM Claim Audit

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Target Model**: `Qwen/Qwen2.5-VL-7B-Instruct`  
**Host Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM)  
**Associated Tables**: `table_03_model_registry.csv`, `table_14_failures.csv`  
**Associated Code**: `src/phase14/experiments.py` (lines 80–106)  

---

## 1. Executive Summary

Phase 14 reported that the target 7B vision-language model (`Qwen2.5-VL-7B-Instruct`) encountered a physical CUDA OutOfMemory (OOM) error under FP16 precision on the RTX 3050 6GB Laptop GPU, establishing that 7B-class models cannot be hosted locally in unquantized 16-bit format within a 6GB budget.

This audit independently verified the exact mechanism of this claim:
1. **Mathematical Feasibility**:
   - `Qwen2.5-VL-7B-Instruct` has approximately 7.61 billion parameters.
   - In unquantized FP16 (2 bytes per parameter), model weights alone consume:
     $$7.61 \times 10^9 \times 2 \approx 15.22 \text{ GB}$$
   - When including vision transformer image patch embeddings and KV cache activation memory, the minimum required VRAM exceeds $16.0 \text{ GB}$.
   - On a hardware budget of $6.0 \text{ GB}$ ($6,144 \text{ MB}$), FP16 execution is mathematically impossible.
2. **Implementation Mechanism**:
   - Rather than downloading 15 GB of model weights across the network to trigger a runtime crash, `src/phase14/experiments.py` simulated this boundary via a synthetic negative control allocation:
     ```python
     _ = torch.empty((16 * 1024 * 1024 * 1024 // 2,), dtype=torch.float16, device="cuda:0")
     ```
   - This physically provoked an actual `torch.OutOfMemoryError` on CUDA device 0:
     `"CUDA out of memory. Tried to allocate 16.00 GiB on 6.00 GiB GPU."`
   - The exception was caught, documented, and registered as a legitimate hardware capacity boundary.

---

## 2. Audit Assessment and Scientific Qualification

- **Physical Feasibility Boundary**: **`VERIFIED`**. Unquantized 7B models cannot physically execute within 6 GB VRAM.
- **Reporting Nuance**: In `reports/phase14/PHASE14_REPORT.md` and Table 14, this event was described as a "physical CUDA OOM for Qwen2.5-VL-7B-Instruct".
- **Finding (P2 Qualification)**: While the physical VRAM limitation is indisputable, the audit notes that the OOM was triggered via a calibrated 16 GB synthetic tensor allocation on CUDA, rather than attempting to load weights from the Hugging Face repository.
- **Required Manuscript Correction**: The paper must clarify that unquantized 7B feasibility was evaluated via physical memory capacity bounds and allocation negative controls, rather than runtime weight loading.
