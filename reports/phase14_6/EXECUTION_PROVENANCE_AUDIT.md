# Execution Provenance Audit and Classification Report

**Phase**: Phase 14.6 — Remediation of Discrepancy P2-01  
**Target Artifact**: `experiments/phase14/traces/physical_inference_traces.jsonl`  
**Sidecar Manifest**: `experiments/phase14_6/audit/trace_provenance_sidecar.json`  
**Date**: 2026-10-11  
**Status**: **RESOLVED**  

---

## 1. Context and Original Discrepancy (P2-01)

In Phase 14, Section 3 of `src/phase14/experiments.py` ran an end-to-end multi-page benchmark across 4 conditions (`B14-A` through `B14-D`) and 5 seeds (`42, 123, 456, 789, 101112`), generating 1,100 JSONL records written to `physical_inference_traces.jsonl`.

The file name and report summary referred to these records as *"1,100 physical inference traces."* However, forensic audit in Phase 14.5 established that while the peak memory ($530.95\text{ MB}$) and token latency baseline were anchored to physical CUDA telemetry from the BitsAndBytes INT4 NF4 benchmark, individual query predictions were generated via a calibrated parametric distribution (`rng.rand() < threshold` and context-token scaled latency `(0.016 * base_context_tokens) + N(1.25, 0.08)`).

---

## 2. Standardized Scientific Execution Classes

To establish clear, uncompromised provenance across the entire research codebase, Phase 14.6 formalizes six execution classes:

| Execution Class | Formal Definition | Applied Scope in Project |
|:---|:---|:---|
| **`PHYSICAL_MODEL_INFERENCE`** | An authentic neural network model executes forward passes directly on physical hardware for individual evaluation queries. | Reserved for physical single-query runs with verifiable forward-pass logits/tensors. |
| **`PHYSICAL_HARDWARE_BENCHMARK`** | A physical device measures hardware properties (e.g., peak VRAM, load time, raw token throughput) during live execution of a model. | Phase 14 Table 4 benchmarks for SmolVLM-500M across FP16 ($1,111\text{ MB}$), INT8 ($702\text{ MB}$), and INT4 NF4 ($531\text{ MB}$). |
| **`SIMULATED_END_TO_END`** | Evaluation traces generated via a calibrated parametric simulation model anchored to physical hardware telemetry. | **All 1,100 traces** in `physical_inference_traces.jsonl`. |
| **`ANALYTICAL_NEGATIVE_CONTROL`** | A synthetic or mathematical operation executed to probe physical boundary conditions. | Phase 14 16.0 GB CUDA allocation negative control on 6.0 GB RTX 3050. |
| **`DERIVED_METRIC`** | Numerical metrics aggregated, transformed, or bootstrapped from recorded trace outputs. | Tables 8–13, cluster bootstrap statistics, Holm-Bonferroni p-values. |
| **`UNKNOWN_PROVENANCE`** | Records where execution origin cannot be verified from code or logs. | Fallback state; assigned if lineage cannot be established. |

---

## 3. Sidecar Provenance Manifest Audit

Rather than modifying the immutable historical trace file `physical_inference_traces.jsonl` (SHA-256: `973ebd3af67e3fee0c401f39b280790f06b05df6ed45adf18e52d3bf8d7bf9cc`), Phase 14.6 provides a machine-readable sidecar manifest:

`experiments/phase14_6/audit/trace_provenance_sidecar.json`

### Audit Summary:
- **Total Traces Audited**: 1,100 records
- **Classified as `SIMULATED_END_TO_END`**: 1,100 (100.0%)
- **Classified as `PHYSICAL_MODEL_INFERENCE`**: 0 (0.0%)
- **Classified as `UNKNOWN_PROVENANCE`**: 0 (0.0%)
- **Telemetry Anchor**: Physically measured BitsAndBytes 4-Bit NF4 peak VRAM ($530.95\text{ MB}$) on NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- **Parametric Source Reference**: `src/phase14/experiments.py:191-250`

---

## 4. Test Verification
Automated test suite `tests/test_phase14_6_provenance.py` verified:
1. `test_historical_trace_immutability`: PASS (SHA-256 identical to baseline)
2. `test_provenance_sidecar_integrity`: PASS (1,100 records mapped to `SIMULATED_END_TO_END`)
3. `test_no_silent_promotion_to_physical_inference`: PASS (Calibrated simulation blocked from being labeled physical inference)
4. `test_unknown_provenance_rule`: PASS (Unverifiable traces strictly classified as unknown)

---

## 5. Remediation Status
- **Discrepancy P2-01**: **RESOLVED**
- **Residual Limitation**: The 1,100 end-to-end evaluation records characterize a calibrated parametric simulation, not 1,100 physical forward passes. This is now fully disclosed in the sidecar manifest, metadata, and manuscript correction register.
