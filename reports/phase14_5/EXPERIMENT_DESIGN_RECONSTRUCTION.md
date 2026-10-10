# Experiment Design Reconstruction Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Repository**: `vlm-idp-research`  
**Manifest Path**: `experiments/phase14_5/manifests/experiment_design_manifest.json`  

---

## 1. Executive Summary

The experimental design of Phase 14 was independently reconstructed through forensic deconstruction of the source implementation (`src/phase14/experiments.py`), corpus registries (`data/phase12_authentic/corpus_manifest.json`), split partitions (`data/phase12_authentic/splits.json`), and raw execution traces (`experiments/phase14/traces/physical_inference_traces.jsonl`).

The design evaluates physical VLM execution under real-world visual degradation across two distinct layers:
1. **Physical Quantization Microbenchmark Layer**: Evaluates hardware limits (7B FP16 OOM negative control) and physical inference throughput of a fallback VLM (`SmolVLM-500M-Instruct`) across FP16, INT8, and INT4 (NF4) on an NVIDIA RTX 3050 6GB Laptop GPU.
2. **End-to-End Evaluation Matrix Layer**: Evaluates four pipeline conditions across 5 evaluation seeds on 55 authentic test document queries grouped into 11 independent document families.

---

## 2. Experimental Population and Partitioning

- **Target Corpus**: Phase 12 Authentic Real-World Document Corpus.
- **Total Corpus Size**: 260 documents, 260 queries, 52 document families (5 documents per family).
- **Split Distribution**:
  - **Train**: 31 families, 155 documents / queries (59.6%)
  - **Validation**: 10 families, 50 documents / queries (19.2%)
  - **Test**: 11 families, 55 documents / queries (21.2%)
- **Test Unit Independence**: Exactly 11 independent document families in the test partition.
- **Evaluation Queries**: Exactly 5 queries per family (1 query per document $\times$ 5 documents).
- **Cluster Isolation**: Cluster-level partitioning ensures that all documents belonging to a family remain within a single split, preventing cross-document contamination.

---

## 3. Evaluation Matrix (Conditions $\times$ Seeds)

The end-to-end evaluation cross-tabulates 4 pipeline conditions and 5 random seeds:

| Condition ID | Pipeline Name | Pages Fed | Description | Expected Latency / Quality Behavior |
|:---|:---|:---:|:---|:---|
| **B14-A** | `full_document_vlm` | 5 | Unpruned full-document VLM baseline | High context latency (5 pages), baseline accuracy, lower IoU |
| **B14-B** | `retrieval_pruned_vlm` | 2 | Multimodal retrieval page-pruned VLM | Reduced context latency (60% page reduction), elevated accuracy |
| **B14-C** | `retrieval_grounded_vlm` | 2 | Multimodal retrieval + spatial grounding | Moderate latency, high grounding IoU, reduced unsupported answer rate |
| **B14-D** | `full_proposed_pipeline` | 2 | Integrated pipeline (retrieval + grounding + uncertainty + abstention) | Lowest unsupported rate, highest accuracy and safe coverage |

- **Evaluation Seeds**: $S \in \{42, 123, 456, 789, 101112\}$ ($N = 5$).
- **Total Trace Observations**: $4 \text{ conditions} \times 5 \text{ seeds} \times 55 \text{ queries} = 1,100 \text{ observations}$.

---

## 4. Hardware Environment and Precision Modes

- **Physical GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, Ampere SM 8.6).
- **Software Stack**: CUDA Driver 13.0, PyTorch 2.14.1 with CUDA 12.6 runtime.
- **Precision Modes Evaluated**:
  - **Q0 (FP16)**: Full 16-bit floating point.
  - **Q1 (INT8)**: 8-bit quantization via `bitsandbytes.load_in_8bit`.
  - **Q2 (INT4 NF4)**: 4-bit NormalFloat quantization via `bitsandbytes.load_in_4bit`.
- **Target vs. Fallback Model Hierarchy**:
  - Target model `Qwen2.5-VL-7B-Instruct` (~14.5 GB FP16) cannot fit into 6 GB VRAM and was evaluated via a negative allocation control.
  - Fallback model `SmolVLM-500M-Instruct` (~1.0 GB FP16) was physically loaded and benchmarked on GPU across all precision modes.
