# Dataset Provenance and Family Leakage Audit Report

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Source Dataset**: Phase 12 Authentic Document Corpus (`data/phase12_authentic/`)  
**Split Definition**: `data/phase12_authentic/splits.json`  
**Evaluation Traces**: `experiments/phase14/traces/physical_inference_traces.jsonl`  

---

## 1. Executive Summary

A forensic leakage and provenance audit was conducted on the evaluation dataset supporting Phase 14. The audit verified whether the 55 test queries and documents evaluated in Phase 14 satisfy strict family-level isolation from training and validation partitions.

| Verification Item | Tested Property | Result | Scientific Status |
|:---|:---|:---:|:---|
| **Family-Level Train / Val Overlap** | $\text{Families}_{\text{train}} \cap \text{Families}_{\text{val}} = \emptyset$ | 0 overlapping families | `VERIFIED` |
| **Family-Level Train / Test Overlap** | $\text{Families}_{\text{train}} \cap \text{Families}_{\text{test}} = \emptyset$ | 0 overlapping families | `VERIFIED` |
| **Family-Level Val / Test Overlap** | $\text{Families}_{\text{val}} \cap \text{Families}_{\text{test}} = \emptyset$ | 0 overlapping families | `VERIFIED` |
| **Document-Level Split Contamination** | $\text{Docs}_i \cap \text{Docs}_j = \emptyset, \forall i \neq j$ | 0 overlapping documents | `VERIFIED` |
| **Trace Membership in Test Split** | $\text{Docs}_{\text{trace}} \subseteq \text{Docs}_{\text{test}}$ | 55 / 55 (100.0%) | `VERIFIED` |
| **Out-of-Split Trace Documents** | $\text{Docs}_{\text{trace}} \setminus \text{Docs}_{\text{test}}$ | 0 documents | `VERIFIED` |

---

## 2. Partition Architecture

The Phase 12 authentic corpus comprises 260 real-world multi-page document images organized into 52 document families (5 documents per family, representing homogeneous layout formats across distinct business instances):

```
Total Authentic Corpus: 52 Families / 260 Documents / 260 Queries
  ├── Train Split:      31 Families / 155 Documents / 155 Queries (59.6%)
  ├── Validation Split: 10 Families /  50 Documents /  50 Queries (19.2%)
  └── Test Split:       11 Families /  55 Documents /  55 Queries (21.2%)
```

### Partition Properties:
1. **Cluster Unit**: Partitions are formed strictly at the `family_id` cluster level. No family is split across subsets.
2. **Evaluation Scope**: Phase 14 evaluation strictly isolates the 11 test families ($N = 11$ clusters) for all reported metrics and bootstrapping procedures.

---

## 3. Data Leakage Investigation

1. **Prompt/Few-Shot Contamination Check**: Test documents were examined against prompt construction files in `src/phase14/experiments.py`. The prompts used (`<image>What type of document is this? Answer concisely:`) are generic zero-shot prompts containing zero in-context test examples.
2. **Parameter Tuning Contamination Check**: In Phase 14, no model fine-tuning was performed. The evaluation used frozen pre-trained checkpoints with post-training quantization (`bitsandbytes`). No test documents were exposed during calibration.
3. **Cluster Integrity**: All 55 test documents in `physical_inference_traces.jsonl` map to the 11 test families: `fam_auth_42` through `fam_auth_52`. None of these families appear in the training or validation splits.

---

## 4. Scientific Verdict

- **Dataset Provenance**: `VERIFIED`
- **Family Independence**: `VERIFIED`
- **Data Leakage Risk**: **ZERO (PASS)**
