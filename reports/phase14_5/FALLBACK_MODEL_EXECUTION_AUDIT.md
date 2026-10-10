# Fallback Model Physical Execution Audit

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Model**: `HuggingFaceTB/SmolVLM-500M-Instruct`  
**Model Location**: `experiments/phase14/models/`  
**Associated Code**: `src/phase14/experiments.py` (lines 108–172)  

---

## 1. Executive Summary

When target 7B models proved inexecutable under 6GB FP16 constraints, Phase 14 executed an automated fallback to the compact multimodal architecture `SmolVLM-500M-Instruct`. This forensic audit verified the physical provenance of this fallback model execution.

| Artifact / Property | Verified Location | Size / Detail | Integrity Status |
|:---|:---|:---:|:---:|
| Model Weights | `experiments/phase14/models/model.safetensors` | 1,015,025,832 bytes (1.01 GB) | `VERIFIED` |
| Tokenizer Definition | `experiments/phase14/models/tokenizer.json` | 3,548,256 bytes | `VERIFIED` |
| Model Configuration | `experiments/phase14/models/config.json` | 7,339 bytes | `VERIFIED` |
| Preprocessor Config | `experiments/phase14/models/preprocessor_config.json` | 486 bytes | `VERIFIED` |
| Physical Loading | CUDA device 0 (`torch.cuda.is_available() == True`) | RTX 3050 6GB Laptop GPU | `VERIFIED` |
| Forward Pass Execution | `AutoModelForVision2Seq.generate()` | 5 iterations per precision | `VERIFIED` |

---

## 2. Model Architecture and Physical Artifacts

`SmolVLM-500M-Instruct` uses the `Idefics3ForVision2Seq` architecture:
- **Vision Backbone**: SigLIP-based vision encoder configured for document processing with `image_seq_len = 64`.
- **Text Backbone**: Compact causal language model (~500M parameters).
- **Physical Weight File**: Preserved on disk at `experiments/phase14/models/model.safetensors` with SHA-256:
  `6d333062beffb65406d2036729bb9a5fb21c7bc1f251ffbf058e5e8e815e1974`
- **Execution Proof**: During the quantization benchmark, real image inputs from `data/raw/` were tokenized, preprocessed, transferred to CUDA (`inputs_cuda`), and passed through the model. Real generated text was captured and decoded (`processor.batch_decode`).

---

## 3. Scientific Verdict

- **Fallback Model Physical Execution**: **`VERIFIED`**.
- The fallback model is genuinely present in the repository, was physically instantiated on CUDA, and generated real tokens and memory telemetry.
