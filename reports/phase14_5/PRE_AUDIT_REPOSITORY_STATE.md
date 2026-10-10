# Pre-Audit Repository State Assessment

**Audit Phase**: Phase 14.5 — Forensic Reproducibility, Statistical Validation & Scientific Integrity Audit  
**Date**: 2026-10-10  
**Repository**: `vlm-idp-research`  
**Research Title**: Vision-Language Models for Intelligent Document Processing under Real-World Visual Degradation  

---

## 1. Executive Summary of Repository State

Before executing any Phase 14.5 audit scripts, recalculations, or file modifications, the Git repository topology, history, remote configuration, working tree, and commit ancestry were forensically cataloged.

- **Current Git Branch**: `master`
- **Current HEAD Commit**: `baf597c1512dcce4b2b4c91a0d0a874b1459e332` (`baf597c1`)
- **Expected Phase 14 Frozen Baseline Commit**: `583e8a8fa93967a15ddfbdb97d247cfcdbaad922` (`583e8a8f`)
- **Phase 14 Parent Commit**: `62f3c632bf1e4fbdbdd13e9375c97412d4c652ef` (`62f3c632`)
- **Ancestry Verification**: `583e8a8f` is a **direct confirmed ancestor** of HEAD (`baf597c1`). `git merge-base --is-ancestor 583e8a8f HEAD` exited with code `0`.
- **Working Tree Status**: Pristine / clean (`git status --porcelain` returned empty prior to Phase 14.5 artifact creation).
- **Remote Configuration**:
  - `origin https://github.com/officialayush5839-arch/vlm-idp-research.git (fetch)`
  - `origin https://github.com/officialayush5839-arch/vlm-idp-research.git (push)`
- **Remote Policy Compliance**: No remote push commands will be executed. All commits remain strictly local.
- **Pre-existing Phase 14.5 Work**: None detected. All Phase 14.5 structures are instantiated de novo.

---

## 2. Forensic Commit History Analysis

Between the frozen Phase 14 benchmark commit (`583e8a8f`) and the current HEAD (`baf597c1`), 11 commits were recorded:

| Commit Hash | Commit Type | Subject |
|:------------|:------------|:--------|
| `baf597c1` | `feat(fixtures)` | add sample_test_invoice.pdf for testing Document Studio |
| `8a2f8014` | `docs` | add CLASSIFICATION.md and integrate into README |
| `b7f4c360` | `feat(orchestrator)` | support tabular column matching for rate, price, tax, and multi-line dates |
| `7408b819` | `fix(orchestrator)` | integrate PyMuPDF layout parsing and spatial proximity value extraction |
| `d66a8e37` | `fix(ui)` | improve upload error handling and diagnostics |
| `4449e13d` | `feat(phase15)` | add interactive document extraction studio |
| `67376e46` | `feat(launcher)` | add run.bat launcher and full-stack API server on localhost:8896 |
| `371c9332` | `fix(assets)` | resolve SVG CSS transform collision and layout overlap in architecture pipeline |
| `df26e228` | `feat(assets)` | track 3d animated svgs and scrolling ticker in git |
| `075d98ef` | `feat(docs)` | embed native 3d animated isometric visuals and scrolling ticker into readme |
| `e6df91d4` | `docs` | add polished research readme, mit license, and interactive 3d web showcase |

### File Diff Analysis (`583e8a8f` to `HEAD`)

A comprehensive path differential (`git diff --name-status 583e8a8f HEAD`) reveals:
1. **Historical Phase 0–14 Artifacts**: Exactly **zero** files in `experiments/phase14/`, `reports/phase14/`, `src/phase14/`, `experiments/phase13/`, `experiments/phase12/`, etc., were modified between commit `583e8a8f` and `HEAD`.
2. **README Modification**: `README.md` was modified to showcase research results and embed interactive visuals.
3. **Phase 15 Document Studio Additions**: New files were added in `src/runtime/`, `src/server.py`, `docs/`, `tests/test_phase15_*.py`, `CLASSIFICATION.md`, `LICENSE`, and `run.bat`.

Therefore, the experimental evidence produced in Phase 14 remains intact in the commit tree.

---

## 3. Execution Environment Diagnostics

- **Operating System**: Windows 11 AMD64 (10.0.26200)
- **Primary Audit Interpreter**: `C:\Users\ARYAN - AYUSH\OneDrive\Desktop\Nlp\vlm-idp-research\.venv_phase14\Scripts\python.exe`
- **Python Version**: 3.14.6 (tags/v3.14.6:c63aec6)
- **PyTorch Version**: `2.14.1+cu126` (CUDA 12.6 enabled)
- **Physical GPU Hardware**: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (SM 8.6, 6,143.5 MB VRAM)
- **CUDA Device Available**: `True` (Device ID: `0`)

---

## 4. Machine-Readable Provenance File

The exact machine-readable metadata captured at the commencement of Phase 14.5 is preserved in:
`experiments/phase14_5/provenance/git_baseline.json`
