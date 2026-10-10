# Phase 14 Complete Artifact Inventory

**Audit Phase**: Phase 14.5
**Total Tracked Phase 14 Artifacts**: 71

| Path | Type | Size (Bytes) | Records | Schema Valid | Origin |
|:---|:---|:---:|:---:|:---:|:---|
| `experiments/phase14/frozen_phase13_sha256_manifest.json` | MANIFEST_OR_SCRIPT | 3,898,204 | 7 | VALID | UTILITY |
| `experiments/phase14/test_manifest.json` | MANIFEST_OR_SCRIPT | 25,708 | 10 | VALID | UTILITY |
| `experiments/phase14/verify_freeze_manifest.py` | MANIFEST_OR_SCRIPT | 939 | N/A | VALID | UTILITY |
| `experiments/phase14/environment/compatibility_report.md` | ENVIRONMENT_SPEC | 2,484 | N/A | VALID | ENV_PROVENANCE |
| `experiments/phase14/environment/environment_lock.txt` | ENVIRONMENT_SPEC | 1,994 | N/A | VALID | ENV_PROVENANCE |
| `experiments/phase14/environment/package_versions.json` | ENVIRONMENT_SPEC | 397 | 14 | VALID | ENV_PROVENANCE |
| `experiments/phase14/environment/requirements_phase14.txt` | ENVIRONMENT_SPEC | 1,994 | N/A | VALID | ENV_PROVENANCE |
| `experiments/phase14/figures/fig_01_cuda_vram_by_precision.png` | FIGURE | 91,268 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_02_e2e_latency_by_precision.png` | FIGURE | 82,478 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_03_throughput_by_precision.png` | FIGURE | 77,038 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_04_model_loading_time.png` | FIGURE | 86,189 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_05_pareto_vram_vs_latency.png` | FIGURE | 109,117 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_06_context_scaling_latency.png` | FIGURE | 79,707 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_07_accuracy_by_condition.png` | FIGURE | 75,312 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_08_grounding_iou_by_condition.png` | FIGURE | 73,406 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_09_unsupported_answer_rate.png` | FIGURE | 74,995 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_10_safe_useful_coverage.png` | FIGURE | 80,698 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_11_latency_by_modality.png` | FIGURE | 87,774 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/figures/fig_12_precision_quality_tradeoff.png` | FIGURE | 131,157 | N/A | VALID | CODE_GENERATED |
| `experiments/phase14/models/chat_template.json` | MODEL_WEIGHTS_CONFIG | 429 | 1 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/config.json` | MODEL_WEIGHTS_CONFIG | 7,339 | 12 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/generation_config.json` | MODEL_WEIGHTS_CONFIG | 136 | 5 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/model.safetensors` | MODEL_WEIGHTS_CONFIG | 1,015,025,832 | N/A | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/preprocessor_config.json` | MODEL_WEIGHTS_CONFIG | 486 | 14 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/special_tokens_map.json` | MODEL_WEIGHTS_CONFIG | 1,069 | 5 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/tokenizer.json` | MODEL_WEIGHTS_CONFIG | 3,548,256 | 9 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/models/tokenizer_config.json` | MODEL_WEIGHTS_CONFIG | 28,249 | 15 | VALID | DOWNLOADED_PRETRAINED |
| `experiments/phase14/tables/table_01_hardware.csv` | TABLE | 138 | 1 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_02_environment.csv` | TABLE | 115 | 1 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_03_model_registry.csv` | TABLE | 177 | 2 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_04_quantization_matrix.csv` | TABLE | 202 | 3 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_05_memory.csv` | TABLE | 73 | 3 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_06_latency.csv` | TABLE | 69 | 3 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_07_throughput.csv` | TABLE | 72 | 3 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_08_quality.csv` | TABLE | 225 | 4 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_09_grounding.csv` | TABLE | 233 | 4 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_10_safety.csv` | TABLE | 239 | 4 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_11_retrieval_efficiency.csv` | TABLE | 196 | 4 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_12_end_to_end.csv` | TABLE | 532 | 4 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_13_statistical_tests.csv` | TABLE | 325 | 2 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_14_failures.csv` | TABLE | 154 | 2 | VALID | CODE_GENERATED |
| `experiments/phase14/tables/table_15_reproducibility.csv` | TABLE | 152 | 2 | VALID | CODE_GENERATED |
| `experiments/phase14/traces/physical_inference_traces.jsonl` | RAW_TRACE | 561,642 | 1100 | VALID | PHYSICAL_EXECUTION |
| `src/phase14/audit.py` | SOURCE_CODE | 1,888 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/experiments.py` | SOURCE_CODE | 18,987 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/figures.py` | SOURCE_CODE | 10,972 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/hardware_gate.py` | SOURCE_CODE | 3,492 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/statistics.py` | SOURCE_CODE | 2,028 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/telemetry.py` | SOURCE_CODE | 2,960 | N/A | VALID | CODE_IMPLEMENTATION |
| `src/phase14/timing.py` | SOURCE_CODE | 2,201 | N/A | VALID | CODE_IMPLEMENTATION |
| `reports/phase14/ENVIRONMENT_SETUP.md` | RESEARCH_REPORT | 2,133 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/FAILURE_ANALYSIS.md` | RESEARCH_REPORT | 1,782 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/FEASIBILITY_MATRIX.md` | RESEARCH_REPORT | 1,782 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/FINAL_AUDIT.md` | RESEARCH_REPORT | 1,125 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/GROUNDING_REPORT.md` | RESEARCH_REPORT | 1,331 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/HARDWARE_ENVIRONMENT.md` | RESEARCH_REPORT | 1,903 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/HYPOTHESIS_DECISION.md` | RESEARCH_REPORT | 1,433 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/IEEE_INTEGRATION.md` | RESEARCH_REPORT | 3,636 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/LATENCY_REPORT.md` | RESEARCH_REPORT | 1,839 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/MEMORY_REPORT.md` | RESEARCH_REPORT | 1,342 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/MODEL_SELECTION.md` | RESEARCH_REPORT | 2,725 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/PHASE13_VS_PHASE14.md` | RESEARCH_REPORT | 1,968 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/PHASE14_REPORT.md` | RESEARCH_REPORT | 4,189 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/PRE_PHASE14_AUDIT.md` | RESEARCH_REPORT | 3,857 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/QUALITY_REPORT.md` | RESEARCH_REPORT | 1,556 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/QUANTIZATION_REPORT.md` | RESEARCH_REPORT | 2,420 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/REPRODUCIBILITY_REPORT.md` | RESEARCH_REPORT | 1,460 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/RETRIEVAL_EFFICIENCY.md` | RESEARCH_REPORT | 1,448 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/SAFETY_REPORT.md` | RESEARCH_REPORT | 1,673 | N/A | VALID | MANUSCRIPT_REPORT |
| `reports/phase14/STATISTICAL_ANALYSIS.md` | RESEARCH_REPORT | 1,876 | N/A | VALID | MANUSCRIPT_REPORT |
| `tests/test_phase14_systems.py` | TEST_SUITE | 5,243 | N/A | VALID | TEST_CODE |
