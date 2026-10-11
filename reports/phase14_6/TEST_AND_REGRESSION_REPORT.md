# Test and Regression Report

**Phase**: Phase 14.6 — Testing and Adversarial Validation  
**Date**: 2026-10-11  
**Test Framework**: `pytest 9.1.1` on Python 3.14.6  
**Total Tests Executed**: **36**  
**Passed**: **36**  
**Failed**: **0**  
**Execution Duration**: 63.03s  
**Status**: **ALL TESTS PASSED**  

---

## 1. Test Suite Summary Breakdown

| Test File | Target Scope | Tests Count | Status |
|:---|:---|:---:|:---:|
| **`tests/test_phase14_6_provenance.py`** | Provenance sidecar manifest, execution class enforcement, immutability | 4 | **PASSED** (100%) |
| **`tests/test_phase14_6_metrics.py`** | SUC mathematical coupling check, remediated grounded SUC, UAR split | 5 | **PASSED** (100%) |
| **`tests/test_phase14_6_statistics.py`** | Holm-Bonferroni algorithm, order preservation, monotonicity, cluster bootstrap | 6 | **PASSED** (100%) |
| **`tests/test_phase14_6_reporting.py`** | Remediated table generation, figure provenance banner, historical immutability | 5 | **PASSED** (100%) |
| **`tests/test_phase14_systems.py`** | Phase 14 baseline system checks (telemetry, profiler, hardware gate, 15 tables) | 10 | **PASSED** (100%) |
| **`tests/test_phase14_5_audit.py`** | Phase 14.5 forensic concordance and mathematical audit | 6 | **PASSED** (100%) |
| **Total Combined Suite** | Complete regression and remediation suite | **36** | **36 / 36 PASSED** |

---

## 2. Granular Test Item Results

### Phase 14.6 Remediation Tests (20 Tests)
1. `test_phase14_6_provenance.py::test_historical_trace_immutability` $\rightarrow$ **PASSED**
2. `test_phase14_6_provenance.py::test_provenance_sidecar_integrity` $\rightarrow$ **PASSED**
3. `test_phase14_6_provenance.py::test_no_silent_promotion_to_physical_inference` $\rightarrow$ **PASSED**
4. `test_phase14_6_provenance.py::test_unknown_provenance_rule` $\rightarrow$ **PASSED**
5. `test_phase14_6_metrics.py::test_historical_suc_exact_coupling` $\rightarrow$ **PASSED**
6. `test_phase14_6_metrics.py::test_remediated_suc_grounding_filter` $\rightarrow$ **PASSED**
7. `test_phase14_6_metrics.py::test_remediated_suc_abstention_filter` $\rightarrow$ **PASSED**
8. `test_phase14_6_metrics.py::test_remediated_metric_engine_empty_input` $\rightarrow$ **PASSED**
9. `test_phase14_6_metrics.py::test_unconditional_vs_conditional_uar` $\rightarrow$ **PASSED**
10. `test_phase14_6_statistics.py::test_holm_bonferroni_empty_and_single` $\rightarrow$ **PASSED**
11. `test_phase14_6_statistics.py::test_holm_bonferroni_known_case` $\rightarrow$ **PASSED**
12. `test_phase14_6_statistics.py::test_holm_bonferroni_order_preservation` $\rightarrow$ **PASSED**
13. `test_phase14_6_statistics.py::test_holm_bonferroni_monotonicity_and_bounds` $\rightarrow$ **PASSED**
14. `test_phase14_6_statistics.py::test_holm_bonferroni_phase14_hypothesis_family` $\rightarrow$ **PASSED**
15. `test_phase14_6_statistics.py::test_cluster_bootstrap_family_paired` $\rightarrow$ **PASSED**
16. `test_phase14_6_reporting.py::test_remediated_tables_exist` $\rightarrow$ **PASSED**
17. `test_phase14_6_reporting.py::test_table_12_remediated_structure_and_metrics` $\rightarrow$ **PASSED**
18. `test_phase14_6_reporting.py::test_table_13_remediated_holm_adjustment` $\rightarrow$ **PASSED**
19. `test_phase14_6_reporting.py::test_figure_svg_provenance_banner` $\rightarrow$ **PASSED**
20. `test_phase14_6_reporting.py::test_historical_tables_remain_untouched` $\rightarrow$ **PASSED**

### Historical Regression Baseline Tests (16 Tests)
21. `test_phase14_systems.py::test_phase14_hardware_gate` $\rightarrow$ **PASSED**
22. `test_phase14_systems.py::test_phase14_timing_profiler` $\rightarrow$ **PASSED**
23. `test_phase14_systems.py::test_phase14_telemetry_monitor` $\rightarrow$ **PASSED**
24. `test_phase14_systems.py::test_phase14_ast_and_sentinel_audits` $\rightarrow$ **PASSED**
25. `test_phase14_systems.py::test_phase14_traces_integrity` $\rightarrow$ **PASSED**
26. `test_phase14_systems.py::test_phase14_tables_all_15_exist` $\rightarrow$ **PASSED**
27. `test_phase14_systems.py::test_phase14_figures_all_12_exist` $\rightarrow$ **PASSED**
28. `test_phase14_systems.py::test_phase14_quantization_matrix_empirical_values` $\rightarrow$ **PASSED**
29. `test_phase14_systems.py::test_phase14_retrieval_and_safety_gains` $\rightarrow$ **PASSED**
30. `test_phase14_systems.py::test_phase14_statistical_hypotheses` $\rightarrow$ **PASSED**
31. `test_phase14_5_audit.py::test_trace_schema_and_counts` $\rightarrow$ **PASSED**
32. `test_phase14_5_audit.py::test_end_to_end_table12_concordance` $\rightarrow$ **PASSED**
33. `test_phase14_5_audit.py::test_statistical_tests_table13_concordance` $\rightarrow$ **PASSED**
34. `test_phase14_5_audit.py::test_suc_mathematical_collapse` $\rightarrow$ **PASSED**
35. `test_phase14_5_audit.py::test_split_family_leakage` $\rightarrow$ **PASSED**
36. `test_phase14_5_audit.py::test_quantization_table04_validity` $\rightarrow$ **PASSED**

---

## 3. Regression Verdict
- Historical integrity maintained with **zero regressions**.
- All Phase 14.6 mathematical, provenance, and statistical invariants verified.
- **Verdict**: **PASS**
