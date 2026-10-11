"""Tests for Phase 14.7 Small-Cluster (K=11) Family-Level Statistics Engine."""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from src.phase14_7.small_sample_statistics import SmallClusterStatisticalEngine

TRACES_PATH = Path("experiments/phase14/traces/physical_inference_traces.jsonl")


@pytest.fixture(scope="module")
def trace_records():
    assert TRACES_PATH.exists(), f"Missing traces file at {TRACES_PATH}"
    lines = [json.loads(line) for line in TRACES_PATH.open("r", encoding="utf-8")]
    return pd.DataFrame(lines)


def test_family_level_aggregation_geometry(trace_records):
    """Verify that aggregation yields exactly 11 document families per condition."""
    df_fam = SmallClusterStatisticalEngine.aggregate_family_level_outcomes(trace_records, tau=0.50)
    
    assert len(df_fam) == 44, f"Expected 44 rows (11 families * 4 conditions), got {len(df_fam)}"
    families = df_fam["family_id"].unique()
    assert len(families) == 11, f"Expected 11 unique families, got {len(families)}"
    
    # Each family has exactly 5 queries and 25 total trace evaluations
    for fam in families:
        fam_sub = df_fam[df_fam["family_id"] == fam]
        assert set(fam_sub["n_queries"]) == {5}
        assert set(fam_sub["n_evaluations"]) == {25}


def test_exact_paired_permutation_test():
    """Verify exact permutation test over 2^11 = 2048 sign combinations."""
    # When all 11 diffs are strictly positive
    diffs_all_pos = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.14, 0.19, 0.21, 0.16])
    res = SmallClusterStatisticalEngine.exact_paired_permutation_test(diffs_all_pos)
    assert res["total_permutations"] == 2048
    # Only 2 permutations (all +1 and all -1) can achieve >= observed mean
    assert res["extreme_permutations"] == 2
    assert pytest.approx(res["exact_p_value"], 1e-5) == 2 / 2048  # 0.0009765625


def test_h14_3_small_sample_cluster_statistics(trace_records):
    """Verify H14-3 Grounded SUC (tau=0.50) small-cluster statistical evaluation."""
    df_fam = SmallClusterStatisticalEngine.aggregate_family_level_outcomes(trace_records, tau=0.50)
    res = SmallClusterStatisticalEngine.evaluate_comprehensive_family_tests(
        df_family_outcomes=df_fam,
        metric_col="grounded_suc_tau50",
        cond_baseline="B14-A",
        cond_treatment="B14-D",
        hypothesis_id="H14-3_GROUNDED",
        comparison_name="B14-D vs B14-A Grounded SUC (tau=0.50)"
    )
    
    assert res["k_clusters"] == 11
    assert res["positive_families"] == 11
    assert res["negative_families"] == 0
    assert res["exact_permutation_p_val"] == pytest.approx(2 / 2048, abs=1e-5)
    assert res["wilcoxon_p_val"] == pytest.approx(2 / 2048, abs=1e-5)
    assert res["sign_test_p_val"] == pytest.approx(2 / 2048, abs=1e-5)
    assert res["lofo_sign_preserved"] is True
    assert res["lofo_min_delta_pct"] > 60.0  # robustly large positive delta regardless of excluded family


def test_h14_4_small_sample_cluster_statistics(trace_records):
    """Verify H14-4 UAR small-cluster statistical evaluation."""
    df_fam = SmallClusterStatisticalEngine.aggregate_family_level_outcomes(trace_records, tau=0.50)
    res = SmallClusterStatisticalEngine.evaluate_comprehensive_family_tests(
        df_family_outcomes=df_fam,
        metric_col="uar_unconditional",
        cond_baseline="B14-A",
        cond_treatment="B14-D",
        hypothesis_id="H14-4_UAR",
        comparison_name="B14-D vs B14-A Unconditional UAR"
    )
    
    assert res["k_clusters"] == 11
    assert res["positive_families"] == 0
    assert res["negative_families"] == 11
    assert res["exact_permutation_p_val"] == pytest.approx(2 / 2048, abs=1e-5)
    assert res["lofo_sign_preserved"] is True
    assert res["lofo_max_delta_pct"] < -15.0  # consistently negative UAR delta across all LOFO splits
