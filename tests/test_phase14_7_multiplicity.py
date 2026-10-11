"""Unit tests for Phase 14.7 Multiple Testing and Partitioning Engine."""

import pytest
from src.phase14_7.multiplicity import MultiplicityEngine


def test_holm_bonferroni_correction_logic():
    raw_tests = [
        {"name": "test_1", "raw_p_value": 0.01},
        {"name": "test_2", "raw_p_value": 0.04},
        {"name": "test_3", "raw_p_value": 0.001},
    ]
    # m = 3
    # Sorted raw p: 0.001 (rank 1), 0.01 (rank 2), 0.04 (rank 3)
    # Adjusted p:
    # rank 1: 0.001 * 3 = 0.003
    # rank 2: 0.01 * 2 = 0.02
    # rank 3: 0.04 * 1 = 0.04
    adj = MultiplicityEngine.apply_holm_bonferroni(raw_tests, alpha=0.05)
    adj_map = {t["name"]: t for t in adj}
    
    assert adj_map["test_3"]["adjusted_p_value"] == 0.003
    assert adj_map["test_3"]["holm_significant"] is True
    
    assert adj_map["test_1"]["adjusted_p_value"] == 0.02
    assert adj_map["test_1"]["holm_significant"] is True
    
    assert adj_map["test_2"]["adjusted_p_value"] == 0.04
    assert adj_map["test_2"]["holm_significant"] is True


def test_confirmatory_m2_partition():
    family_results = [
        {
            "hypothesis_id": "H14-3_GROUNDED",
            "comparison": "B14-D vs B14-A Grounded SUC",
            "metric": "grounded_suc_tau50",
            "exact_permutation_p_val": 0.000977
        },
        {
            "hypothesis_id": "H14-4_UAR",
            "comparison": "B14-D vs B14-A UAR",
            "metric": "uar_unconditional",
            "exact_permutation_p_val": 0.000977
        }
    ]
    threshold_results = [
        {"iou_threshold": 0.30, "p_value": 0.000977},
        {"iou_threshold": 0.40, "p_value": 0.000977},
        {"iou_threshold": 0.50, "p_value": 0.000977},
        {"iou_threshold": 0.60, "p_value": 0.000977},
        {"iou_threshold": 0.70, "p_value": 0.000977},
        {"iou_threshold": 0.80, "p_value": 0.000977},
    ]
    
    partitions = MultiplicityEngine.evaluate_confirmatory_and_exploratory_partitions(
        family_results, threshold_results
    )
    
    conf = partitions["confirmatory"]
    assert len(conf) == 2
    for c in conf:
        assert c["family_size_m"] == 2
        # Max adjusted p with raw p=0.000977 and m=2 is 0.000977 * 2 = 0.001954 < 0.05
        assert c["adjusted_p_value"] <= 0.002
        assert c["holm_significant"] is True

    expl = partitions["exploratory_thresholds"]
    assert len(expl) == 6
    for e in expl:
        assert e["family_size_m"] == 6
        assert e["holm_significant"] is True
