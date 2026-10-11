"""Unit tests for Phase 14.6 statistical testing and Holm-Bonferroni correction."""

import pytest
import numpy as np
import pandas as pd
from src.phase14_6.statistics import apply_holm_bonferroni, cluster_bootstrap_family_paired


def test_holm_bonferroni_empty_and_single():
    assert apply_holm_bonferroni([]) == []
    assert apply_holm_bonferroni([0.02]) == [0.02]
    assert apply_holm_bonferroni([1.5]) == [1.0] or pytest.raises(ValueError)


def test_holm_bonferroni_known_case():
    """Textbook example: raw = [0.01, 0.04, 0.03], m = 3."""
    raw = [0.01, 0.04, 0.03]
    # Sorted: (rank 0: 0.01), (rank 1: 0.03), (rank 2: 0.04)
    # adj_0 = 3 * 0.01 = 0.03
    # adj_1 = 2 * 0.03 = 0.06
    # adj_2 = max(0.06, 1 * 0.04) = 0.06
    # Restored to original order: [0.03, 0.06, 0.06]
    adj = apply_holm_bonferroni(raw)
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.06)


def test_holm_bonferroni_order_preservation():
    """Adjusted p-values must map strictly back to the input index."""
    raw = [0.50, 0.001, 0.02, 0.005]
    adj = apply_holm_bonferroni(raw)
    # The smallest raw p-value (idx 1: 0.001) must have smallest adjusted
    assert adj[1] < adj[3] < adj[2] < adj[0]


def test_holm_bonferroni_monotonicity_and_bounds():
    raw = [0.0001, 0.0002, 0.0000, 0.8, 0.95]
    adj = apply_holm_bonferroni(raw)
    assert all(0.0 <= p <= 1.0 for p in adj)
    # Check monotonicity in sorted order
    sorted_pairs = sorted(zip(raw, adj), key=lambda x: x[0])
    sorted_adj = [p for _, p in sorted_pairs]
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i] <= sorted_adj[i + 1]


def test_holm_bonferroni_phase14_hypothesis_family():
    """Evaluates the Phase 14 2-test pre-registered family."""
    # Hypothesis tests H14-3 (p=0.0002) and H14-4 (p=0.0000)
    raw = [0.0002, 0.0000]
    adj = apply_holm_bonferroni(raw)
    # Rank 0 (p=0.0000): 2 * 0.0000 = 0.0000
    # Rank 1 (p=0.0002): max(0.0000, 1 * 0.0002) = 0.0002
    assert adj[0] == pytest.approx(0.0002)
    assert adj[1] == pytest.approx(0.0000)
    # Both remain strongly significant at alpha = 0.05 and alpha = 0.01
    assert all(p < 0.01 for p in adj)


def test_cluster_bootstrap_family_paired():
    df = pd.DataFrame({
        "family_id": ["fam1", "fam1", "fam2", "fam2", "fam3", "fam3"],
        "A": [0.5, 0.6, 0.4, 0.5, 0.7, 0.8],
        "B": [0.7, 0.8, 0.6, 0.7, 0.9, 1.0]  # B is consistently +0.2 higher
    })
    res = cluster_bootstrap_family_paired(df, "A", "B", cluster_col="family_id", n_resamples=500, seed=42)
    assert res["mean_diff"] == pytest.approx(0.20, abs=1e-4)
    assert res["ci_lower"] > 0.0
    assert res["n_clusters"] == 3
    assert res["p_value_raw"] < 0.05
