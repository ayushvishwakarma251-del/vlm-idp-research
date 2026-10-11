"""Unit tests for Phase 14.7 IoU threshold sensitivity and rank-order invariance."""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.phase14_7.threshold_sensitivity import ThresholdSensitivityEngine, GRID_THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = REPO_ROOT / "experiments" / "phase14" / "traces" / "physical_inference_traces.jsonl"


@pytest.fixture
def traces_df():
    traces = []
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return pd.DataFrame(traces)


def test_monotonic_non_increasing_coverage(traces_df):
    """For any condition, as threshold tau increases, correct grounded coverage must not increase."""
    grid_df = ThresholdSensitivityEngine.evaluate_grid(traces_df, GRID_THRESHOLDS)
    for cond in grid_df["condition"].unique():
        sub = grid_df[grid_df["condition"] == cond].sort_values("tau")
        values = sub["correct_grounded_coverage_pct"].tolist()
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], f"Monotonicity violation in {cond} at tau index {i}: {values}"


def test_strict_rank_order_invariance(traces_df):
    """Proposed condition (B14-D) must rank highest across all preregistered thresholds."""
    paired_df = ThresholdSensitivityEngine.evaluate_paired_effects(traces_df, GRID_THRESHOLDS)
    for _, row in paired_df.iterrows():
        tau = row["tau"]
        assert row["ranking"] == "D > C > B > A", f"Rank reversal observed at tau={tau}"
        assert row["delta_D_vs_A_pp"] > 0.0, f"Proposed advantage not positive at tau={tau}"


def test_delta_d_vs_a_range(traces_df):
    """Delta(D - A) spans from +16.36 pp (tau=0.30) to +89.09 pp (tau=0.60, 0.70)."""
    paired_df = ThresholdSensitivityEngine.evaluate_paired_effects(traces_df, GRID_THRESHOLDS)
    piv = paired_df.set_index("tau")
    assert piv.loc[0.30, "delta_D_vs_A_pp"] == pytest.approx(16.36, abs=0.1)
    assert piv.loc[0.50, "delta_D_vs_A_pp"] == pytest.approx(70.91, abs=0.1)
    assert piv.loc[0.70, "delta_D_vs_A_pp"] == pytest.approx(89.09, abs=0.1)
