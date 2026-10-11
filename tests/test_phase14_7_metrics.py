"""Unit tests for Phase 14.7 metric decomposition and denominator fairness."""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.phase14_7.metric_validation import GroundedMetricValidator

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = REPO_ROOT / "experiments" / "phase14" / "traces" / "physical_inference_traces.jsonl"


def test_iou_threshold_exact_boundary():
    """IoU == tau passes, IoU < tau fails."""
    df = pd.DataFrame({
        "exact_match_accuracy": [1.0, 1.0],
        "grounding_iou": [0.500, 0.499],
        "unsupported_answer_rate": [0.0, 0.0]
    })
    res = GroundedMetricValidator.compute_decomposed_metrics(df, iou_threshold=0.50)
    # Item 0 passes (IoU 0.50), Item 1 fails (IoU 0.499) -> 1/2 = 50.0%
    assert res["correct_grounded_coverage_pct"] == 50.0


def test_separation_of_grounding_and_correctness():
    """Grounded Coverage and Correct Grounded Coverage must measure distinct dimensions."""
    df = pd.DataFrame({
        "exact_match_accuracy": [0.0, 1.0],  # Item 0 is incorrect, Item 1 is correct
        "grounding_iou": [0.85, 0.35],       # Item 0 has high IoU, Item 1 has low IoU
        "unsupported_answer_rate": [0.0, 0.0]
    })
    res = GroundedMetricValidator.compute_decomposed_metrics(df, iou_threshold=0.50)
    # Item 0 is grounded (IoU >= 0.50), so grounded_coverage = 1/2 = 50.0%
    assert res["grounded_coverage_pct"] == 50.0
    # Neither item is BOTH correct AND grounded: correct_grounded_coverage = 0/2 = 0.0%
    assert res["correct_grounded_coverage_pct"] == 0.0


def test_empty_dataframe_handling():
    """Empty dataframe returns 0.0 without throwing exceptions."""
    df = pd.DataFrame()
    res = GroundedMetricValidator.compute_decomposed_metrics(df)
    assert res["n_eligible"] == 0
    assert res["correct_grounded_coverage_pct"] == 0.0


def test_trace_denominator_fairness_on_corpus():
    """All 4 conditions in historical traces must share the exact same 55 test queries."""
    traces = []
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    df_all = pd.DataFrame(traces)

    fairness = GroundedMetricValidator.audit_denominator_fairness(df_all)
    assert fairness["queries_strictly_identical"] is True
    assert fairness["ref_query_count"] == 55
    assert set(fairness["conditions_audited"]) == {"B14-A", "B14-B", "B14-C", "B14-D"}
    for cond, q_count in fairness["total_queries_per_condition"].items():
        assert q_count == 55
