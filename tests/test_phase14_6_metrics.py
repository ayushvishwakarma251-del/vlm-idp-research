"""Unit tests for Phase 14.6 SUC metric audit and remediation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.phase14_6.metrics import HistoricalMetricAuditor, RemediatedMetricEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = REPO_ROOT / "experiments" / "phase14" / "traces" / "physical_inference_traces.jsonl"


def test_historical_suc_exact_coupling():
    """Historical traces must confirm exact mathematical coupling between SUC and Accuracy."""
    traces = []
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    df = pd.DataFrame(traces)

    audit = HistoricalMetricAuditor.audit_historical_suc_coupling(df)
    assert audit["total_records"] == 1100
    assert audit["exact_matches"] == 1100
    assert audit["match_rate"] == 1.0
    assert audit["mathematically_coupled"] is True
    assert audit["correlation_r"] == pytest.approx(1.0, abs=1e-6)


def test_remediated_suc_grounding_filter():
    """Remediated SUC must require grounding IoU >= threshold to be credited."""
    acc = np.array([1.0, 1.0, 1.0, 0.0])
    iou = np.array([0.80, 0.40, 0.90, 0.70])  # item 2 has low IoU (0.40 < 0.50)
    abstained = np.array([0.0, 0.0, 0.0, 0.0])

    suc = RemediatedMetricEngine.compute_grounded_safe_useful_coverage(acc, iou, abstained, iou_threshold=0.50)
    # Only item 0 and 2 satisfy acc=1.0 AND iou>=0.50 AND abstained=0 -> 2/4 = 0.50
    assert suc == pytest.approx(0.50)


def test_remediated_suc_abstention_filter():
    """Abstained queries must not count toward Safe Useful Coverage."""
    acc = np.array([1.0, 1.0])
    iou = np.array([0.80, 0.85])
    abstained = np.array([0.0, 1.0])  # item 1 was abstained

    suc = RemediatedMetricEngine.compute_grounded_safe_useful_coverage(acc, iou, abstained, iou_threshold=0.50)
    # Only item 0 satisfies all conditions -> 1/2 = 0.50
    assert suc == pytest.approx(0.50)


def test_remediated_metric_engine_empty_input():
    """Empty dataframe must return clean zero values without exception."""
    df_empty = pd.DataFrame()
    res = RemediatedMetricEngine.compute_comprehensive_metrics(df_empty)
    assert res["n_samples"] == 0
    assert res["exact_match_accuracy"] == 0.0
    assert res["remediated_suc_grounded"] == 0.0


def test_unconditional_vs_conditional_uar():
    """Conditional UAR evaluates hallucinations among incorrect answers."""
    df = pd.DataFrame({
        "exact_match_accuracy": [1.0, 1.0, 0.0, 0.0],
        "grounding_iou": [0.8, 0.8, 0.4, 0.4],
        "unsupported_answer_rate": [0.0, 0.0, 1.0, 0.0],
        "safe_useful_coverage": [1.0, 1.0, 0.0, 0.0]
    })
    res = RemediatedMetricEngine.compute_comprehensive_metrics(df)
    # Unconditional UAR = 1 / 4 = 0.25
    assert res["unsupported_answer_rate_unconditional"] == pytest.approx(0.25)
    # Conditional UAR = 1 unsupported among 2 incorrect = 0.50
    assert res["unsupported_answer_rate_conditional"] == pytest.approx(0.50)
