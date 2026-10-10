import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.phase14_5.audit_engine import (
    load_traces,
    validate_traces_schema,
    recompute_end_to_end_table,
    verify_table12_concordance,
    run_cluster_bootstrap
)

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_trace_schema_and_counts():
    """Verify raw inference traces have 1,100 valid records with exact schema."""
    traces = load_traces()
    res = validate_traces_schema(traces)
    assert res["total_records"] == 1100
    assert res["valid_records"] == 1100
    assert res["unique_experiment_ids"] == 1100
    assert res["is_valid"] is True

def test_end_to_end_table12_concordance():
    """Verify that recomputed Table 12 matches historical Table 12 to machine precision."""
    res = verify_table12_concordance()
    assert res["is_concordant"] is True
    assert res["max_discrepancy"] < 1e-6

def test_statistical_tests_table13_concordance():
    """Verify that recomputed cluster bootstrap tests reproduce Table 13."""
    traces = load_traces()
    df = pd.DataFrame(traces)
    df_agg = df.groupby(["condition", "family_id"]).agg({
        "exact_match_accuracy": "mean",
        "unsupported_answer_rate": "mean"
    }).reset_index()
    piv = df_agg.pivot(index="family_id", columns="condition")

    # H14-3
    df_h14_3 = pd.DataFrame({
        "family_id": list(piv.index),
        "A": piv["exact_match_accuracy"]["B14-A"].values,
        "B": piv["exact_match_accuracy"]["B14-B"].values
    })
    res_3 = run_cluster_bootstrap(df_h14_3, "A", "B", seed=42)
    assert abs(res_3["mean_diff"] - 0.083636) < 1e-4
    assert abs(res_3["ci_lower"] - 0.036364) < 1e-4
    assert abs(res_3["ci_upper"] - 0.134545) < 1e-4
    assert res_3["p_value"] < 0.001

    # H14-4
    df_h14_4 = pd.DataFrame({
        "family_id": list(piv.index),
        "A": piv["unsupported_answer_rate"]["B14-C"].values,
        "B": piv["unsupported_answer_rate"]["B14-A"].values
    })
    res_4 = run_cluster_bootstrap(df_h14_4, "A", "B", seed=42)
    assert abs(res_4["mean_diff"] - 0.210909) < 1e-4
    assert abs(res_4["ci_lower"] - 0.178182) < 1e-4
    assert abs(res_4["ci_upper"] - 0.247273) < 1e-4
    assert res_4["p_value"] < 0.001

def test_suc_mathematical_collapse():
    """Audit verification that SUC is identically equal to accuracy in traces."""
    traces = load_traces()
    for r in traces:
        assert r["safe_useful_coverage"] == r["exact_match_accuracy"]

def test_split_family_leakage():
    """Verify zero family-level overlap between train, val, and test splits."""
    splits_path = REPO_ROOT / "data" / "phase12_authentic" / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    train_fams = set(splits["family_splits"]["train"])
    val_fams = set(splits["family_splits"]["validation"])
    test_fams = set(splits["family_splits"]["test"])
    
    assert len(train_fams.intersection(val_fams)) == 0
    assert len(train_fams.intersection(test_fams)) == 0
    assert len(val_fams.intersection(test_fams)) == 0

def test_quantization_table04_validity():
    """Verify quantization table 04 contents and physical measurements."""
    t04_path = REPO_ROOT / "experiments" / "phase14" / "tables" / "table_04_quantization_matrix.csv"
    df = pd.read_csv(t04_path)
    assert len(df) == 3
    assert set(df["Precision"]) == {"Q0_FP16", "Q1_INT8", "Q2_INT4"}
    
    # INT4 reduces VRAM relative to FP16
    fp16_vram = df[df["Precision"] == "Q0_FP16"]["Peak_VRAM_MB"].iloc[0]
    int4_vram = df[df["Precision"] == "Q2_INT4"]["Peak_VRAM_MB"].iloc[0]
    reduction = (fp16_vram - int4_vram) / fp16_vram
    assert reduction > 0.50  # Over 50% VRAM reduction
