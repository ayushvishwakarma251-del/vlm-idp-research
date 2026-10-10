"""Phase 14.5 Reproducible Independent Audit Engine.

Provides self-contained routines to:
1. Verify raw traces (parse, schema, counts, no NaNs).
2. Recompute end-to-end metrics across B14-A to B14-D.
3. Recompute family-clustered paired bootstrap and Holm-Bonferroni statistics.
4. Verify table and figure consistency.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_PATH = REPO_ROOT / "experiments" / "phase14" / "traces" / "physical_inference_traces.jsonl"
HIST_T12_PATH = REPO_ROOT / "experiments" / "phase14" / "tables" / "table_12_end_to_end.csv"
HIST_T13_PATH = REPO_ROOT / "experiments" / "phase14" / "tables" / "table_13_statistical_tests.csv"
HIST_T04_PATH = REPO_ROOT / "experiments" / "phase14" / "tables" / "table_04_quantization_matrix.csv"

def load_traces() -> List[Dict[str, Any]]:
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def validate_traces_schema(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_fields = {
        "experiment_id", "condition", "condition_name", "seed", "query_id",
        "document_id", "family_id", "modality", "pages_fed", "page_reduction_pct",
        "input_tokens", "output_tokens", "physical_latency_s", "peak_vram_mb",
        "exact_match_accuracy", "grounding_iou", "unsupported_answer_rate",
        "safe_useful_coverage"
    }
    
    total = len(traces)
    valid_count = 0
    exp_ids = set()
    
    for r in traces:
        if set(r.keys()) != expected_fields:
            continue
        if r["experiment_id"] in exp_ids:
            continue
        exp_ids.add(r["experiment_id"])
        
        # Check values
        if r["physical_latency_s"] <= 0 or r["peak_vram_mb"] <= 0:
            continue
        if not (0.0 <= r["exact_match_accuracy"] <= 1.0):
            continue
        if not (0.0 <= r["grounding_iou"] <= 1.0):
            continue
        if not (0.0 <= r["unsupported_answer_rate"] <= 1.0):
            continue
            
        valid_count += 1
        
    return {
        "total_records": total,
        "valid_records": valid_count,
        "unique_experiment_ids": len(exp_ids),
        "is_valid": total == 1100 and valid_count == 1100
    }

def recompute_end_to_end_table(traces: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(traces)
    results = []
    
    for cond in ["B14-A", "B14-B", "B14-C", "B14-D"]:
        sub = df[df["condition"] == cond]
        results.append({
            "Condition": cond,
            "Exact_Match": sub["exact_match_accuracy"].mean(),
            "Grounding_IoU": sub["grounding_iou"].mean(),
            "Unsupported_Rate": sub["unsupported_answer_rate"].mean(),
            "Safe_Useful_Coverage": sub["safe_useful_coverage"].mean(),
            "Latency_s": sub["physical_latency_s"].mean(),
            "Page_Reduction_Pct": sub["page_reduction_pct"].mean()
        })
        
    return pd.DataFrame(results)

def verify_table12_concordance() -> Dict[str, Any]:
    traces = load_traces()
    df_recomp = recompute_end_to_end_table(traces)
    df_hist = pd.read_csv(HIST_T12_PATH)
    
    max_diff = 0.0
    for col in ["Exact_Match", "Grounding_IoU", "Unsupported_Rate", "Safe_Useful_Coverage", "Latency_s", "Page_Reduction_Pct"]:
        diff = np.max(np.abs(df_recomp[col].values - df_hist[col].values))
        if diff > max_diff:
            max_diff = float(diff)
            
    return {
        "max_discrepancy": max_diff,
        "is_concordant": max_diff < 1e-6
    }

def run_cluster_bootstrap(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    cluster_col: str = "family_id",
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    families = df[cluster_col].unique()
    n_families = len(families)

    diff_actual = float(np.mean(df[col_b] - df[col_a]))
    family_groups = {fam: sub for fam, sub in df.groupby(cluster_col)}

    boot_diffs = []
    for _ in range(n_resamples):
        sampled_fams = rng.choice(families, size=n_families, replace=True)
        sampled_sub = pd.concat([family_groups[fam] for fam in sampled_fams], ignore_index=True)
        boot_diffs.append(float(np.mean(sampled_sub[col_b] - sampled_sub[col_a])))

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": diff_actual,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val
    }
