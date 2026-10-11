"""Phase 14.6 Statistical Testing and Multiplicity Correction Engine.

Provides mathematically verified cluster bootstrap inference and Holm-Bonferroni
step-down familywise error rate adjustment.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies the Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).

    Algorithm:
    1. Given m raw p-values, sort them in ascending order: p_(1) <= p_(2) <= ... <= p_(m)
    2. Adjust each p-value: adj_(i) = (m - i + 1) * p_(i)
    3. Enforce monotonicity: adj'_(i) = max(adj'_(i-1), adj_(i))
    4. Bound to [0.0, 1.0]
    5. Restore original input ordering.

    Guarantees:
    - Strictly preserves original input order.
    - Preserves monotonic non-decreasing order in sorted ranks.
    - Handles ties, zero, one, and single-test families correctly.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [min(1.0, max(0.0, float(p_values[0])))]

    # Validate inputs
    for p in p_values:
        if not (0.0 <= p <= 1.0) and not np.isnan(p):
            raise ValueError(f"p-value must be in [0.0, 1.0], got {p}")

    # Index and sort
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_adj = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        raw_adj = multiplier * p_val
        bounded_adj = min(1.0, max(0.0, raw_adj))
        running_max = max(running_max, bounded_adj)
        sorted_adj[rank] = min(1.0, running_max)

    # Reconstruct original order
    adjusted = [0.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        adjusted[orig_idx] = sorted_adj[rank]

    return adjusted


def cluster_bootstrap_family_paired(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    cluster_col: str = "family_id",
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """Performs cluster bootstrap hypothesis testing at the independent document family level.

    Ensures that resampling occurs at the family (cluster) level rather than the query level,
    preserving internal correlation structures across queries within a family.
    """
    rng = np.random.RandomState(seed)
    families = np.array(df[cluster_col].unique())
    n_families = len(families)

    diff_actual = float(np.mean(df[col_b] - df[col_a]))

    # Group data by cluster
    family_groups = {fam: sub for fam, sub in df.groupby(cluster_col)}

    boot_diffs = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sampled_fams = rng.choice(families, size=n_families, replace=True)
        sampled_sub = pd.concat([family_groups[fam] for fam in sampled_fams], ignore_index=True)
        boot_diffs[b] = float(np.mean(sampled_sub[col_b] - sampled_sub[col_a]))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided empirical p-value testing null hypothesis (delta == 0)
    p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": diff_actual,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_raw": p_val,
        "n_resamples": n_resamples,
        "n_clusters": n_families,
        "cluster_col": cluster_col
    }
