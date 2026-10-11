"""Phase 14.7 Small-Cluster (K=11) Family-Level Statistical Audit Engine.

Implements exact small-sample statistical inference over independent document family clusters:
1. Family-level paired aggregation
2. Exact paired permutation test (2^11 = 2,048 permutations)
3. Wilcoxon signed-rank test (exact / zero-handling)
4. Exact two-sided sign test (binomial)
5. Paired Student's t-test (df = 10)
6. Leave-One-Family-Out (LOFO) leverage and sensitivity analysis
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats


class SmallClusterStatisticalEngine:
    """Computes exact small-sample and cluster-level statistical tests."""

    @staticmethod
    def aggregate_family_level_outcomes(df_traces: pd.DataFrame, tau: float = 0.50) -> pd.DataFrame:
        """Aggregates per-query trace records into exactly 11 family-level outcome rows per condition."""
        records = []
        families = sorted(df_traces["family_id"].unique())
        conditions = sorted(df_traces["condition"].unique())

        for fam in families:
            fam_sub = df_traces[df_traces["family_id"] == fam]
            for cond in conditions:
                cond_sub = fam_sub[fam_sub["condition"] == cond]
                acc = cond_sub["exact_match_accuracy"].to_numpy().astype(float)
                iou = cond_sub["grounding_iou"].to_numpy().astype(float)
                uar = cond_sub["unsupported_answer_rate"].to_numpy().astype(float)
                lat = cond_sub["physical_latency_s"].to_numpy().astype(float)

                # Grounded SUC at threshold tau
                suc_grounded = (acc == 1.0) & (iou >= tau)

                # Error-conditional UAR
                err_mask = (acc == 0.0)
                cond_uar = float(np.mean(uar[err_mask])) if np.sum(err_mask) > 0 else 0.0

                records.append({
                    "family_id": fam,
                    "condition": cond,
                    "condition_name": cond_sub["condition_name"].iloc[0],
                    "n_queries": len(cond_sub["query_id"].unique()),
                    "n_evaluations": len(cond_sub),
                    "exact_match_accuracy": float(np.mean(acc)),
                    "grounding_iou_mean": float(np.mean(iou)),
                    "grounded_suc_tau50": float(np.mean(suc_grounded)),
                    "uar_unconditional": float(np.mean(uar)),
                    "uar_conditional": cond_uar,
                    "mean_latency_s": float(np.mean(lat))
                })
        return pd.DataFrame(records)

    @staticmethod
    def exact_paired_permutation_test(diffs: np.ndarray) -> Dict[str, Any]:
        """Calculates exact two-sided p-value via all 2^K sign-flip permutations.
        For K = 11, total permutations = 2^11 = 2,048.
        """
        k = len(diffs)
        observed_mean = float(np.mean(diffs))
        abs_observed = abs(observed_mean)

        # Generate all 2^k sign combinations (+1, -1)
        signs = np.array(list(itertools.product([-1, 1], repeat=k)))
        permuted_means = np.mean(signs * diffs, axis=1)

        # Two-sided p-value: fraction of permutations with |mean| >= |observed|
        extreme_count = int(np.sum(np.abs(permuted_means) >= abs_observed - 1e-12))
        p_val = extreme_count / len(permuted_means)

        return {
            "observed_mean": observed_mean,
            "total_permutations": len(permuted_means),
            "extreme_permutations": extreme_count,
            "exact_p_value": p_val
        }

    @staticmethod
    def leave_one_family_out_analysis(df_paired: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Any]:
        """Performs Leave-One-Family-Out (LOFO) jackknife sensitivity across all K clusters."""
        diffs = df_paired[col_b].to_numpy() - df_paired[col_a].to_numpy()
        families = df_paired["family_id"].tolist()
        k = len(diffs)

        overall_mean = float(np.mean(diffs))
        lofo_means = {}

        for i, fam in enumerate(families):
            lofo_diffs = np.delete(diffs, i)
            lofo_means[fam] = float(np.mean(lofo_diffs))

        lofo_vals = list(lofo_means.values())
        min_effect = min(lofo_vals)
        max_effect = max(lofo_vals)
        max_deviation = max(abs(v - overall_mean) for v in lofo_vals)
        sign_preserved = all((v > 0) == (overall_mean > 0) for v in lofo_vals)

        return {
            "overall_mean": overall_mean,
            "min_lofo_effect": min_effect,
            "max_lofo_effect": max_effect,
            "max_deviation": max_deviation,
            "sign_preserved_across_all_lofo": sign_preserved,
            "family_lofo_means": lofo_means
        }

    @classmethod
    def evaluate_comprehensive_family_tests(
        cls,
        df_family_outcomes: pd.DataFrame,
        metric_col: str,
        cond_baseline: str,
        cond_treatment: str,
        hypothesis_id: str,
        comparison_name: str
    ) -> Dict[str, Any]:
        """Runs the complete small-sample statistical test suite on a paired comparison."""
        piv = df_family_outcomes.pivot(index="family_id", columns="condition", values=metric_col).reset_index()
        k = len(piv)

        diffs = (piv[cond_treatment] - piv[cond_baseline]).to_numpy()
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / np.sqrt(k)

        # 1. Paired Student's t-test (df = 10)
        t_stat, t_pval = stats.ttest_1samp(diffs, 0.0)

        # 2. Exact Paired Permutation Test (2^11 = 2048)
        perm_res = cls.exact_paired_permutation_test(diffs)

        # 3. Wilcoxon Signed-Rank Test
        try:
            # Using method='exact' for small samples
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)

        # 4. Exact Sign Test
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_nonzero = n_pos + n_neg
        sign_pval = float(stats.binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided").pvalue) if n_nonzero > 0 else 1.0

        # 5. 95% Confidence Interval (Student's t, df = k-1)
        t_crit = stats.t.ppf(0.975, df=k-1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        # 6. Leave-One-Family-Out Analysis
        lofo_res = cls.leave_one_family_out_analysis(piv, cond_baseline, cond_treatment)

        return {
            "hypothesis_id": hypothesis_id,
            "comparison": comparison_name,
            "metric": metric_col,
            "cond_baseline": cond_baseline,
            "cond_treatment": cond_treatment,
            "k_clusters": k,
            "mean_delta_pct": round(mean_diff * 100.0, 2),
            "median_delta_pct": round(median_diff * 100.0, 2),
            "ci_lower_95_pct": round(ci_lower * 100.0, 2),
            "ci_upper_95_pct": round(ci_upper * 100.0, 2),
            "t_statistic": round(float(t_stat), 4),
            "p_val_t_test": round(float(t_pval), 6),
            "exact_permutation_p_val": round(float(perm_res["exact_p_value"]), 6),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_val": round(float(w_pval), 6),
            "positive_families": n_pos,
            "negative_families": n_neg,
            "sign_test_p_val": round(sign_pval, 6),
            "lofo_min_delta_pct": round(lofo_res["min_lofo_effect"] * 100.0, 2),
            "lofo_max_delta_pct": round(lofo_res["max_lofo_effect"] * 100.0, 2),
            "lofo_sign_preserved": lofo_res["sign_preserved_across_all_lofo"]
        }
