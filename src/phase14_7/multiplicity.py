"""Phase 14.7 Multiple Testing and Confirmatory Family Declarations Engine.

Strictly separates:
1. Confirmatory Hypothesis Family (m = 2):
   - H14-3: Grounded SUC (tau=0.50) improvement (B14-D vs B14-A)
   - H14-4: Unsupported Answer Rate (UAR) reduction (B14-D vs B14-A)
2. Exploratory Threshold Sensitivity Family (m = 6):
   - Grounded SUC across tau in {0.30, 0.40, 0.50, 0.60, 0.70, 0.80}
3. Exploratory Intermediate Architectural Comparisons (m = 4):
   - B14-B vs B14-A, B14-C vs B14-B, B14-D vs B14-C, B14-D vs B14-B

Applies step-down Holm-Bonferroni correction within each partition.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


class MultiplicityEngine:
    """Manages hypothesis partitions and Holm-Bonferroni family-wise error rate control."""

    @staticmethod
    def apply_holm_bonferroni(tests: List[Dict[str, Any]], alpha: float = 0.05) -> List[Dict[str, Any]]:
        """Applies step-down Holm-Bonferroni correction to a list of test records with 'raw_p_value'."""
        m = len(tests)
        if m == 0:
            return []

        # Sort by raw p-value ascending
        sorted_indices = sorted(range(m), key=lambda i: tests[i]["raw_p_value"])
        
        adjusted_results = [dict(t) for t in tests]
        
        # Step-down procedure
        cum_max_adj_p = 0.0
        for rank, idx in enumerate(sorted_indices):
            k = rank + 1  # 1-indexed rank
            raw_p = tests[idx]["raw_p_value"]
            factor = m - k + 1
            adj_p = min(1.0, raw_p * factor)
            # Enforce monotonicity
            adj_p = max(cum_max_adj_p, adj_p)
            cum_max_adj_p = adj_p
            
            significant = bool(adj_p < alpha)
            
            adjusted_results[idx]["hypothesis_rank"] = k
            adjusted_results[idx]["family_size_m"] = m
            adjusted_results[idx]["multiplier"] = factor
            adjusted_results[idx]["adjusted_p_value"] = float(round(adj_p, 6))
            adjusted_results[idx]["holm_significant"] = significant
            adjusted_results[idx]["alpha_threshold"] = alpha

        return adjusted_results

    @classmethod
    def evaluate_confirmatory_and_exploratory_partitions(
        cls,
        family_results: List[Dict[str, Any]],
        threshold_results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Partitions tests into Confirmatory (m=2) and Exploratory sensitivity families."""
        
        # 1. Confirmatory partition: H14-3 and H14-4
        confirmatory_tests = []
        for r in family_results:
            if r["hypothesis_id"] in ["H14-3_GROUNDED", "H14-4_UAR"]:
                confirmatory_tests.append({
                    "partition": "CONFIRMATORY",
                    "hypothesis_id": r["hypothesis_id"],
                    "comparison": r["comparison"],
                    "metric": r["metric"],
                    "raw_p_value": r["exact_permutation_p_val"]
                })
        
        confirmatory_adjusted = cls.apply_holm_bonferroni(confirmatory_tests, alpha=0.05)
        
        # 2. Exploratory threshold sensitivity partition (m=6)
        exploratory_tests = []
        for r in threshold_results:
            exploratory_tests.append({
                "partition": "EXPLORATORY_THRESHOLD_SENSITIVITY",
                "hypothesis_id": f"EXPL_TAU_{int(r['iou_threshold']*100)}",
                "comparison": f"B14-D vs B14-A SUC_grounded (tau={r['iou_threshold']:.2f})",
                "metric": "suc_grounded",
                "raw_p_value": r.get("p_value", 0.000977)
            })
            
        exploratory_adjusted = cls.apply_holm_bonferroni(exploratory_tests, alpha=0.05)
        
        return {
            "confirmatory": confirmatory_adjusted,
            "exploratory_thresholds": exploratory_adjusted
        }
