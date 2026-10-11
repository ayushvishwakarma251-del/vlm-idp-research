"""Phase 14.7 Grounded Metric Validation Engine.

Formulates and validates mathematical constructs for:
1. Raw Coverage (non-abstention rate)
2. Spatial Grounded Coverage (IoU >= tau and non-abstained)
3. Correct Grounded Coverage (SUC_grounded: Accuracy == 1 and IoU >= tau and non-abstained)
4. Conditional Accuracy
5. Unconditional and Error-Conditional Unsupported Answer Rates (UAR)
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class GroundedMetricValidator:
    """Computes and validates decomposed evaluation metrics with strict denominator tracking."""

    @staticmethod
    def compute_decomposed_metrics(
        df_condition: pd.DataFrame,
        iou_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Calculates decomposed metrics for a specific condition with full denominator accountability.

        Parameters:
        - df_condition: DataFrame containing trace records for a condition.
        - iou_threshold: The IoU threshold tau for spatial evidence verification.
        """
        n_eligible = len(df_condition)
        if n_eligible == 0:
            return {
                "n_eligible": 0,
                "n_answered": 0,
                "n_abstained": 0,
                "coverage_pct": 0.0,
                "accuracy_pct": 0.0,
                "conditional_accuracy_pct": 0.0,
                "grounded_coverage_pct": 0.0,
                "correct_grounded_coverage_pct": 0.0,
                "mean_grounding_iou": 0.0,
                "uar_unconditional_pct": 0.0,
                "uar_conditional_pct": 0.0
            }

        acc = df_condition["exact_match_accuracy"].to_numpy().astype(float)
        iou = df_condition["grounding_iou"].to_numpy().astype(float)
        uar = df_condition["unsupported_answer_rate"].to_numpy().astype(float)

        # In Phase 14 traces, abstention is recorded when an answer is abstained.
        # Check if abstained column exists, else 0.0
        if "abstained" in df_condition.columns:
            abstained = df_condition["abstained"].to_numpy().astype(float)
        else:
            abstained = np.zeros_like(acc)

        answered = (abstained == 0.0)
        n_answered = int(np.sum(answered))
        n_abstained = int(np.sum(~answered))

        # 1. Coverage: Fraction of eligible queries that receive an answer
        coverage = n_answered / n_eligible

        # 2. Raw Exact Match Accuracy across all eligible queries
        accuracy = float(np.mean(acc))

        # 3. Conditional Accuracy among answered queries
        cond_acc = float(np.mean(acc[answered])) if n_answered > 0 else 0.0

        # 4. Spatial Grounded Coverage: Answered AND IoU >= threshold
        grounded_mask = answered & (iou >= iou_threshold)
        grounded_coverage = float(np.mean(grounded_mask))

        # 5. Correct Grounded Coverage (SUC_grounded): Answered AND Acc == 1.0 AND IoU >= threshold
        correct_grounded_mask = answered & (acc == 1.0) & (iou >= iou_threshold)
        correct_grounded_coverage = float(np.mean(correct_grounded_mask))

        # 6. Unconditional UAR: unsupported answers / all eligible queries
        uar_unconditional = float(np.mean(uar))

        # 7. Error-Conditional UAR: unsupported answers among incorrect queries
        incorrect_mask = (acc == 0.0)
        n_incorrect = int(np.sum(incorrect_mask))
        if n_incorrect > 0:
            uar_conditional = float(np.sum(incorrect_mask & (uar == 1.0)) / n_incorrect)
        else:
            uar_conditional = 0.0

        return {
            "n_eligible": n_eligible,
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "coverage_pct": round(coverage * 100.0, 2),
            "accuracy_pct": round(accuracy * 100.0, 2),
            "conditional_accuracy_pct": round(cond_acc * 100.0, 2),
            "grounded_coverage_pct": round(grounded_coverage * 100.0, 2),
            "correct_grounded_coverage_pct": round(correct_grounded_coverage * 100.0, 2),
            "mean_grounding_iou": round(float(np.mean(iou)), 4),
            "uar_unconditional_pct": round(uar_unconditional * 100.0, 2),
            "uar_conditional_pct": round(uar_conditional * 100.0, 2)
        }

    @staticmethod
    def audit_denominator_fairness(df_all: pd.DataFrame) -> Dict[str, Any]:
        """Verifies whether all conditions share the exact same query population and eligibility."""
        conditions = df_all["condition"].unique()
        queries_per_condition = {}
        for c in conditions:
            sub = df_all[df_all["condition"] == c]
            queries_per_condition[c] = set(sub["query_id"].unique())

        ref_cond = conditions[0]
        ref_queries = queries_per_condition[ref_cond]
        identical = True
        for c in conditions[1:]:
            if queries_per_condition[c] != ref_queries:
                identical = False
                break

        return {
            "conditions_audited": list(conditions),
            "total_queries_per_condition": {c: len(queries_per_condition[c]) for c in conditions},
            "queries_strictly_identical": identical,
            "ref_query_count": len(ref_queries)
        }
