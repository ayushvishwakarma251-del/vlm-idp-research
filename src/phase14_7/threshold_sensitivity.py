"""Phase 14.7 IoU Threshold Sensitivity Engine.

Evaluates metric stability, rank-order consistency, and effect-size behavior
across a preregistered grid of IoU thresholds: tau in {0.30, 0.40, 0.50, 0.60, 0.70, 0.80}.
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.phase14_7.metric_validation import GroundedMetricValidator

GRID_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


class ThresholdSensitivityEngine:
    """Executes sensitivity grid sweeps and generates comparison tables and figures."""

    @staticmethod
    def evaluate_grid(df_all: pd.DataFrame, thresholds: List[float] = GRID_THRESHOLDS) -> pd.DataFrame:
        """Sweeps across thresholds and conditions to compute grounded coverage metrics."""
        rows = []
        for tau in thresholds:
            for cond in sorted(df_all["condition"].unique()):
                sub = df_all[df_all["condition"] == cond]
                metrics = GroundedMetricValidator.compute_decomposed_metrics(sub, iou_threshold=tau)
                rows.append({
                    "tau": tau,
                    "condition": cond,
                    "condition_name": sub["condition_name"].iloc[0],
                    "n_eligible": metrics["n_eligible"],
                    "exact_match_pct": metrics["accuracy_pct"],
                    "spatial_grounded_coverage_pct": metrics["grounded_coverage_pct"],
                    "correct_grounded_coverage_pct": metrics["correct_grounded_coverage_pct"],
                    "mean_iou": metrics["mean_grounding_iou"]
                })
        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_paired_effects(df_all: pd.DataFrame, thresholds: List[float] = GRID_THRESHOLDS) -> pd.DataFrame:
        """Computes paired differences (Condition - B14-A) across thresholds."""
        grid_df = ThresholdSensitivityEngine.evaluate_grid(df_all, thresholds)
        piv = grid_df.pivot(index="tau", columns="condition", values="correct_grounded_coverage_pct")

        rows = []
        for tau in thresholds:
            b14_a = piv.loc[tau, "B14-A"]
            b14_b = piv.loc[tau, "B14-B"]
            b14_c = piv.loc[tau, "B14-C"]
            b14_d = piv.loc[tau, "B14-D"]

            rows.append({
                "tau": tau,
                "B14_A_pct": b14_a,
                "B14_B_pct": b14_b,
                "B14_C_pct": b14_c,
                "B14_D_pct": b14_d,
                "delta_B_vs_A_pp": round(b14_b - b14_a, 2),
                "delta_C_vs_A_pp": round(b14_c - b14_a, 2),
                "delta_D_vs_A_pp": round(b14_d - b14_a, 2),
                "ranking": "D > C > B > A" if (b14_d >= b14_c >= b14_b >= b14_a) else "RANK_REVERSAL"
            })
        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_family_level_threshold_effects(df_all: pd.DataFrame, thresholds: List[float] = GRID_THRESHOLDS) -> pd.DataFrame:
        """Calculates family-level SUC across all thresholds and conditions."""
        rows = []
        families = sorted(df_all["family_id"].unique())
        for tau in thresholds:
            for fam in families:
                sub_fam = df_all[df_all["family_id"] == fam]
                fam_res = {"tau": tau, "family_id": fam}
                for cond in sorted(df_all["condition"].unique()):
                    cond_sub = sub_fam[sub_fam["condition"] == cond]
                    m = GroundedMetricValidator.compute_decomposed_metrics(cond_sub, iou_threshold=tau)
                    fam_res[f"SUC_{cond}"] = m["correct_grounded_coverage_pct"]
                fam_res["delta_D_minus_A_pp"] = round(fam_res["SUC_B14-D"] - fam_res["SUC_B14-A"], 2)
                rows.append(fam_res)
        return pd.DataFrame(rows)

    @staticmethod
    def generate_figures(
        sensitivity_df: pd.DataFrame,
        out_dir: Path
    ) -> Dict[str, Path]:
        """Generates SVG figures for threshold sensitivity."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Grounded Coverage vs Tau
        piv = sensitivity_df.pivot(index="tau", columns="condition", values="correct_grounded_coverage_pct")
        
        # Build SVG line plot
        # Coordinates: tau 0.30..0.80 -> X: 100..700; SUC 0..100% -> Y: 380..80
        def x_pos(tau): return 100 + (tau - 0.30) / 0.50 * 600
        def y_pos(val): return 380 - (val / 100.0) * 300

        lines_svg = []
        colors = {"B14-A": "#ef4444", "B14-B": "#f59e0b", "B14-C": "#00e5ff", "B14-D": "#38ef7d"}
        
        for cond, col in colors.items():
            pts = []
            for tau in GRID_THRESHOLDS:
                val = piv.loc[tau, cond]
                pts.append(f"{x_pos(tau):.1f},{y_pos(val):.1f}")
            poly = " ".join(pts)
            lines_svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="3" points="{poly}" />')
            for tau in GRID_THRESHOLDS:
                val = piv.loc[tau, cond]
                lines_svg.append(f'<circle cx="{x_pos(tau):.1f}" cy="{y_pos(val):.1f}" r="4" fill="{col}" />')

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" width="800" height="450" style="background:#070a13; font-family:sans-serif;">
  <text x="40" y="40" fill="#f8fafc" font-size="18" font-weight="bold">Phase 14.7: Grounded SUC Sensitivity across IoU Thresholds (0.30 to 0.80)</text>
  <text x="40" y="65" fill="#94a3b8" font-size="12">Testing Condition Invariance: B14-D (Proposed) vs B14-C, B14-B, B14-A</text>

  <!-- Legend -->
  <line x1="520" y1="35" x2="540" y2="35" stroke="#38ef7d" stroke-width="3" />
  <text x="548" y="39" fill="#f8fafc" font-size="12">B14-D (Proposed)</text>
  <line x1="520" y1="55" x2="540" y2="55" stroke="#00e5ff" stroke-width="3" />
  <text x="548" y="59" fill="#f8fafc" font-size="12">B14-C (Grounded)</text>
  <line x1="680" y1="35" x2="700" y2="35" stroke="#f59e0b" stroke-width="3" />
  <text x="708" y="39" fill="#f8fafc" font-size="12">B14-B (Retrieval)</text>
  <line x1="680" y1="55" x2="700" y2="55" stroke="#ef4444" stroke-width="3" />
  <text x="708" y="59" fill="#f8fafc" font-size="12">B14-A (Full-Doc)</text>

  <!-- Grid lines -->
  <line x1="100" y1="380" x2="700" y2="380" stroke="#334155" stroke-width="1.5" />
  <line x1="100" y1="380" x2="100" y2="80" stroke="#334155" stroke-width="1.5" />
  
  <!-- Y ticks -->
  <text x="60" y="385" fill="#64748b" font-size="11">0%</text>
  <text x="60" y="305" fill="#64748b" font-size="11">25%</text>
  <text x="60" y="230" fill="#64748b" font-size="11">50%</text>
  <text x="60" y="155" fill="#64748b" font-size="11">75%</text>
  <text x="55" y="85" fill="#64748b" font-size="11">100%</text>

  <!-- X ticks -->
  {" ".join(f'<text x="{x_pos(t)-12}" y="405" fill="#94a3b8" font-size="12">t={t:.2f}</text>' for t in GRID_THRESHOLDS)}

  <!-- Data lines -->
  {" ".join(lines_svg)}
</svg>"""

        p_fig1 = out_dir / "grounded_coverage_threshold_sensitivity.svg"
        p_fig1.write_text(svg_content, encoding="utf-8")
        paths["fig_sensitivity"] = p_fig1

        return paths
