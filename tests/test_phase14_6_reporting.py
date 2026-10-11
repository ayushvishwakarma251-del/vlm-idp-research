"""Unit tests for Phase 14.6 reporting pipeline, tables, and figures."""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "experiments" / "phase14_6" / "tables"
FIGURES_DIR = REPO_ROOT / "experiments" / "phase14_6" / "figures"
HIST_TABLES_DIR = REPO_ROOT / "experiments" / "phase14" / "tables"


def test_remediated_tables_exist():
    assert (TABLES_DIR / "table_04_quantization_matrix_remediated.csv").exists()
    assert (TABLES_DIR / "table_12_end_to_end_remediated.csv").exists()
    assert (TABLES_DIR / "table_13_statistical_tests_remediated.csv").exists()
    assert (TABLES_DIR / "table_14_failures_remediated.csv").exists()


def test_table_12_remediated_structure_and_metrics():
    df12 = pd.read_csv(TABLES_DIR / "table_12_end_to_end_remediated.csv")
    assert len(df12) == 4
    assert set(df12["Condition"]) == {"B14-A", "B14-B", "B14-C", "B14-D"}
    # Historical SUC must be marked as coupled
    assert all(df12["Historical_SUC_Coupled"] == True)
    # Remediated SUC must show true grounding constraint in B14-A
    b14_a = df12[df12["Condition"] == "B14-A"].iloc[0]
    assert b14_a["Remediated_SUC_Grounded_IoU50"] < b14_a["Exact_Match_Accuracy"]


def test_table_13_remediated_holm_adjustment():
    df13 = pd.read_csv(TABLES_DIR / "table_13_statistical_tests_remediated.csv")
    assert len(df13) == 2
    # Verify independent clusters = 11
    assert all(df13["Independent_Clusters"] == 11)
    # Verify both raw and adjusted p-values are reported
    assert "P_Value_Raw" in df13.columns
    assert "P_Value_Holm_Adjusted" in df13.columns
    assert all(df13["P_Value_Holm_Adjusted"] <= 0.0002)
    assert all(df13["Decision"] == "SUPPORTED")


def test_figure_svg_provenance_banner():
    svg_path = FIGURES_DIR / "figure_01_remediated_suc_comparison.svg"
    assert svg_path.exists()
    content = svg_path.read_text(encoding="utf-8")
    assert "Execution Class: SIMULATED_END_TO_END" in content


def test_historical_tables_remain_untouched():
    """Historical tables in phase14 must exist and remain untouched."""
    hist_t12 = HIST_TABLES_DIR / "table_12_end_to_end.csv"
    hist_t13 = HIST_TABLES_DIR / "table_13_statistical_tests.csv"
    assert hist_t12.exists()
    assert hist_t13.exists()
    df_old13 = pd.read_csv(hist_t13)
    # Old table had raw p-values without Holm adjustment column
    assert "P_Value_Holm_Adjusted" not in df_old13.columns
