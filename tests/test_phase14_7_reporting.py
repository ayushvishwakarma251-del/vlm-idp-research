"""Unit tests for Phase 14.7 Master Reporting and Workflow Engine."""

from pathlib import Path
import pytest
from src.phase14_7.reporting import Phase14_7WorkflowEngine

TRACES_PATH = Path("experiments/phase14/traces/physical_inference_traces.jsonl")


def test_master_pipeline_execution(tmp_path):
    engine = Phase14_7WorkflowEngine(TRACES_PATH)
    res = engine.run_complete_audit_pipeline(tmp_path)
    
    assert res["status"] == "SUCCESS"
    assert (tmp_path / "tables" / "threshold_sensitivity.csv").exists()
    assert (tmp_path / "tables" / "family_level_statistical_results.csv").exists()
    assert (tmp_path / "audit" / "claim_evidence_ledger.csv").exists()
    assert (tmp_path / "figures" / "grounded_coverage_threshold_sensitivity.svg").exists()
