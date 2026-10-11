"""Unit and regression tests for Phase 14.6 execution provenance audit."""

import json
import hashlib
from pathlib import Path
import pytest

from src.phase14_6.provenance import ExecutionClass, ProvenanceAuditor

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = REPO_ROOT / "experiments" / "phase14" / "traces" / "physical_inference_traces.jsonl"
SIDECAR_PATH = REPO_ROOT / "experiments" / "phase14_6" / "audit" / "trace_provenance_sidecar.json"
EXPECTED_SHA256 = "973ebd3af67e3fee0c401f39b280790f06b05df6ed45adf18e52d3bf8d7bf9cc"


def test_historical_trace_immutability():
    """Historical trace file must maintain its exact frozen SHA-256 digest."""
    assert TRACE_PATH.exists(), "Historical trace file missing"
    with open(TRACE_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, f"Hash mismatch: {actual_hash} != {EXPECTED_SHA256}"


def test_provenance_sidecar_integrity():
    """Sidecar manifest must accurately classify all 1,100 traces."""
    assert SIDECAR_PATH.exists(), "Sidecar manifest missing"
    with open(SIDECAR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_records"] == 1100
    dist = data["execution_class_distribution"]
    assert dist[ExecutionClass.SIMULATED_END_TO_END.value] == 1100
    assert dist[ExecutionClass.PHYSICAL_MODEL_INFERENCE.value] == 0
    assert len(data["records"]) == 1100


def test_no_silent_promotion_to_physical_inference():
    """Simulation records anchored to telemetry cannot be promoted to physical inference."""
    sample_trace = {
        "experiment_id": "EXP14_B14-A_S42_q_doc_auth_42_01_val",
        "condition": "B14-A",
        "seed": 42,
        "peak_vram_mb": 530.95,
        "physical_latency_s": 7.4687
    }
    result = ProvenanceAuditor.classify_trace(sample_trace)
    assert result["execution_class"] == ExecutionClass.SIMULATED_END_TO_END.value
    assert result["actual_forward_pass_executed"] is False
    assert result["simulation_flag"] is True


def test_unknown_provenance_rule():
    """Traces without unambiguous origin are classified as UNKNOWN_PROVENANCE."""
    unknown_trace = {
        "experiment_id": "SYNTHETIC_UNKNOWN_001",
        "condition": "CUSTOM",
        "seed": 999
    }
    result = ProvenanceAuditor.classify_trace(unknown_trace)
    assert result["execution_class"] == ExecutionClass.UNKNOWN_PROVENANCE.value
    assert result["actual_forward_pass_executed"] is False
