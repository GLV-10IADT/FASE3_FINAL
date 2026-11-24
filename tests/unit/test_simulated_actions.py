import json
from pathlib import Path

import pytest

from medical_assistant.workflows import JsonlSimulatedAlertSink, SimulatedAlert


def test_simulated_alert_sink_writes_only_local_reviewable_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outbox" / "alerts.jsonl"
    sink = JsonlSimulatedAlertSink(path)
    alert = SimulatedAlert(
        correlation_id="correlation-synthetic",
        action_type="pending_exam",
        status="awaiting_human_review",
        source_ids=("EXAM-SYNTH-001",),
    )

    sink.write(alert)

    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert event == {
        "correlation_id": "correlation-synthetic",
        "action_type": "pending_exam",
        "status": "awaiting_human_review",
        "source_ids": ["EXAM-SYNTH-001"],
        "simulated": True,
    }


def test_simulated_alert_sink_rejects_non_reviewable_action(tmp_path: Path) -> None:
    sink = JsonlSimulatedAlertSink(tmp_path / "alerts.jsonl")
    alert = SimulatedAlert(
        correlation_id="correlation-synthetic",
        action_type="pending_exam",
        status="sent",
        source_ids=("EXAM-SYNTH-001",),
    )

    with pytest.raises(ValueError, match="await human review"):
        sink.write(alert)

