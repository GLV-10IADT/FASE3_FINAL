import json
from pathlib import Path

from medical_assistant.audit import AuditEvent, JsonlAuditLogger


def make_event() -> AuditEvent:
    return AuditEvent(
        correlation_id="correlation-1",
        occurred_at="2026-11-22T13:00:00+00:00",
        actor_id="user-1",
        actor_role="clinician",
        intent="knowledge_query",
        status="complete",
        review_required=False,
        review_reason=None,
        source_refs=("data/synthetic/protocols.json#DOC-SYNTH-001",),
        patient_data_accessed=False,
        model_version="deterministic-baseline-1",
        prompt_version="none",
        duration_ms=4,
        retrieval_version="intfloat/multilingual-e5-small@test-revision",
    )


def test_jsonl_logger_appends_parseable_events(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "audit.jsonl"
    logger = JsonlAuditLogger(log_path)

    logger.write(make_event())
    logger.write(make_event())

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["correlation_id"] == "correlation-1" for line in lines)
    assert all(
        json.loads(line)["retrieval_version"]
        == "intfloat/multilingual-e5-small@test-revision"
        for line in lines
    )
