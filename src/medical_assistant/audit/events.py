import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AuditActor:
    user_id: str
    role: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("Audit actor user_id must be a non-empty string")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("Audit actor role must be a non-empty string")


@dataclass(frozen=True)
class AuditEvent:
    correlation_id: str
    occurred_at: str
    actor_id: str
    actor_role: str
    intent: str
    status: str
    review_required: bool
    review_reason: str | None
    source_refs: tuple[str, ...]
    patient_data_accessed: bool
    model_version: str
    prompt_version: str
    duration_ms: int
    error_code: str | None = None
    action_refs: tuple[str, ...] = ()
    model_invoked: bool = False
    translation_used: bool = False
    translation_version: str | None = None
    retrieval_version: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "correlation_id": self.correlation_id,
                "occurred_at": self.occurred_at,
                "actor_id": self.actor_id,
                "actor_role": self.actor_role,
                "intent": self.intent,
                "status": self.status,
                "review_required": self.review_required,
                "review_reason": self.review_reason,
                "source_refs": list(self.source_refs),
                "patient_data_accessed": self.patient_data_accessed,
                "model_version": self.model_version,
                "prompt_version": self.prompt_version,
                "duration_ms": self.duration_ms,
                "error_code": self.error_code,
                "action_refs": list(self.action_refs),
                "model_invoked": self.model_invoked,
                "translation_used": self.translation_used,
                "translation_version": self.translation_version,
                "retrieval_version": self.retrieval_version,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...


class JsonlAuditLogger:
    """Append structured audit events to a JSON Lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, event: AuditEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as audit_file:
            audit_file.write(event.to_json())
            audit_file.write("\n")
