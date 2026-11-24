import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SimulatedAlert:
    correlation_id: str
    action_type: str
    status: str
    source_ids: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "correlation_id": self.correlation_id,
                "action_type": self.action_type,
                "status": self.status,
                "source_ids": list(self.source_ids),
                "simulated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class SimulatedAlertSink(Protocol):
    def write(self, alert: SimulatedAlert) -> None: ...


class JsonlSimulatedAlertSink:
    """Writes local demonstration events; it never delivers external alerts."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, alert: SimulatedAlert) -> None:
        if alert.status != "awaiting_human_review":
            raise ValueError("Simulated alerts must await human review")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as output_file:
            output_file.write(alert.to_json())
            output_file.write("\n")
