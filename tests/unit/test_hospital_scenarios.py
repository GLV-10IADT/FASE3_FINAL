import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from medical_assistant.audit import AuditActor, AuditEvent
from medical_assistant.patient_data import SyntheticPatientStore, SyntheticStaffStore
from medical_assistant.retrieval import DocumentIndex, load_synthetic_documents
from medical_assistant.workflows import (
    AssistantWorkflow,
    JsonlSimulatedAlertSink,
    WorkflowOperation,
)


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = ROOT / "tests" / "fixtures" / "hospital_scenarios.json"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
SCENARIOS = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


class RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class CountingChatModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "counting-hospital-scenario-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        self.calls += 1
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Resposta sintética."))
            ]
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item["scenario_id"])
def test_frozen_synthetic_hospital_scenarios(
    scenario: dict[str, Any],
    tmp_path: Path,
) -> None:
    audit_sink = RecordingAuditSink()
    model = CountingChatModel()
    patient_store = SyntheticPatientStore(SYNTHETIC_DIR / "patients.json")
    staff = SyntheticStaffStore(SYNTHETIC_DIR / "staff.json").read_staff(
        scenario["user_id"]
    )
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        patient_store,
        audit_sink,
        chat_model=model,
        model_version="fake-model+synthetic-adapter",
        simulated_alert_sink=JsonlSimulatedAlertSink(tmp_path / "alerts.jsonl"),
    )
    patient_id = scenario["patient_id"]
    result = workflow.run(
        scenario["question"],
        actor=AuditActor(user_id=staff.user_id, role=staff.role.value),
        patient_id=patient_id,
        access_context=staff.access_context() if patient_id is not None else None,
        operation=WorkflowOperation(scenario["operation"]),
    )

    assert result.status.value == scenario["expected_status"]
    assert result.model_invoked is scenario["expected_model_invoked"]
    assert model.calls == int(scenario["expected_model_invoked"])
    source_ids = (
        {citation.source_id for citation in result.response.sources}
        if result.response is not None
        else set()
    )
    assert set(scenario["expected_source_ids"]).issubset(source_ids)
    assert (
        result.simulated_alert.status if result.simulated_alert is not None else None
    ) == scenario["expected_alert_status"]
    assert len(audit_sink.events) == 1
    assert audit_sink.events[0].model_invoked is scenario["expected_model_invoked"]
    assert patient_id is None or patient_id not in audit_sink.events[0].to_json()
    assert audit_sink.events[0].action_refs == (
        ("pending_exam:awaiting_human_review",)
        if scenario["expected_alert_status"] is not None
        else ()
    )
