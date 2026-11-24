from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from medical_assistant.audit import AuditActor
from medical_assistant.patient_data import PatientRole, SyntheticStaffMember
from medical_assistant.workflows import (
    RequestIntent,
    WorkflowOperation,
    WorkflowResult,
    WorkflowStatus,
)


ROOT = Path(__file__).resolve().parents[2]


class RecordingWorkflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, question: str, **kwargs: object) -> WorkflowResult:
        self.calls.append((question, kwargs))
        return WorkflowResult(
            status=WorkflowStatus.INSUFFICIENT_EVIDENCE,
            intent=RequestIntent.SYNTHETIC_CASE_REVIEW,
            response=None,
        )


def test_streamlit_interview_uses_chat_and_never_uses_patient_context() -> None:
    workflow = RecordingWorkflow()
    runtime = SimpleNamespace(
        workflow=workflow,
        actor=AuditActor(user_id="demo", role="clinician"),
        staff_member=SyntheticStaffMember(
            "demo",
            PatientRole.CLINICIAN,
            frozenset(),
        ),
        protocol_count=6,
        patient_record_count=25,
        medquad_record_count=0,
        model_version="test-model",
        retrieval_version="test-embeddings",
    )
    streamlit.cache_resource.clear()

    try:
        with patch(
            "medical_assistant.application.build_assistant_runtime",
            return_value=runtime,
        ):
            app = AppTest.from_file(str(ROOT / "scripts" / "chatbot_web.py")).run(
                timeout=30
            )
            next(
                button
                for button in app.button
                if button.label == "Carregar assistente local"
            ).click().run(timeout=30)

            interview_input = app.chat_input[1]
            for answer in (
                "SIM",
                "adulto",
                "dor de cabeça",
                "PULAR",
                "PULAR",
                "PULAR",
                "PULAR",
                "SIM",
            ):
                interview_input.set_value(answer).run(timeout=30)
                interview_input = app.chat_input[1]

        assert not app.exception
        assert len(workflow.calls) == 1
        question, arguments = workflow.calls[0]
        assert "Queixa principal: dor de cabeça" in question
        assert arguments["operation"] is WorkflowOperation.REVIEW_SYNTHETIC_CASE
        assert "patient_id" not in arguments
        assert "access_context" not in arguments
    finally:
        streamlit.cache_resource.clear()
