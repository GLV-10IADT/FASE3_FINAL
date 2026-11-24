from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from medical_assistant.audit import AuditActor, AuditEvent
from medical_assistant.chatbot import (
    _collect_synthetic_case_summary,
    run_chat_session,
)
from medical_assistant.patient_data import SyntheticPatientStore, SyntheticStaffStore
from medical_assistant.retrieval import (
    DocumentIndex,
    SourceDocument,
    load_synthetic_documents,
)
from medical_assistant.workflows import AssistantWorkflow, JsonlSimulatedAlertSink


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"


class RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


def test_synthetic_case_interview_collects_summary_only_after_confirmation() -> None:
    prompts = iter(
        [
            "sim",
            "adulto",
            "cefaleia",
            "náusea",
            "há dois dias",
            "",
            "",
            "sim",
        ]
    )
    output: list[str] = []

    summary = _collect_synthetic_case_summary(
        lambda: next(prompts),
        output.append,
    )

    assert summary is not None
    assert "Faixa etária: adulto" in summary.summary
    assert "Queixa principal: cefaleia" in summary.summary
    assert "Sintomas associados: náusea" in summary.summary
    assert "Exames já realizados" not in summary.summary
    assert summary.retrieval_query == "cefaleia náusea há dois dias"
    assert any("não é gravado na auditoria" in line for line in output)


def test_chatbot_runs_guided_case_review_and_displays_sources() -> None:
    audit_sink = RecordingAuditSink()
    patient_store = SyntheticPatientStore(SYNTHETIC_DIR / "patients.json")
    staff_member = SyntheticStaffStore(SYNTHETIC_DIR / "staff.json").read_staff(
        "demo-clinician"
    )
    document = SourceDocument(
        id="case-review:cli",
        text=(
            "Cefaleia, náusea e evolução há dois dias são descritas em literatura "
            "que menciona migrânea e ressonância."
        ),
        source="synthetic-case-review",
        title="Contexto sintético",
        section="abstract_context",
        source_dataset="synthetic",
        source_id="CASE-CLI-001",
        synthetic=True,
    )
    workflow = AssistantWorkflow(
        DocumentIndex([document]),
        patient_store,
        audit_sink,
        minimum_relevance=0.2,
        chat_model=FakeListChatModel(
            responses=[
                "Possibilidades citadas nas fontes: migrânea [CASE-CLI-001]. "
                "Exames citados nas fontes: ressonância [CASE-CLI-001]."
            ]
        ),
        model_version="fake-case-review-v1",
    )
    prompts = iter(
        [
            "/atendimento",
            "sim",
            "adulto",
            "cefaleia",
            "náusea",
            "há dois dias",
            "",
            "",
            "sim",
            "/sair",
        ]
    )
    output: list[str] = []

    run_chat_session(
        workflow,
        AuditActor(user_id=staff_member.user_id, role=staff_member.role.value),
        staff_member,
        patient_store,
        read_line=lambda: next(prompts),
        write_line=output.append,
    )

    assert any("Possibilidades citadas nas fontes: migrânea" in line for line in output)
    assert any("Exames citados nas fontes: ressonância" in line for line in output)
    assert any("Revisão humana: pending" in line for line in output)
    assert any("CASE-CLI-001" in line for line in output)
    assert audit_sink.events[0].intent == "synthetic_case_review"
    assert audit_sink.events[0].review_required is True
    assert "cefaleia" not in audit_sink.events[0].to_json()


def test_chatbot_routes_chat_and_exam_command_through_workflow(
    tmp_path: Path,
) -> None:
    audit_sink = RecordingAuditSink()
    patient_store = SyntheticPatientStore(SYNTHETIC_DIR / "patients.json")
    staff_member = SyntheticStaffStore(SYNTHETIC_DIR / "staff.json").read_staff(
        "demo-clinician"
    )
    chat_model = FakeListChatModel(responses=["O exame está pendente."])
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        patient_store,
        audit_sink,
        chat_model=chat_model,
        model_version="fake-model+synthetic-adapter",
        simulated_alert_sink=JsonlSimulatedAlertSink(tmp_path / "alerts.jsonl"),
    )
    prompts = iter(
        [
            "/ajuda",
            "/paciente PATIENT-SYNTH-001",
            "Qual é o status do exame?",
            "/exames",
            "/paciente PATIENT-REAL-001",
            "/sair",
        ]
    )
    output: list[str] = []

    run_chat_session(
        workflow,
        AuditActor(user_id=staff_member.user_id, role=staff_member.role.value),
        staff_member,
        patient_store,
        read_line=lambda: next(prompts),
        write_line=output.append,
    )

    assert any("Modo local de operador único" in line for line in output)
    assert any("/paciente <ID>" in line for line in output)
    assert any("/prontuarios" in line for line in output)
    assert any("Não trate respostas como decisão clínica." in line for line in output)
    assert any("O exame está pendente." in line for line in output)
    assert any("EXAM-SYNTH-001" in line for line in output)
    assert any("Sessão encerrada." in line for line in output)
    assert all("PATIENT-REAL-001" not in line for line in output)
    assert any("não autorizado" in line for line in output)
    assert len(audit_sink.events) == 2
    assert audit_sink.events[0].model_invoked is True
    assert audit_sink.events[1].model_invoked is False
    assert all("Qual é o status do exame?" not in event.to_json() for event in audit_sink.events)
