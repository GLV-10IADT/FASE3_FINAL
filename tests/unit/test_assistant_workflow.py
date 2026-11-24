from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from medical_assistant.assistant import SourceCitation
from medical_assistant.audit import AuditActor, AuditEvent
from medical_assistant.patient_data import (
    AccessContext,
    PatientRole,
    SyntheticPatientStore,
)
from medical_assistant.retrieval import (
    DocumentIndex,
    SourceDocument,
    load_synthetic_documents,
)
from medical_assistant.safety import SafetyReason
from medical_assistant.workflows import (
    AssistantWorkflow,
    SimulatedAlert,
    RequestIntent,
    SourceDetail,
    WorkflowStatus,
    WorkflowOperation,
)


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
PATIENT_ID = "PATIENT-SYNTH-001"
ACTOR = AuditActor(user_id="demo-clinician", role="clinician")


class RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class RecordingSimulatedAlertSink:
    def __init__(self) -> None:
        self.alerts: list[SimulatedAlert] = []

    def write(self, alert: SimulatedAlert) -> None:
        self.alerts.append(alert)


class FixedTranslator:
    def __init__(self, translation: str) -> None:
        self.translation = translation
        self.inputs: list[str] = []

    def translate(self, text: str) -> str:
        self.inputs.append(text)
        return self.translation


class ContextRecordingChatModel(BaseChatModel):
    _last_prompt: str = PrivateAttr(default="")

    @property
    def _llm_type(self) -> str:
        return "context-recording-test-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._last_prompt = str(messages[-1].content)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="O exame sintético está pendente. [EXAM-SYNTH-001]"
                    )
                )
            ]
        )


@pytest.fixture
def audit_sink() -> RecordingAuditSink:
    return RecordingAuditSink()


@pytest.fixture
def workflow(audit_sink: RecordingAuditSink) -> AssistantWorkflow:
    return AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
    )


def test_workflow_answers_information_request_with_retrieved_citations(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run("recuperação citação fontes", actor=ACTOR)

    assert result.status is WorkflowStatus.COMPLETE
    assert result.intent is RequestIntent.KNOWLEDGE_QUERY
    assert result.response is not None
    assert result.response.sources[0].source_id == "DOC-SYNTH-001"
    assert "Trecho recuperado:" in result.response.answer


def test_workflow_includes_the_retrieved_faq_answer(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run(
        "Como identificar origem metadados de um trecho?",
        actor=ACTOR,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.response is not None
    assert "metadados source, id, versão e seção" in result.response.answer
    assert result.response.sources[0].source_id == "FAQ-SYNTH-001"


def test_langchain_rag_uses_retrieved_context_and_respects_safety_gates() -> None:
    audit_sink = RecordingAuditSink()
    chat_model = FakeListChatModel(
        responses=[
            "Resposta informativa.",
            "Resposta inesperada sem evidência.",
            "Resposta inesperada sem revisão.",
        ]
    )
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        chat_model=chat_model,
        retrieval_version="hybrid-e5-test",
        model_version="fake-chat-model-v1",
    )

    no_evidence = workflow.run("xylophytic zephyr nanoparticle", actor=ACTOR)
    review_required = workflow.run("Qual tratamento devo prescrever?", actor=ACTOR)
    answered = workflow.run(
        "Como identificar origem metadados de um trecho?",
        actor=ACTOR,
    )

    assert no_evidence.status is WorkflowStatus.INSUFFICIENT_EVIDENCE
    assert review_required.status is WorkflowStatus.REVIEW_REQUIRED
    assert chat_model.i == 1
    assert answered.status is WorkflowStatus.COMPLETE
    assert answered.response is not None
    assert "Resposta informativa." in answered.response.answer
    assert answered.response.answer.endswith("Fontes: [FAQ-SYNTH-001]")
    assert answered.response.review_status.value == "pending"
    assert audit_sink.events[-1].model_version == "fake-chat-model-v1"
    assert audit_sink.events[-1].prompt_version == "langchain-rag-v2"
    assert audit_sink.events[-1].retrieval_version == "hybrid-e5-test"
    assert audit_sink.events[-1].model_invoked is True
    assert audit_sink.events[0].model_invoked is False
    assert audit_sink.events[1].model_invoked is False


def test_synthetic_case_review_is_source_grounded_and_requires_human_review() -> None:
    audit_sink = RecordingAuditSink()
    chat_model = FakeListChatModel(
        responses=[
            "Possibilidades citadas nas fontes: migrânea [CASE-DOC-001]. "
            "Exames citados nas fontes: ressonância [CASE-DOC-001]."
        ]
    )
    index = DocumentIndex(
        [
            SourceDocument(
                id="case-review:001",
                text=(
                    "Faixa etária adulto; queixa cefaleia; sintomas náusea. "
                    "A literatura menciona migrânea e ressonância."
                ),
                source="synthetic-case-review",
                title="Contexto sintético de pesquisa",
                section="abstract_context",
                source_dataset="synthetic",
                source_id="CASE-DOC-001",
                synthetic=True,
            )
        ]
    )
    workflow = AssistantWorkflow(
        index,
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        minimum_relevance=0.2,
        chat_model=chat_model,
        model_version="fake-case-review-v1",
    )
    case_summary = "Faixa etária adulto; queixa cefaleia; sintomas náusea"

    result = workflow.run(
        case_summary,
        actor=ACTOR,
        operation=WorkflowOperation.REVIEW_SYNTHETIC_CASE,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.intent is RequestIntent.SYNTHETIC_CASE_REVIEW
    assert result.response is not None
    assert result.response.review_status.value == "pending"
    assert "[CASE-DOC-001]" in result.response.answer
    assert "não representam diagnóstico" in result.response.uncertainty
    assert result.model_invoked is True
    assert audit_sink.events[-1].review_required is True
    assert audit_sink.events[-1].prompt_version.endswith(
        "+synthetic-case-review-v2"
    )
    assert case_summary not in audit_sink.events[-1].to_json()


def test_urgent_synthetic_case_is_blocked_before_model_generation() -> None:
    chat_model = FakeListChatModel(
        responses=["Não deve ser gerada uma resposta clínica."]
    )
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
        chat_model=chat_model,
        model_version="fake-case-review-v1",
    )

    result = workflow.run(
        "Faixa etária adulto; queixa dor no peito",
        actor=ACTOR,
        operation=WorkflowOperation.REVIEW_SYNTHETIC_CASE,
    )

    assert result.status is WorkflowStatus.REVIEW_REQUIRED
    assert result.model_invoked is False
    assert chat_model.i == 0


def test_synthetic_case_review_is_restricted_to_clinician_role() -> None:
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
    )

    with pytest.raises(PermissionError, match="demo clinician role"):
        workflow.run(
            "Faixa etária adulto; queixa cefaleia",
            actor=AuditActor(user_id="demo-guest", role="guest"),
            operation=WorkflowOperation.REVIEW_SYNTHETIC_CASE,
        )


def test_portuguese_question_is_translated_for_english_evidence() -> None:
    audit_sink = RecordingAuditSink()
    chat_model = FakeListChatModel(
        responses=["The mitochondria play an important role in this process."]
    )
    document = SourceDocument(
        id="pubmedqa:21645374",
        text=(
            "Mitochondria play an early role in programmed cell death "
            "in lace plant leaves."
        ),
        source="https://pubmed.ncbi.nlm.nih.gov/21645374/",
        title="PubMedQA abstract context",
        section="abstract_context",
        source_dataset="pubmedqa",
        source_id="21645374",
    )
    workflow = AssistantWorkflow(
        DocumentIndex([document]),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        chat_model=chat_model,
        query_translator=FixedTranslator(
            "Do mitochondria play a role in programmed cell death in lace plant leaves?"
        ),
        answer_translator=FixedTranslator(
            "As mitocôndrias têm um papel importante nesse processo."
        ),
        translation_version="OPUS-MT-test-pt-en-en-pt",
        model_version="fake-chat-model-v1",
    )

    result = workflow.run(
        "Qual é o papel das mitocôndrias na morte celular programada da planta lace plant?",
        actor=ACTOR,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.response is not None
    assert "mitocôndrias" in result.response.answer
    assert result.response.sources[0].source_id == "21645374"
    assert result.model_invoked is True
    assert result.translation_used is True
    assert audit_sink.events[-1].model_invoked is True
    assert audit_sink.events[-1].translation_used is True
    assert audit_sink.events[-1].translation_version == "OPUS-MT-test-pt-en-en-pt"
    assert audit_sink.events[-1].prompt_version == "langchain-rag-pt-en-pt-v2"


def test_translated_search_keeps_original_language_fixture_matches() -> None:
    chat_model = FakeListChatModel(
        responses=["An answer based on the retrieved synthetic FAQ."]
    )
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
        chat_model=chat_model,
        query_translator=FixedTranslator(
            "What is the recommended procedure for an unrelated topic?"
        ),
        answer_translator=FixedTranslator("Resposta com fonte."),
        translation_version="OPUS-MT-test-pt-en-en-pt",
        model_version="fake-chat-model-v1",
    )

    result = workflow.run(
        "Como identificar origem metadados de um trecho?",
        actor=ACTOR,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.response is not None
    assert result.response.sources[0].source_id == "FAQ-SYNTH-001"


def test_safety_gate_runs_before_query_translation() -> None:
    chat_model = FakeListChatModel(responses=["A translation must not be generated."])
    audit_sink = RecordingAuditSink()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        chat_model=chat_model,
        query_translator=FixedTranslator(
            "What treatment should I prescribe?"
        ),
        translation_version="OPUS-MT-test-pt-en",
        model_version="fake-chat-model-v1",
    )

    result = workflow.run("Qual tratamento devo prescrever?", actor=ACTOR)

    assert result.status is WorkflowStatus.REVIEW_REQUIRED
    assert result.model_invoked is False
    assert result.translation_used is False
    assert chat_model.i == 0
    assert audit_sink.events[-1].model_invoked is False
    assert audit_sink.events[-1].translation_used is False


def test_audit_distinguishes_translation_from_llm_generation() -> None:
    audit_sink = RecordingAuditSink()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        chat_model=FakeListChatModel(responses=[]),
        query_translator=FixedTranslator("xylophytic zephyr nanoparticle"),
        translation_version="OPUS-MT-test-pt-en",
        model_version="fake-chat-model-v1",
    )

    result = workflow.run("Pergunta sem evidência", actor=ACTOR)

    assert result.status is WorkflowStatus.INSUFFICIENT_EVIDENCE
    assert result.model_invoked is False
    assert result.translation_used is True
    assert audit_sink.events[-1].model_invoked is False
    assert audit_sink.events[-1].translation_used is True
    assert audit_sink.events[-1].translation_version == "OPUS-MT-test-pt-en"


def test_workflow_abstains_when_no_evidence_is_found(workflow: AssistantWorkflow) -> None:
    result = workflow.run("xylophytic zephyr nanoparticle", actor=ACTOR)

    assert result.status is WorkflowStatus.INSUFFICIENT_EVIDENCE
    assert result.response is not None
    assert result.response.sources == ()


def test_workflow_returns_explicit_error_for_unauthorized_patient_access(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run(
        "consultar registro",
        actor=AuditActor(user_id="demo-guest", role="guest"),
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id="demo-guest",
            role=PatientRole.GUEST,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    assert result.status is WorkflowStatus.TOOL_ERROR
    assert result.error_code == "patient_access_denied"
    assert result.response is None
    assert result.patient_record is None


def test_workflow_returns_error_for_missing_patient_fixture(
    workflow: AssistantWorkflow,
) -> None:
    patient_id = "PATIENT-SYNTH-UNKNOWN"
    result = workflow.run(
        "consultar registro",
        actor=ACTOR,
        patient_id=patient_id,
        access_context=AccessContext(
            user_id="demo-clinician",
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({patient_id}),
        ),
    )

    assert result.status is WorkflowStatus.TOOL_ERROR
    assert result.error_code == "patient_not_found"
    assert result.response is None


def test_workflow_routes_clinical_action_requests_to_human_review(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run("Qual tratamento devo prescrever?", actor=ACTOR)

    assert result.status is WorkflowStatus.REVIEW_REQUIRED
    assert result.intent is RequestIntent.CLINICAL_ACTION
    assert result.response is not None
    assert result.response.review_status.value == "pending"
    assert result.patient_record is None
    assert result.review_reason is SafetyReason.CLINICAL_ACTION
    assert "encaminha alertas automaticamente" in result.response.answer


def test_workflow_routes_urgent_signals_to_review_without_claiming_auto_alert(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run("Estou com falta de ar.", actor=ACTOR)

    assert result.status is WorkflowStatus.REVIEW_REQUIRED
    assert result.review_reason is SafetyReason.URGENT_SIGNAL
    assert result.response is not None
    assert "envia alertas" in result.response.answer


def test_workflow_uses_patient_access_context_and_returns_source(
    workflow: AssistantWorkflow,
) -> None:
    result = workflow.run(
        "consultar registros",
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id="demo-clinician",
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.patient_record is not None
    assert result.patient_record.patient_id == PATIENT_ID
    assert result.response is not None
    assert result.response.review_status.value == "pending"
    assert result.response.sources[0].source_id == "RECORD-SYNTH-001"


def test_llm_receives_only_whitelisted_synthetic_patient_facts() -> None:
    chat_model = ContextRecordingChatModel()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
        chat_model=chat_model,
        model_version="test-model+synthetic-adapter",
    )

    result = workflow.run(
        "Qual é o status do exame?",
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id=ACTOR.user_id,
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    assert result.response is not None
    assert "EXAM-SYNTH-001" in chat_model._last_prompt
    assert "status do exame: pending" in chat_model._last_prompt
    assert "PATIENT-SYNTH-001" not in chat_model._last_prompt
    assert result.response.review_status.value == "pending"
    assert {source.source_id for source in result.response.sources} >= {
        "RECORD-SYNTH-001",
        "EXAM-SYNTH-001",
    }


def test_pending_exam_node_prepares_only_a_local_reviewable_alert() -> None:
    audit_sink = RecordingAuditSink()
    alert_sink = RecordingSimulatedAlertSink()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
        simulated_alert_sink=alert_sink,
    )
    access_context = AccessContext(
        user_id=ACTOR.user_id,
        role=PatientRole.CLINICIAN,
        authorized_patient_ids=frozenset({PATIENT_ID}),
    )

    result = workflow.run(
        "Verifique exames pendentes",
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=access_context,
        operation=WorkflowOperation.CHECK_PENDING_EXAMS,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.intent is RequestIntent.EXAM_STATUS_CHECK
    assert result.simulated_alert is not None
    assert result.simulated_alert.status == "awaiting_human_review"
    assert result.simulated_alert.source_ids == ("EXAM-SYNTH-001",)
    assert len(alert_sink.alerts) == 1
    assert result.response is not None
    assert "nenhum sistema hospitalar foi consultado" in result.response.uncertainty
    assert audit_sink.events[-1].action_refs == (
        "pending_exam:awaiting_human_review",
    )
    assert audit_sink.events[-1].review_required is True
    assert PATIENT_ID not in audit_sink.events[-1].to_json()


def test_exam_check_without_pending_items_does_not_create_alert() -> None:
    alert_sink = RecordingSimulatedAlertSink()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
        simulated_alert_sink=alert_sink,
    )
    patient_id = "PATIENT-SYNTH-002"
    access_context = AccessContext(
        user_id=ACTOR.user_id,
        role=PatientRole.CLINICIAN,
        authorized_patient_ids=frozenset({patient_id}),
    )

    result = workflow.run(
        "Verifique exames pendentes",
        actor=ACTOR,
        patient_id=patient_id,
        access_context=access_context,
        operation=WorkflowOperation.CHECK_PENDING_EXAMS,
    )

    assert result.status is WorkflowStatus.COMPLETE
    assert result.simulated_alert is None
    assert alert_sink.alerts == []
    assert result.response is not None
    assert result.response.review_status.value == "not_required"


def test_exam_check_fails_explicitly_without_simulated_outbox() -> None:
    audit_sink = RecordingAuditSink()
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        audit_sink,
    )
    access_context = AccessContext(
        user_id=ACTOR.user_id,
        role=PatientRole.CLINICIAN,
        authorized_patient_ids=frozenset({PATIENT_ID}),
    )

    result = workflow.run(
        "Verifique exames pendentes",
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=access_context,
        operation=WorkflowOperation.CHECK_PENDING_EXAMS,
    )

    assert result.status is WorkflowStatus.TOOL_ERROR
    assert result.error_code == "simulated_alert_sink_not_configured"
    assert audit_sink.events[-1].patient_data_accessed is True


def test_workflow_audit_contains_trace_metadata_without_question_or_patient_id(
    workflow: AssistantWorkflow,
    audit_sink: RecordingAuditSink,
) -> None:
    sensitive_query = "private query that must not be stored"
    result = workflow.run(
        sensitive_query,
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id=ACTOR.user_id,
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    assert len(audit_sink.events) == 1
    event = audit_sink.events[0]
    serialized = event.to_json()
    assert event.correlation_id == result.correlation_id
    assert event.actor_id == ACTOR.user_id
    assert event.actor_role == ACTOR.role
    assert event.patient_data_accessed is True
    assert event.source_refs == ("synthetic_patient_fixture",)
    assert event.model_version == "deterministic-baseline-2"
    assert event.prompt_version == "none"
    assert sensitive_query not in serialized
    assert PATIENT_ID not in serialized
    assert "RECORD-SYNTH-001" not in serialized


def test_workflow_audits_denied_access_without_patient_identifiers(
    workflow: AssistantWorkflow,
    audit_sink: RecordingAuditSink,
) -> None:
    result = workflow.run(
        "consultar registro",
        actor=AuditActor(user_id="demo-guest", role="guest"),
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id="demo-guest",
            role=PatientRole.GUEST,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    event = audit_sink.events[-1]
    assert result.error_code == "patient_access_denied"
    assert event.status == WorkflowStatus.TOOL_ERROR.value
    assert event.error_code == "patient_access_denied"
    assert event.patient_data_accessed is False
    assert PATIENT_ID not in event.to_json()


def test_patient_access_context_must_match_audit_actor(
    workflow: AssistantWorkflow,
    audit_sink: RecordingAuditSink,
) -> None:
    result = workflow.run(
        "consultar registro",
        actor=ACTOR,
        patient_id=PATIENT_ID,
        access_context=AccessContext(
            user_id="different-user",
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({PATIENT_ID}),
        ),
    )

    assert result.error_code == "actor_context_mismatch"
    assert result.patient_record is None
    assert audit_sink.events[-1].error_code == "actor_context_mismatch"


def test_audit_write_failure_is_not_hidden() -> None:
    class FailingAuditSink:
        def write(self, event: AuditEvent) -> None:
            assert event.status == WorkflowStatus.COMPLETE.value
            raise OSError("audit storage unavailable")

    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        FailingAuditSink(),
    )

    with pytest.raises(OSError, match="audit storage unavailable"):
        workflow.run("recuperação citação fontes", actor=ACTOR)


def test_workflow_rejects_empty_question(workflow: AssistantWorkflow) -> None:
    with pytest.raises(ValueError, match="question"):
        workflow.run(" ", actor=ACTOR)


def test_workflow_exposes_retrieval_scores_for_ui_source_cards() -> None:
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR)),
        SyntheticPatientStore(SYNTHETIC_DIR / "patients.json"),
        RecordingAuditSink(),
        minimum_relevance=0.1,
    )

    result = workflow.run(
        "Como identificar origem metadados de um trecho?",
        actor=ACTOR,
    )

    assert result.source_details
    assert result.source_details[0].citation.source_id == "FAQ-SYNTH-001"
    assert result.source_details[0].excerpt
    assert result.source_details[0].lexical_score is not None
    assert result.source_details[0].semantic_score is None


def test_patient_source_detail_cannot_include_record_content() -> None:
    with pytest.raises(ValueError, match="must not expose content or scores"):
        SourceDetail(
            citation=SourceCitation(
                source="local:authorized_patient_record",
                title="CASE-001",
                section="history",
                source_id="REC-001",
            ),
            excerpt="Sensitive clinical content",
            patient_record=True,
        )
