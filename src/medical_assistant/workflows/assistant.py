from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from time import perf_counter
from typing import NotRequired, TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from medical_assistant.assistant.response import (
    AssistantResponse,
    ReviewStatus,
    SourceCitation,
    validate_source_citations,
)
from medical_assistant.assistant.langchain_rag import (
    DocumentIndexRetriever,
    LangChainRagAnswerer,
)
from medical_assistant.audit import AuditActor, AuditEvent, AuditSink
from medical_assistant.patient_data import (
    AccessContext,
    AccessDeniedError,
    ExamStatus,
    PatientNotFoundError,
    PatientRecord,
    PatientStore,
)
from medical_assistant.retrieval import DocumentIndex
from medical_assistant.safety import SafetyPolicy, SafetyReason
from medical_assistant.translation import TextTranslator
from medical_assistant.workflows.actions import SimulatedAlert, SimulatedAlertSink


class RequestIntent(StrEnum):
    KNOWLEDGE_QUERY = "knowledge_query"
    PATIENT_QUERY = "patient_query"
    CLINICAL_ACTION = "clinical_action"
    EXAM_STATUS_CHECK = "exam_status_check"
    SYNTHETIC_CASE_REVIEW = "synthetic_case_review"


class WorkflowOperation(StrEnum):
    QUERY = "query"
    CHECK_PENDING_EXAMS = "check_pending_exams"
    REVIEW_SYNTHETIC_CASE = "review_synthetic_case"


class WorkflowStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REVIEW_REQUIRED = "review_required"
    TOOL_ERROR = "tool_error"


@dataclass(frozen=True)
class SourceDetail:
    citation: SourceCitation
    excerpt: str | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    patient_record: bool = False

    def __post_init__(self) -> None:
        if self.patient_record and (
            self.excerpt is not None
            or self.lexical_score is not None
            or self.semantic_score is not None
        ):
            raise ValueError("Patient source details must not expose content or scores")


@dataclass(frozen=True)
class WorkflowResult:
    status: WorkflowStatus
    intent: RequestIntent
    response: AssistantResponse | None
    source_details: tuple[SourceDetail, ...] = ()
    patient_record: PatientRecord | None = None
    error_code: str | None = None
    review_reason: SafetyReason | None = None
    correlation_id: str | None = None
    simulated_alert: SimulatedAlert | None = None
    model_invoked: bool = False
    translation_used: bool = False


class WorkflowState(TypedDict):
    question: str
    retrieval_query: NotRequired[str | None]
    intent: NotRequired[RequestIntent]
    patient_id: NotRequired[str | None]
    access_context: NotRequired[AccessContext | None]
    patient_record: NotRequired[PatientRecord | None]
    operation: NotRequired[WorkflowOperation]
    translation_used: NotRequired[bool]
    pending_exam_ids: NotRequired[tuple[str, ...]]
    simulated_alert: NotRequired[SimulatedAlert | None]
    retrieved_documents: NotRequired[tuple[Document, ...]]
    error_code: NotRequired[str | None]
    review_reason: NotRequired[SafetyReason | None]
    actor: NotRequired[AuditActor]
    correlation_id: NotRequired[str]
    result: NotRequired[WorkflowResult]


def _citation_from_document(document: Document) -> SourceCitation:
    metadata = document.metadata
    fields = ("source", "title", "section", "source_id")
    if any(not isinstance(metadata.get(field), str) for field in fields):
        raise ValueError("Retrieved LangChain documents must include source metadata")
    return SourceCitation(
        source=metadata["source"],
        title=metadata["title"],
        section=metadata["section"],
        source_id=metadata["source_id"],
    )


def _citations_from_patient_record(record: PatientRecord) -> tuple[SourceCitation, ...]:
    source = record.source
    citations = [
        SourceCitation(
            source=source,
            title=record.display_label,
            section=item.record_type,
            source_id=item.record_id,
        )
        for item in record.records
    ]
    for encounter in record.encounters:
        citations.append(
            SourceCitation(
                source=source,
                title=record.display_label,
                section=f"encounter:{encounter.status}",
                source_id=encounter.encounter_id,
            )
        )
        citations.extend(
            SourceCitation(
                source=source,
                title=record.display_label,
                section=f"exam:{exam.status.value}",
                source_id=exam.exam_id,
            )
            for exam in encounter.exams
        )
    return tuple(citations)


def _documents_from_patient_record(record: PatientRecord) -> tuple[Document, ...]:
    source = record.source
    documents = [
        Document(
            page_content=item.content,
            metadata={
                "source": source,
                "title": record.display_label,
                "section": item.record_type,
                "source_id": item.record_id,
            },
        )
        for item in record.records
    ]
    for encounter in record.encounters:
        documents.append(
            Document(
                page_content=f"Estado do atendimento: {encounter.status}.",
                metadata={
                    "source": source,
                    "title": record.display_label,
                    "section": "encounter",
                    "source_id": encounter.encounter_id,
                },
            )
        )
        documents.extend(
            Document(
                page_content=(
                    f"{exam.display_name}; status do exame: {exam.status.value}."
                ),
                metadata={
                    "source": source,
                    "title": record.display_label,
                    "section": "exam",
                    "source_id": exam.exam_id,
                },
            )
            for exam in encounter.exams
        )
    return tuple(documents)


class AssistantWorkflow:
    """Safety-gated LangGraph workflow over local sources and patient records."""

    def __init__(
        self,
        document_index: DocumentIndex,
        patient_store: PatientStore,
        audit_sink: AuditSink,
        *,
        retrieval_limit: int = 5,
        minimum_relevance: float = 0.5,
        chat_model: BaseChatModel | None = None,
        query_translator: TextTranslator | None = None,
        answer_translator: TextTranslator | None = None,
        translation_version: str | None = None,
        retrieval_version: str | None = None,
        model_version: str | None = None,
        simulated_alert_sink: SimulatedAlertSink | None = None,
    ) -> None:
        if retrieval_limit < 1:
            raise ValueError("retrieval_limit must be greater than zero")
        if not 0 <= minimum_relevance <= 1:
            raise ValueError("minimum_relevance must be between zero and one")
        if chat_model is not None and (
            not isinstance(model_version, str) or not model_version.strip()
        ):
            raise ValueError("model_version must be supplied when chat_model is enabled")
        if chat_model is None and model_version is not None:
            raise ValueError("model_version cannot be supplied without chat_model")
        if query_translator is not None and chat_model is None:
            raise ValueError("query_translator requires chat_model")
        if answer_translator is not None and chat_model is None:
            raise ValueError("answer_translator requires chat_model")
        if (query_translator is not None or answer_translator is not None) and (
            not isinstance(translation_version, str) or not translation_version.strip()
        ):
            raise ValueError("translation_version is required when translators are enabled")
        if (
            query_translator is None
            and answer_translator is None
            and translation_version is not None
        ):
            raise ValueError("translation_version requires at least one translator")
        self._retriever = DocumentIndexRetriever(
            document_index=document_index,
            limit=retrieval_limit,
            minimum_relevance=minimum_relevance,
        )
        self._answerer = (
            LangChainRagAnswerer(
                chat_model,
                answer_translator=answer_translator,
            )
            if chat_model
            else None
        )
        self._query_translator = query_translator
        self._answer_translator = answer_translator
        self._translation_version = translation_version
        if retrieval_version is not None and not retrieval_version.strip():
            raise ValueError("retrieval_version must be a non-empty string when supplied")
        self._retrieval_version = retrieval_version
        self._model_version = model_version or "deterministic-baseline-2"
        self._prompt_version = (
            "langchain-rag-pt-en-pt-v2"
            if query_translator is not None and answer_translator is not None
            else "langchain-rag-pt-en-v2"
            if query_translator is not None
            else "langchain-rag-en-pt-v2"
            if answer_translator is not None
            else "langchain-rag-v2"
            if chat_model
            else "none"
        )
        self._patient_store = patient_store
        self._audit_sink = audit_sink
        self._simulated_alert_sink = simulated_alert_sink
        self._safety_policy = SafetyPolicy()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("classify", self._classify)
        graph.add_node("review", self._require_review)
        graph.add_node("read_patient", self._read_patient)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("compose", self._compose)
        graph.add_node("check_pending_exams", self._check_pending_exams)
        graph.add_node("prepare_simulated_alert", self._prepare_simulated_alert)
        graph.add_node("compose_exam_status", self._compose_exam_status)
        graph.add_node("tool_error", self._tool_error)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "review": "review",
                "read_patient": "read_patient",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("review", END)
        graph.add_conditional_edges(
            "read_patient",
            self._route_after_patient_read,
            {
                "tool_error": "tool_error",
                "retrieve": "retrieve",
                "check_pending_exams": "check_pending_exams",
            },
        )
        graph.add_conditional_edges(
            "check_pending_exams",
            self._route_after_exam_check,
            {
                "prepare_simulated_alert": "prepare_simulated_alert",
                "compose_exam_status": "compose_exam_status",
            },
        )
        graph.add_conditional_edges(
            "prepare_simulated_alert",
            self._route_after_alert_preparation,
            {"tool_error": "tool_error", "compose_exam_status": "compose_exam_status"},
        )
        graph.add_edge("retrieve", "compose")
        graph.add_edge("compose_exam_status", END)
        graph.add_edge("compose", END)
        graph.add_edge("tool_error", END)
        return graph.compile()

    def _classify(self, state: WorkflowState) -> dict[str, object]:
        assessment = self._safety_policy.assess(state["question"])
        if assessment.requires_human_review:
            intent = RequestIntent.CLINICAL_ACTION
        elif (
            state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.CHECK_PENDING_EXAMS
        ):
            intent = RequestIntent.EXAM_STATUS_CHECK
        elif (
            state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.REVIEW_SYNTHETIC_CASE
        ):
            intent = RequestIntent.SYNTHETIC_CASE_REVIEW
        elif state.get("patient_id"):
            intent = RequestIntent.PATIENT_QUERY
        else:
            intent = RequestIntent.KNOWLEDGE_QUERY
        return {"intent": intent, "review_reason": assessment.reason}

    @staticmethod
    def _route_after_classification(state: WorkflowState) -> str:
        if state["intent"] is RequestIntent.CLINICAL_ACTION:
            return "review"
        if (
            state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.REVIEW_SYNTHETIC_CASE
        ):
            return "retrieve"
        if (
            state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.CHECK_PENDING_EXAMS
        ):
            return "read_patient"
        if state.get("patient_id"):
            return "read_patient"
        return "retrieve"

    def _require_review(self, state: WorkflowState) -> dict[str, WorkflowResult]:
        review_reason = state.get("review_reason")
        if review_reason is SafetyReason.URGENT_SIGNAL:
            answer = (
                "Foi detectado um possível sinal de urgência. O protótipo não presta "
                "atendimento nem envia alertas; acione a equipe responsável conforme "
                "o protocolo local."
            )
            uncertainty = "A identificação é baseada em palavras-chave e não é triagem clínica."
        else:
            answer = (
                "Esta solicitação exige revisão profissional antes de qualquer conduta. "
                "O protótipo não prescreve, executa ordens nem encaminha alertas automaticamente."
            )
            uncertainty = "A classificação é heurística; é necessária avaliação humana."
        response = AssistantResponse(
            answer=answer,
            sources=(),
            uncertainty=uncertainty,
            review_status=ReviewStatus.PENDING,
        )
        return {
            "result": WorkflowResult(
                status=WorkflowStatus.REVIEW_REQUIRED,
                intent=state["intent"],
                response=response,
                review_reason=review_reason,
            )
        }

    def _read_patient(self, state: WorkflowState) -> dict[str, object]:
        patient_id = state.get("patient_id")
        access_context = state.get("access_context")
        actor = state.get("actor")
        if not patient_id or not isinstance(access_context, AccessContext):
            return {"error_code": "patient_access_denied"}
        if (
            not isinstance(actor, AuditActor)
            or access_context.user_id != actor.user_id
            or access_context.role.value != actor.role
        ):
            return {"error_code": "actor_context_mismatch"}

        try:
            patient_record = self._patient_store.read_patient(patient_id, access_context)
        except AccessDeniedError:
            return {"error_code": "patient_access_denied"}
        except PatientNotFoundError:
            return {"error_code": "patient_not_found"}
        return {"patient_record": patient_record, "error_code": None}

    @staticmethod
    def _route_after_patient_read(state: WorkflowState) -> str:
        if state.get("error_code"):
            return "tool_error"
        if (
            state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.CHECK_PENDING_EXAMS
        ):
            return "check_pending_exams"
        return "retrieve"

    @staticmethod
    def _check_pending_exams(state: WorkflowState) -> dict[str, tuple[str, ...]]:
        patient_record = state.get("patient_record")
        if not isinstance(patient_record, PatientRecord):
            return {"pending_exam_ids": ()}
        pending_exam_ids = tuple(
            exam.exam_id
            for encounter in patient_record.encounters
            for exam in encounter.exams
            if exam.status is ExamStatus.PENDING
        )
        return {"pending_exam_ids": pending_exam_ids}

    @staticmethod
    def _route_after_exam_check(state: WorkflowState) -> str:
        return (
            "prepare_simulated_alert"
            if state.get("pending_exam_ids")
            else "compose_exam_status"
        )

    def _prepare_simulated_alert(self, state: WorkflowState) -> dict[str, object]:
        if self._simulated_alert_sink is None:
            return {"error_code": "simulated_alert_sink_not_configured"}
        alert = SimulatedAlert(
            correlation_id=state["correlation_id"],
            action_type="pending_exam",
            status="awaiting_human_review",
            source_ids=state["pending_exam_ids"],
        )
        self._simulated_alert_sink.write(alert)
        return {"simulated_alert": alert, "error_code": None}

    @staticmethod
    def _route_after_alert_preparation(state: WorkflowState) -> str:
        return "tool_error" if state.get("error_code") else "compose_exam_status"

    @staticmethod
    def _compose_exam_status(state: WorkflowState) -> dict[str, WorkflowResult]:
        patient_record = state.get("patient_record")
        if not isinstance(patient_record, PatientRecord):
            return AssistantWorkflow._tool_error(
                {
                    "intent": state["intent"],
                    "error_code": "patient_not_found",
                }
            )
        exam_lines = [
            f"{exam.display_name}: {exam.status.value}"
            for encounter in patient_record.encounters
            for exam in encounter.exams
        ]
        if exam_lines:
            answer = "Estado dos exames do registro selecionado: " + "; ".join(exam_lines) + "."
        else:
            answer = "O registro selecionado não contém exames para este atendimento."
        simulated_alert = state.get("simulated_alert")
        if simulated_alert is not None:
            answer += " Alerta simulado preparado e aguardando revisão humana."
        response = AssistantResponse(
            answer=answer,
            sources=_citations_from_patient_record(patient_record),
            uncertainty=(
                "Estado extraído do registro local selecionado; "
                "nenhum sistema hospitalar foi consultado."
            ),
            review_status=(
                ReviewStatus.PENDING
                if simulated_alert is not None
                else ReviewStatus.NOT_REQUIRED
            ),
        )
        source_details = tuple(
            SourceDetail(citation=citation, patient_record=True)
            for citation in response.sources
        )
        return {
            "result": WorkflowResult(
                status=WorkflowStatus.COMPLETE,
                intent=RequestIntent.EXAM_STATUS_CHECK,
                response=response,
                source_details=source_details,
                patient_record=patient_record,
                simulated_alert=simulated_alert,
            )
        }

    def _retrieve(self, state: WorkflowState) -> dict[str, object]:
        question = state.get("retrieval_query") or state["question"]
        translated_question = (
            self._query_translator.translate(question)
            if self._query_translator is not None
            else None
        )
        results = self._retriever.search_results(
            question,
            query_variants=(
                (translated_question,) if translated_question is not None else ()
            ),
        )
        return {
            "retrieved_documents": self._retriever.documents_from_results(results),
            "translation_used": translated_question is not None,
        }

    def _compose(self, self_state: WorkflowState) -> dict[str, WorkflowResult]:
        retrieved = self_state.get("retrieved_documents", ())
        patient_record = self_state.get("patient_record")
        case_review = (
            self_state.get("operation", WorkflowOperation.QUERY)
            is WorkflowOperation.REVIEW_SYNTHETIC_CASE
        )
        model_invoked = False
        translation_used = self_state.get("translation_used", False)
        patient_documents = (
            _documents_from_patient_record(patient_record)
            if isinstance(patient_record, PatientRecord)
            else ()
        )
        citations = tuple(_citation_from_document(document) for document in retrieved)
        if isinstance(patient_record, PatientRecord):
            citations += _citations_from_patient_record(patient_record)
        source_details = tuple(
            SourceDetail(
                citation=_citation_from_document(document),
                excerpt=document.page_content[:500],
                lexical_score=(
                    document.metadata.get("lexical_score")
                    if isinstance(document.metadata.get("lexical_score"), float)
                    else None
                ),
                semantic_score=(
                    document.metadata.get("semantic_score")
                    if isinstance(document.metadata.get("semantic_score"), float)
                    else None
                ),
            )
            for document in retrieved
        )
        if isinstance(patient_record, PatientRecord):
            source_details += tuple(
                SourceDetail(citation=_citation_from_document(document), patient_record=True)
                for document in patient_documents
            )

        if not citations:
            response = AssistantResponse(
                answer="Não encontrei fontes suficientes para responder à solicitação.",
                sources=(),
                uncertainty=(
                    "Não há evidência correspondente no índice consultado; "
                    "não gere hipóteses ou exames."
                    if case_review
                    else "Não há evidência correspondente no índice consultado."
                ),
                review_status=(
                    ReviewStatus.PENDING
                    if case_review
                    else ReviewStatus.NOT_REQUIRED
                ),
            )
            status = WorkflowStatus.INSUFFICIENT_EVIDENCE
        else:
            if self._answerer is not None and (retrieved or patient_documents):
                answer = (
                    self._answerer.answer_case_review(
                        self_state["question"],
                        retrieved,
                    )
                    if case_review
                    else self._answerer.answer(
                        self_state["question"],
                        (*retrieved, *patient_documents),
                    )
                )
                model_invoked = True
                translation_used = (
                    translation_used or self._answer_translator is not None
                )
                uncertainty = (
                    "Possibilidades e exames apenas citados nas fontes recuperadas; "
                    "não representam diagnóstico ou solicitação clínica. Exigem "
                    "revisão profissional e conferência das fontes originais."
                    if case_review
                    else "Resposta gerada com contexto recuperado; requer revisão "
                    "profissional e conferência das fontes originais."
                )
                review_status = ReviewStatus.PENDING
            else:
                answer = (
                    "A revisão de possibilidades exige a LLM local, que não está "
                    "configurada."
                    if case_review
                    else f"Trecho recuperado: {retrieved[0].page_content}"
                    if retrieved
                    else "Foram encontrados registros do paciente; revise as fontes citadas."
                )
                uncertainty = (
                    "A LLM local não gerou hipóteses; revise as fontes com um "
                    "profissional."
                    if case_review
                    else "O trecho foi reproduzido da fonte citada. A busca lexical não "
                    "avalia sua correção clínica; confira o documento original e "
                    "submeta qualquer conduta à revisão profissional."
                )
                review_status = (
                    ReviewStatus.PENDING
                    if patient_record is not None or case_review
                    else ReviewStatus.NOT_REQUIRED
                )
            response = AssistantResponse(
                answer=answer,
                sources=citations,
                uncertainty=uncertainty,
                review_status=review_status,
            )
            status = (
                WorkflowStatus.TOOL_ERROR
                if case_review and self._answerer is None
                else WorkflowStatus.COMPLETE
            )

        available_sources = set(citations)
        validate_source_citations(response, available_sources)
        return {
            "result": WorkflowResult(
                status=status,
                intent=self_state["intent"],
                response=response,
                source_details=source_details,
                patient_record=patient_record,
                error_code=(
                    "case_review_requires_local_llm"
                    if case_review and self._answerer is None
                    else None
                ),
                model_invoked=model_invoked,
                translation_used=translation_used,
            )
        }

    @staticmethod
    def _tool_error(state: WorkflowState) -> dict[str, WorkflowResult]:
        return {
            "result": WorkflowResult(
                status=WorkflowStatus.TOOL_ERROR,
                intent=state["intent"],
                response=None,
                patient_record=state.get("patient_record"),
                error_code=state.get("error_code") or "patient_data_error",
            )
        }

    def run(
        self,
        question: str,
        *,
        actor: AuditActor,
        patient_id: str | None = None,
        access_context: AccessContext | None = None,
        operation: WorkflowOperation = WorkflowOperation.QUERY,
        retrieval_query: str | None = None,
    ) -> WorkflowResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if retrieval_query is not None and (
            not isinstance(retrieval_query, str) or not retrieval_query.strip()
        ):
            raise ValueError("retrieval_query must be a non-empty string when provided")
        if patient_id is not None and (
            not isinstance(patient_id, str) or not patient_id.strip()
        ):
            raise ValueError("patient_id must be a non-empty string when provided")
        if not isinstance(actor, AuditActor):
            raise ValueError("actor must be derived from an authenticated identity")
        if not isinstance(operation, WorkflowOperation):
            raise ValueError("operation must be a WorkflowOperation value")
        if (
            retrieval_query is not None
            and operation is not WorkflowOperation.REVIEW_SYNTHETIC_CASE
        ):
            raise ValueError("retrieval_query is only valid for synthetic case review")
        if (
            operation is WorkflowOperation.REVIEW_SYNTHETIC_CASE
            and actor.role != "clinician"
        ):
            raise PermissionError(
                "Synthetic case review is restricted to the demo clinician role"
            )
        correlation_id = str(uuid4())
        started_at = perf_counter()
        state = self._graph.invoke(
            {
                "question": question.strip(),
                "retrieval_query": (
                    retrieval_query.strip() if retrieval_query is not None else None
                ),
                "patient_id": patient_id,
                "access_context": access_context,
                "operation": operation,
                "actor": actor,
                "correlation_id": correlation_id,
            }
        )
        result = state["result"]
        if not isinstance(result, WorkflowResult):
            raise RuntimeError("LangGraph completed without a workflow result")
        result = WorkflowResult(
            status=result.status,
            intent=result.intent,
            response=result.response,
            source_details=result.source_details,
            patient_record=result.patient_record,
            error_code=result.error_code,
            review_reason=result.review_reason,
            correlation_id=correlation_id,
            simulated_alert=result.simulated_alert,
            model_invoked=result.model_invoked,
            translation_used=result.translation_used,
        )
        self._audit_sink.write(
            self._create_audit_event(
                result=result,
                actor=actor,
                patient_data_accessed=result.patient_record is not None,
                duration_ms=int((perf_counter() - started_at) * 1000),
            )
        )
        return result

    def _create_audit_event(
        self,
        *,
        result: WorkflowResult,
        actor: AuditActor,
        patient_data_accessed: bool,
        duration_ms: int,
    ) -> AuditEvent:
        source_refs: set[str] = set()
        if result.response is not None:
            for citation in result.response.sources:
                if citation.source.startswith("data/synthetic/patients.json"):
                    continue
                source_refs.add(citation.source)
        if (
            patient_data_accessed
            and result.patient_record is not None
            and result.patient_record.source.startswith(
                "data/synthetic/patients.json"
            )
        ):
            source_refs.add("synthetic_patient_fixture")
        action_refs = (
            (
                f"{result.simulated_alert.action_type}:"
                f"{result.simulated_alert.status}",
            )
            if result.simulated_alert is not None
            else ()
        )

        return AuditEvent(
            correlation_id=result.correlation_id or "",
            occurred_at=datetime.now(timezone.utc).isoformat(),
            actor_id=actor.user_id,
            actor_role=actor.role,
            intent=result.intent.value,
            status=result.status.value,
            review_required=(
                result.status is WorkflowStatus.REVIEW_REQUIRED
                or result.simulated_alert is not None
                or (
                    result.response is not None
                    and result.response.review_status is ReviewStatus.PENDING
                )
            ),
            review_reason=result.review_reason.value if result.review_reason else None,
            source_refs=tuple(sorted(source_refs)),
            patient_data_accessed=patient_data_accessed,
            model_version=self._model_version,
            prompt_version=(
                f"{self._prompt_version}+synthetic-case-review-v2"
                if result.intent is RequestIntent.SYNTHETIC_CASE_REVIEW
                else self._prompt_version
            ),
            duration_ms=max(0, duration_ms),
            error_code=result.error_code,
            action_refs=action_refs,
            model_invoked=result.model_invoked,
            translation_used=result.translation_used,
            translation_version=self._translation_version,
            retrieval_version=self._retrieval_version,
        )
