from dataclasses import dataclass

from medical_assistant.workflows import WorkflowResult


@dataclass(frozen=True)
class UiSource:
    source_id: str
    title: str
    section: str
    source: str
    excerpt: str | None
    lexical_score: float | None
    semantic_score: float | None
    patient_record: bool


@dataclass(frozen=True)
class UiResult:
    status: str
    answer: str | None
    error_code: str | None
    uncertainty: str | None
    review_status: str | None
    model_invoked: bool
    sources: tuple[UiSource, ...]
    alert_status: str | None


def to_ui_result(result: WorkflowResult) -> UiResult:
    response = result.response
    sources = tuple(
        UiSource(
            source_id=(
                "registro local" if detail.patient_record else detail.citation.source_id
            ),
            title=(
                "Prontuário selecionado"
                if detail.patient_record
                else detail.citation.title
            ),
            section=(
                "Metadados ocultos"
                if detail.patient_record
                else detail.citation.section
            ),
            source=(
                "registro local selecionado"
                if detail.patient_record
                else detail.citation.source
            ),
            excerpt=None if detail.patient_record else detail.excerpt,
            lexical_score=None if detail.patient_record else detail.lexical_score,
            semantic_score=None if detail.patient_record else detail.semantic_score,
            patient_record=detail.patient_record,
        )
        for detail in result.source_details
    )
    return UiResult(
        status=result.status.value,
        answer=response.answer if response is not None else None,
        error_code=result.error_code,
        uncertainty=response.uncertainty if response is not None else None,
        review_status=(
            response.review_status.value if response is not None else None
        ),
        model_invoked=result.model_invoked,
        sources=sources,
        alert_status=(
            result.simulated_alert.status
            if result.simulated_alert is not None
            else None
        ),
    )
