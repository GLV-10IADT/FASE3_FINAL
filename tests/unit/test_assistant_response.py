import pytest

from medical_assistant.assistant import (
    AssistantResponse,
    ReviewStatus,
    SourceCitation,
    validate_source_citations,
)


def make_source(source_id: str = "DOC-1") -> SourceCitation:
    return SourceCitation(
        source=f"source://{source_id}",
        title="Synthetic source",
        section="demo",
        source_id=source_id,
    )


def test_response_accepts_valid_required_fields_and_sources() -> None:
    source = make_source()
    response = AssistantResponse(
        answer="Resposta informativa.",
        sources=(source,),
        uncertainty="Baseada apenas na fonte sintética de demonstração.",
        review_status=ReviewStatus.PENDING,
    )

    validate_source_citations(response, [source])


def test_response_rejects_empty_answer_or_uncertainty() -> None:
    with pytest.raises(ValueError, match="Answer must"):
        AssistantResponse(
            answer=" ",
            sources=(),
            uncertainty="Sem evidência suficiente.",
            review_status=ReviewStatus.PENDING,
        )
    with pytest.raises(ValueError, match="Uncertainty must"):
        AssistantResponse(
            answer="Resposta informativa.",
            sources=(),
            uncertainty=" ",
            review_status=ReviewStatus.PENDING,
        )


def test_response_rejects_invalid_review_status_and_source_values() -> None:
    with pytest.raises(ValueError, match="ReviewStatus"):
        AssistantResponse(
            answer="Resposta informativa.",
            sources=(),
            uncertainty="Incerteza não avaliada.",
            review_status="pending",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Citation title"):
        SourceCitation(source="source://1", title=" ", section="demo", source_id="1")


def test_response_rejects_citations_not_in_available_sources() -> None:
    response = AssistantResponse(
        answer="Resposta com referência.",
        sources=(make_source("not-indexed"),),
        uncertainty="Fonte não consta no índice atual.",
        review_status=ReviewStatus.PENDING,
    )

    with pytest.raises(ValueError, match="unavailable sources"):
        validate_source_citations(response, [make_source("known")])
