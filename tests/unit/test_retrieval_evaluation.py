import pytest

from medical_assistant.retrieval import (
    DocumentIndex,
    RetrievalEvalCase,
    SourceDocument,
    evaluate_retrieval,
)


def _document(document_id: str, text: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        text=text,
        source=f"source://{document_id}",
        title=document_id,
        section="demo",
        source_dataset="synthetic",
        source_id=document_id,
        synthetic=True,
    )


def test_evaluate_retrieval_reports_recall_hit_rate_and_mrr() -> None:
    index = DocumentIndex(
        [
            _document("DOC-A", "alpha"),
            _document("DOC-B", "beta"),
        ]
    )

    metrics = evaluate_retrieval(
        index,
        [
            RetrievalEvalCase("alpha", ("DOC-A",)),
            RetrievalEvalCase("beta", ("DOC-A",)),
        ],
        top_k=2,
    )

    assert metrics.cases == 2
    assert metrics.top_k == 2
    assert metrics.recall_at_k == 0.5
    assert metrics.hit_rate_at_k == 0.5
    assert metrics.mean_reciprocal_rank_at_k == 0.5


def test_evaluate_retrieval_rejects_expected_sources_outside_the_index() -> None:
    index = DocumentIndex([_document("DOC-A", "alpha")])

    with pytest.raises(ValueError, match="not present in the index"):
        evaluate_retrieval(
            index,
            [RetrievalEvalCase("alpha", ("DOC-MISSING",))],
        )


def test_evaluate_retrieval_requires_nonempty_cases_and_positive_top_k() -> None:
    index = DocumentIndex([_document("DOC-A", "alpha")])

    with pytest.raises(ValueError, match="At least one"):
        evaluate_retrieval(index, [])
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(
            index,
            [RetrievalEvalCase("alpha", ("DOC-A",))],
            top_k=0,
        )


def test_evaluate_retrieval_validates_score_thresholds() -> None:
    index = DocumentIndex([_document("DOC-A", "alpha")])
    case = [RetrievalEvalCase("alpha", ("DOC-A",))]

    with pytest.raises(ValueError, match="min_lexical_score"):
        evaluate_retrieval(index, case, min_lexical_score=1.1)
    with pytest.raises(ValueError, match="min_semantic_score"):
        evaluate_retrieval(index, case, min_semantic_score=1.1)
