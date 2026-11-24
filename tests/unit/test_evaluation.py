import pytest

from medical_assistant.training.evaluation import evaluate_system, token_f1


def test_token_f1_uses_normalized_word_overlap() -> None:
    assert token_f1("Evidência suficiente", "evidencia suficiente!") == 1.0


def test_evaluation_reports_behavior_source_and_citation_errors() -> None:
    records = [
        {
            "source_id": "test-001",
            "answer": "Use a fonte citada.",
            "expected_behavior": "answer",
            "expected_source_id": "FAQ-001",
        }
    ]
    predictions = [
        {
            "source_id": "test-001",
            "prediction": "Não encontrei evidência.",
            "retrieved_source_ids": ["FAQ-002"],
            "citation_source_ids": ["FAQ-UNKNOWN"],
        }
    ]

    metrics = evaluate_system(records, predictions)

    assert metrics["behavior_accuracy"] == 0.0
    assert metrics["expected_source_recall"] == 0.0
    assert metrics["citation_validity"] == 0.0
    assert metrics["cases"][0]["issues"] == [
        "behavior_mismatch",
        "expected_source_not_retrieved",
        "expected_source_not_cited",
        "citation_not_in_retrieved_sources",
    ]


def test_evaluation_rejects_prediction_order_or_count_mismatch() -> None:
    record = {
        "source_id": "test-001",
        "answer": "resposta",
        "expected_behavior": "answer",
    }

    with pytest.raises(ValueError, match="one system prediction"):
        evaluate_system([record], [])

    with pytest.raises(ValueError, match="source IDs"):
        evaluate_system([record], [{"source_id": "other", "prediction": "resposta"}])
