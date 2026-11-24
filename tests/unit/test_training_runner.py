from pathlib import Path
from types import SimpleNamespace

import pytest

from medical_assistant.training.runner import (
    _append_retrieved_citations,
    _apply_model_safety_gates,
    _evaluation_limitations,
    _load_training_retrieval_documents,
    _tokenize_records,
    load_jsonl,
    validate_splits,
)
from medical_assistant.workflows import WorkflowStatus


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_TRAIN = ROOT / "training" / "synthetic" / "train.jsonl"
SYNTHETIC_VALIDATION = ROOT / "training" / "synthetic" / "validation.jsonl"
SYNTHETIC_TEST = ROOT / "training" / "synthetic" / "test.jsonl"


def load_synthetic_splits() -> dict[str, list[dict[str, object]]]:
    return {
        "train": load_jsonl(SYNTHETIC_TRAIN),
        "validation": load_jsonl(SYNTHETIC_VALIDATION),
        "test": load_jsonl(SYNTHETIC_TEST),
    }


def test_checked_in_smoke_splits_are_synthetic_and_disjoint() -> None:
    validate_splits(load_synthetic_splits(), synthetic_only=True)


def test_split_validator_rejects_duplicate_source_ids_across_splits() -> None:
    splits = load_synthetic_splits()
    splits["validation"][0]["source_id"] = splits["train"][0]["source_id"]

    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits, synthetic_only=True)


def test_split_validator_rejects_duplicate_content_across_splits() -> None:
    splits = load_synthetic_splits()
    splits["validation"][0]["question"] = splits["train"][0]["question"]
    splits["validation"][0]["answer"] = splits["train"][0]["answer"]

    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits, synthetic_only=True)


def test_split_validator_rejects_internal_and_unlabeled_data() -> None:
    splits = load_synthetic_splits()
    splits["train"][0]["source_dataset"] = "internal_patient_data"

    with pytest.raises(ValueError, match="Unsupported or internal"):
        validate_splits(splits, synthetic_only=True)

    splits = load_synthetic_splits()
    splits["train"][0]["source_dataset"] = "pubmedqa"
    splits["train"][0]["source_subset"] = "PQA-U"
    with pytest.raises(ValueError, match="PQA-U"):
        validate_splits(splits, synthetic_only=True)


def test_split_validator_rejects_artificial_data_in_evaluation_splits() -> None:
    splits = load_synthetic_splits()
    splits["validation"][0]["source_dataset"] = "pubmedqa"
    splits["validation"][0]["source_subset"] = "PQA-A"
    splits["validation"][0]["synthetic"] = False

    with pytest.raises(ValueError, match="train-only"):
        validate_splits(splits, synthetic_only=False)


def test_jsonl_loader_reports_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"valid": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_jsonl(path)


def test_base_model_output_is_replaced_when_evidence_gate_abstains() -> None:
    record = {
        "source_id": "test-004",
        "question": "Qual é a temperatura da sala no protocolo fictício?",
        "answer": "Não inventar uma resposta.",
    }

    class EvidenceGate:
        def run(self, question: str, *, actor: object) -> SimpleNamespace:
            assert question == record["question"]
            assert actor is not None
            return SimpleNamespace(
                status=WorkflowStatus.INSUFFICIENT_EVIDENCE,
                response=SimpleNamespace(
                    answer="Não encontrei fontes suficientes para responder.",
                    sources=(),
                ),
            )

    guarded = _apply_model_safety_gates(
        workflow=EvidenceGate(),
        workflow_status=WorkflowStatus,
        actor=object(),
        records=[record],
        raw_predictions=[
            {
                "source_id": "test-004",
                "prediction": "A temperatura é 20 graus.",
                "reference": record["answer"],
            }
        ],
    )

    assert guarded[0]["prediction"] == "Não encontrei fontes suficientes para responder."
    assert guarded[0]["observed_behavior"] == "abstain"


def test_base_model_without_context_abstains_and_points_to_retrieved_source() -> None:
    record = {
        "source_id": "test-002",
        "question": "Quais metadados devo consultar?",
        "answer": "Consulte os metadados.",
    }

    class EvidenceGate:
        def run(self, question: str, *, actor: object) -> SimpleNamespace:
            assert question == record["question"]
            assert actor is not None
            return SimpleNamespace(
                status=WorkflowStatus.COMPLETE,
                response=SimpleNamespace(
                    answer="Trecho recuperado.",
                    sources=[SimpleNamespace(source_id="FAQ-SYNTH-001")],
                ),
            )

    guarded = _apply_model_safety_gates(
        workflow=EvidenceGate(),
        workflow_status=WorkflowStatus,
        actor=object(),
        records=[record],
        raw_predictions=[
            {
                "source_id": "test-002",
                "prediction": "Resposta não fundamentada.",
                "reference": record["answer"],
            }
        ],
    )

    assert "não foi fornecida ao modelo-base" in guarded[0]["prediction"]
    assert "Fontes: [FAQ-SYNTH-001]" in guarded[0]["prediction"]
    assert guarded[0]["citation_source_ids"] == ["FAQ-SYNTH-001"]


def test_finetuned_answer_gets_explicit_retrieved_source_citation() -> None:
    answer, cited_sources = _append_retrieved_citations(
        "Consulte os metadados do trecho.",
        ["FAQ-SYNTH-001"],
    )

    assert answer.endswith("Fontes: [FAQ-SYNTH-001]")
    assert cited_sources == ["FAQ-SYNTH-001"]


def test_pubmedqa_context_is_used_and_trimmed_before_answer_tokens() -> None:
    class CharacterTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert tokenize is True
            text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return [ord(character) for character in text]

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(character) for character in text]

        def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return "".join(chr(token) for token in tokens)

    answer = "Expert answer."
    records = [
        {
            "source_id": "PMID-SYNTH-001",
            "question": "Question?",
            "context": "Evidence. " * 200,
            "answer": answer,
        }
    ]

    tokenized, truncated = _tokenize_records(
        CharacterTokenizer(),
        records,
        max_length=400,
    )

    assert truncated == 1
    assert len(tokenized[0]["input_ids"]) <= 400
    assert tokenized[0]["input_ids"][-1] == ord(answer[-1])
    assert tokenized[0]["labels"][-1] == ord(answer[-1])


def test_pubmedqa_evaluation_summary_reports_training_only_retrieval_scope() -> None:
    limitations = _evaluation_limitations(
        test_count=500,
        smoke_test=False,
        uses_pubmedqa=True,
    )

    assert any("500 held-out records" in item for item in limitations)
    assert any("only PubMedQA training-split abstracts" in item for item in limitations)
    assert not any("synthetic smoke-test data" in item for item in limitations)


def test_medquad_evaluation_summary_reports_training_only_retrieval_scope() -> None:
    limitations = _evaluation_limitations(
        test_count=64,
        smoke_test=False,
        uses_pubmedqa=False,
        uses_medquad=True,
    )

    assert any("only MedQuAD training-split documents" in item for item in limitations)


def test_evaluation_retrieval_uses_only_training_split_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from medical_assistant.datasets import DatasetLoadResult, QARecord
    from medical_assistant.retrieval import SourceDocument
    from medical_assistant.training import runner

    pubmedqa_train = QARecord(
        id="pubmedqa:pqal:pmid-train",
        question="train question",
        answer="train answer",
        source_dataset="pubmedqa",
        source_id="pmid-train",
        license="MIT",
        source_subset="PQA-L",
        context="train abstract",
    )
    pubmedqa_test = QARecord(
        id="pubmedqa:pqal:pmid-test",
        question="test question",
        answer="test answer",
        source_dataset="pubmedqa",
        source_id="pmid-test",
        license="MIT",
        source_subset="PQA-L",
        context="test abstract",
    )
    medquad_train = QARecord(
        id="medquad:train",
        question="train MedQuAD question",
        answer="train MedQuAD answer",
        source_dataset="medquad",
        source_id="medquad-train",
        license="CC BY 4.0",
    )
    medquad_test = QARecord(
        id="medquad:test",
        question="test MedQuAD question",
        answer="test MedQuAD answer",
        source_dataset="medquad",
        source_id="medquad-test",
        license="CC BY 4.0",
    )
    pubmedqa_document = SourceDocument(
        id="pubmedqa:pmid-train",
        text="train abstract",
        source="source",
        title="title",
        section="abstract",
        source_dataset="pubmedqa",
        source_id="pmid-train",
    )
    medquad_document = SourceDocument(
        id="medquad:train",
        text="train MedQuAD answer",
        source="source",
        title="title",
        section="QA",
        source_dataset="medquad",
        source_id="medquad-train",
    )
    pubmedqa_test_document = SourceDocument(
        id="pubmedqa:pmid-test",
        text="test abstract",
        source="source",
        title="title",
        section="abstract",
        source_dataset="pubmedqa",
        source_id="pmid-test",
    )
    medquad_test_document = SourceDocument(
        id="medquad:test",
        text="test MedQuAD answer",
        source="source",
        title="title",
        section="QA",
        source_dataset="medquad",
        source_id="medquad-test",
    )
    loaded_pubmedqa_ids: list[str] = []
    loaded_medquad_ids: list[str] = []

    def pubmedqa_documents(records: tuple[QARecord, ...]) -> tuple[SourceDocument, ...]:
        loaded_pubmedqa_ids.extend(record.source_id for record in records)
        document_by_id = {
            "pmid-train": pubmedqa_document,
            "pmid-test": pubmedqa_test_document,
        }
        return tuple(document_by_id[record.source_id] for record in records)

    def medquad_documents(records: tuple[QARecord, ...]) -> tuple[SourceDocument, ...]:
        loaded_medquad_ids.extend(record.id for record in records)
        document_by_id = {
            "medquad:train": medquad_document,
            "medquad:test": medquad_test_document,
        }
        return tuple(document_by_id[record.id] for record in records)

    monkeypatch.setattr(runner, "load_synthetic_documents", lambda _: ())

    def load_pubmedqa(path: Path, *, subset: str) -> DatasetLoadResult:
        assert path == tmp_path / "pqal.json"
        assert subset == "PQA-L"
        return DatasetLoadResult(records=(pubmedqa_train, pubmedqa_test))

    monkeypatch.setattr(
        runner,
        "load_pubmedqa",
        load_pubmedqa,
    )
    monkeypatch.setattr(
        runner,
        "load_pubmedqa_documents",
        pubmedqa_documents,
    )
    def load_medquad(path: Path) -> DatasetLoadResult:
        assert path == tmp_path / "medquad"
        return DatasetLoadResult(records=(medquad_train, medquad_test))

    monkeypatch.setattr(runner, "load_medquad", load_medquad)
    monkeypatch.setattr(
        runner,
        "load_medquad_documents",
        medquad_documents,
    )

    splits = {
        "train": [
            {"source_dataset": "pubmedqa", "source_id": pubmedqa_train.source_id},
            {"source_dataset": "medquad", "id": medquad_train.id},
        ],
        "validation": [],
        "test": [
            {"source_dataset": "pubmedqa", "source_id": pubmedqa_test.source_id},
            {"source_dataset": "medquad", "id": medquad_test.id},
        ],
    }

    documents = _load_training_retrieval_documents(
        tmp_path,
        {
            "retrieval_source": "pqal.json",
            "medquad_source": "medquad",
        },
        splits,
        uses_pubmedqa=True,
        uses_medquad=True,
    )

    assert documents == [pubmedqa_document, medquad_document]
    assert loaded_pubmedqa_ids == [pubmedqa_train.source_id]
    assert loaded_medquad_ids == [medquad_train.id]


def test_smoke_test_evaluation_summary_is_labeled_as_synthetic() -> None:
    limitations = _evaluation_limitations(
        test_count=4,
        smoke_test=True,
        uses_pubmedqa=False,
    )

    assert any("4 synthetic test cases" in item for item in limitations)
    assert any("one step on synthetic smoke-test data" in item for item in limitations)
