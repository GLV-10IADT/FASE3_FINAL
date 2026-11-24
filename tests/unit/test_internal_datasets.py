import json
from dataclasses import replace
from pathlib import Path

import pytest

from medical_assistant.datasets import (
    DatasetSplits,
    export_training_splits,
    load_approved_internal_jsonl,
    preprocess_approved_internal_jsonl,
)
from medical_assistant.training.runner import load_jsonl, validate_splits

ROOT = Path(__file__).resolve().parents[2]


def internal_record(source_id: str = "DOC-001", **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source_dataset": "hospital_protocol",
        "source_id": source_id,
        "source_group_id": source_id,
        "source_version": "v1",
        "question": f"Pergunta interna aprovada {source_id}?",
        "answer": f"Resposta interna aprovada {source_id}.",
        "usage_rights_reference": "GOV-2026-001",
        "institutional_approval_status": "approved",
        "institutional_approval_reference": "APPROVAL-2026-001",
        "institutional_approved_at": "2026-11-20",
        "privacy_review_status": "approved",
        "privacy_review_reference": "PRIVACY-2026-001",
        "privacy_reviewed_at": "2026-11-21",
        "contains_patient_data": False,
        "review_status": "approved",
        "preprocessing_version": "local-regex-v2",
        "preprocessing_redaction_counts": {
            "email": 0,
            "cpf": 0,
            "phone": 0,
            "medical_record_id": 0,
            "hospital_document_id": 0,
            "date_of_birth": 0,
            "postal_code": 0,
        },
        "preprocessing_review_required": False,
    }
    record.update(updates)
    return record


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_internal_loader_requires_provenance_and_review_attestations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "curated.jsonl"
    write_jsonl(path, [internal_record()])

    result = load_approved_internal_jsonl(path)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.source_dataset == "hospital_protocol"
    assert record.source_group_id == "DOC-001"
    assert record.source_version == "v1"
    assert record.license is None
    assert record.institutional_approval_reference == "APPROVAL-2026-001"
    assert record.privacy_review_reference == "PRIVACY-2026-001"
    assert record.contains_patient_data is False
    assert record.preprocessing_version == "local-regex-v2"
    assert record.preprocessing_review_required is False


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"institutional_approval_status": "pending"}, "lacks institutional approval"),
        ({"privacy_review_status": "pending"}, "lacks an approved privacy review"),
        ({"contains_patient_data": True}, "containing no patient data"),
        ({"preprocessing_review_required": True}, "still requires human review"),
        ({"source_dataset": "hospital_patient_record"}, "Unsupported internal source"),
        ({"patient_id": "PATIENT-123"}, "patient identifier fields"),
        ({"privacy_reviewed_at": "not-a-date"}, "invalid 'privacy_reviewed_at'"),
    ],
)
def test_internal_loader_rejects_unapproved_or_patient_data(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "curated.jsonl"
    write_jsonl(path, [internal_record(**updates)])

    with pytest.raises(ValueError, match=message):
        load_approved_internal_jsonl(path)


def test_internal_splits_export_approval_metadata_for_training(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "curated.jsonl"
    write_jsonl(
        source_path,
        [internal_record("DOC-001"), internal_record("DOC-002"), internal_record("DOC-003")],
    )
    records = load_approved_internal_jsonl(source_path).records
    splits = DatasetSplits(
        train=(records[0],),
        validation=(records[1],),
        test=(records[2],),
    )

    paths = export_training_splits(splits, tmp_path / "processed")
    training_splits = {
        split_name: load_jsonl(path)
        for split_name, path in zip(("train", "validation", "test"), paths)
    }
    validate_splits(training_splits, synthetic_only=False)

    training_record = training_splits["train"][0]
    assert training_record["split"] == "train"
    assert training_record["institutional_approval_reference"] == "APPROVAL-2026-001"
    assert training_record["privacy_review_status"] == "approved"
    assert training_record["preprocessing_redaction_counts"]["email"] == 0


def test_preprocessor_redacts_common_identifiers_and_requires_human_review(
    tmp_path: Path,
) -> None:
    source = internal_record()
    for field in (
        "privacy_review_status",
        "privacy_review_reference",
        "privacy_reviewed_at",
        "review_status",
        "preprocessing_version",
        "preprocessing_redaction_counts",
        "preprocessing_review_required",
    ):
        source.pop(field)
    source["question"] = (
        "Contato: ana@example.com CPF 123.456.789-09, telefone: (11) 99999-1234, "
        "MRN: ABCD1234, HSP-123456/2024, nascimento: 05/03/1980, CEP: 12345-678."
    )
    source_path = tmp_path / "approved-source.jsonl"
    output_path = tmp_path / "privacy-review" / "redacted.jsonl"
    write_jsonl(source_path, [source])

    report = preprocess_approved_internal_jsonl(source_path, output_path)

    processed = json.loads(output_path.read_text(encoding="utf-8"))
    assert "ana@example.com" not in processed["question"]
    assert "123.456.789-09" not in processed["question"]
    assert "[EMAIL_REMOVIDO]" in processed["question"]
    assert "[CPF_REMOVIDO]" in processed["question"]
    assert "[TELEFONE_REMOVIDO]" in processed["question"]
    assert "[IDENTIFICADOR_REMOVIDO]" in processed["question"]
    assert "[DOCUMENTO_REMOVIDO]" in processed["question"]
    assert "[DATA_NASCIMENTO_REMOVIDA]" in processed["question"]
    assert "[CEP_REMOVIDO]" in processed["question"]
    assert processed["privacy_review_status"] == "pending"
    assert processed["contains_patient_data"] is None
    assert processed["preprocessing_review_required"] is True
    assert report["human_review_required"] is True
    assert report["redactions_by_category"]["email"] == 1
    assert report["redactions_by_category"]["hospital_document_id"] == 1

    with pytest.raises(ValueError, match="lacks an approved privacy review"):
        load_approved_internal_jsonl(output_path)

    reviewed = dict(processed)
    reviewed["privacy_review_status"] = "approved"
    reviewed["privacy_review_reference"] = "PRIVACY-REVIEW-2026-001"
    reviewed["privacy_reviewed_at"] = "2026-11-22"
    reviewed["contains_patient_data"] = False
    reviewed["review_status"] = "approved"
    reviewed["preprocessing_review_required"] = False
    reviewed_path = tmp_path / "reviewed.jsonl"
    write_jsonl(reviewed_path, [reviewed])

    approved_records = load_approved_internal_jsonl(reviewed_path).records

    assert len(approved_records) == 1
    assert approved_records[0].preprocessing_redaction_counts is not None
    assert approved_records[0].preprocessing_redaction_counts["email"] == 1


def test_preprocessor_does_not_overwrite_source_or_existing_output(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "approved-source.jsonl"
    write_jsonl(source_path, [internal_record()])

    with pytest.raises(ValueError, match="must not overwrite the source"):
        preprocess_approved_internal_jsonl(source_path, source_path)

    output_path = tmp_path / "redacted.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preprocess_approved_internal_jsonl(source_path, output_path)


def test_preprocessor_normalizes_faq_protocol_and_template_schemas(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "institutional-source.jsonl"
    output_path = tmp_path / "processed" / "privacy-review.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "id": "FAQ-001",
                "pergunta": "Como citar uma fonte?",
                "resposta": "Informe o identificador e a versão.",
                "tags": [
                    "contato: editor@example.invalid",
                    "referência: HSP-765432/2024",
                ],
            },
            {
                "id": "PROTOCOL-001",
                "titulo": "Protocolo de revisão",
                "conteudo": "Use fontes aprovadas.",
                "acao_ia": "Solicitar validação humana.",
                "risco": "Não executar condutas.",
            },
            {
                "id": "TEMPLATE-001",
                "contexto_uso": "Solicitação para revisão.",
                "texto": "Fontes: {sources}; incertezas: {uncertainty}.",
                "variaveis": {"sources": ["DOC-001"], "uncertainty": "informar"},
            },
        ],
    )

    report = preprocess_approved_internal_jsonl(
        source_path,
        output_path,
        institutional_approval_reference="APPROVAL-999",
        institutional_approved_at="2026-05-26",
        usage_rights_reference="corpos oficial registro 999/2026",
    )
    output_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["source_dataset"] for record in output_records] == [
        "hospital_faq",
        "hospital_protocol",
        "hospital_template",
    ]
    assert all(
        record["institutional_approval_reference"] == "APPROVAL-999"
        for record in output_records
    )
    assert all(record["source_version"].startswith("sha256:") for record in output_records)
    assert output_records[0]["source_metadata"]["tags"] == [
        "contato: [EMAIL_REMOVIDO]",
        "referência: [DOCUMENTO_REMOVIDO]",
    ]
    assert all(record["privacy_review_status"] == "pending" for record in output_records)
    assert report["records_processed"] == 3
    assert report["redactions_by_category"]["email"] == 1
    assert report["redactions_by_category"]["hospital_document_id"] == 1


def test_preprocessor_excludes_patient_record_schemas(tmp_path: Path) -> None:
    source_path = tmp_path / "institutional-source.jsonl"
    output_path = tmp_path / "processed" / "privacy-review.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "id": "FAQ-001",
                "pergunta": "Como citar uma fonte?",
                "resposta": "Informe o identificador e a versão.",
            },
            {
                "id": "PATIENT-ROW-001",
                "codigo_prontuario": "HSP 1234/2024",
                "data_atendimento": "2026-01-01",
                "paciente_perfil": "Synthetic profile",
                "soap": "Synthetic note",
            },
        ],
    )

    report = preprocess_approved_internal_jsonl(
        source_path,
        output_path,
        institutional_approval_reference="APPROVAL-999",
        institutional_approved_at="2026-05-26",
        usage_rights_reference="RIGHTS-999",
    )
    output_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(output_records) == 1
    assert output_records[0]["source_dataset"] == "hospital_faq"
    assert report["records_excluded_patient_records"] == 1
    assert report["redactions_by_category"]["hospital_document_id"] == 0


def test_internal_split_export_refuses_to_overwrite_existing_files(
    tmp_path: Path,
) -> None:
    record = load_approved_internal_jsonl(
        _write_single_internal_record(tmp_path / "curated.jsonl")
    ).records[0]
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "train.jsonl").write_text("existing\n", encoding="utf-8")
    splits = DatasetSplits(
        train=(record,),
        validation=(replace(record, id="validation", source_id="validation")),
        test=(replace(record, id="test", source_id="test")),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_training_splits(splits, output_dir)


def test_internal_split_export_rejects_empty_splits(tmp_path: Path) -> None:
    record = load_approved_internal_jsonl(
        _write_single_internal_record(tmp_path / "curated.jsonl")
    ).records[0]
    splits = DatasetSplits(train=(record,), validation=(), test=(record,))

    with pytest.raises(ValueError, match="validation"):
        export_training_splits(splits, tmp_path / "processed")


def test_synthetic_institutional_style_corpus_is_not_marked_as_approved() -> None:
    corpus = load_jsonl(ROOT / "data" / "synthetic" / "corpus_institucional_demo.jsonl")

    assert {record["source_type"] for record in corpus} == {
        "protocol",
        "faq",
        "template",
    }
    assert all(record["synthetic"] is True for record in corpus)
    assert all(record["source_dataset"] == "synthetic" for record in corpus)
    assert all(
        record["institutional_approval_status"] == "not_applicable_synthetic"
        for record in corpus
    )
    assert all(
        record["privacy_review_status"] == "not_applicable_synthetic"
        for record in corpus
    )
    assert all(record["contains_patient_data"] is False for record in corpus)


def _write_single_internal_record(path: Path) -> Path:
    write_jsonl(path, [internal_record()])
    return path
