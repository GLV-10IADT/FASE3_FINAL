import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from medical_assistant import application
from medical_assistant.datasets import DatasetLoadResult, QARecord
from medical_assistant.patient_data import PatientRole, SyntheticStaffMember


def test_runtime_indexes_medquad_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "retrieval_source": "pqal.json",
                    "medquad_source": "medquad",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pqal.json").write_text("{}", encoding="utf-8")
    medquad_record = QARecord(
        id="medquad:nih:source:doc:question",
        question="What is this condition?",
        answer="An educational answer.",
        source_dataset="medquad",
        source_id="nih:source:doc:question",
        license="CC BY 4.0",
        source_url="https://example.test/condition",
        focus="Condition",
    )
    def load_pubmedqa(path: Path, *, subset: str) -> DatasetLoadResult:
        assert path == tmp_path / "pqal.json"
        assert subset == "PQA-L"
        return DatasetLoadResult(records=())

    monkeypatch.setattr(application, "load_pubmedqa", load_pubmedqa)
    monkeypatch.setattr(
        application,
        "load_medquad",
        lambda _: DatasetLoadResult(records=(medquad_record,)),
    )
    monkeypatch.setattr(application, "load_synthetic_documents", lambda _: ())
    monkeypatch.setattr(application, "load_approved_protocol_documents", lambda _: ())

    indexed_documents: list[object] = []

    class RecordingIndex:
        def __init__(
            self,
            documents: tuple[object, ...] | list[object],
            *,
            embedding_model: object,
            embedding_cache_path: Path,
        ) -> None:
            assert getattr(embedding_model, "version") == "test-embeddings"
            assert embedding_cache_path == (
                tmp_path / "artifacts" / "retrieval-index" / "embeddings.bin"
            )
            indexed_documents.extend(documents)

    monkeypatch.setattr(application, "DocumentIndex", RecordingIndex)
    monkeypatch.setattr(
        application,
        "LocalE5Embedder",
        Mock(return_value=SimpleNamespace(version="test-embeddings")),
    )
    monkeypatch.setattr(application, "SyntheticPatientStore", Mock(return_value=object()))
    monkeypatch.setattr(
        application,
        "AuthorizedPatientRecordStore",
        Mock(return_value=SimpleNamespace(available_patient_ids=frozenset())),
    )
    monkeypatch.setattr(application, "CompositePatientStore", Mock(return_value=object()))
    staff = SyntheticStaffMember("demo", PatientRole.CLINICIAN, frozenset())
    monkeypatch.setattr(
        application,
        "SyntheticStaffStore",
        Mock(return_value=SimpleNamespace(read_staff=Mock(return_value=staff))),
    )

    class FakeTranslator:
        version = "test-translator"

        @classmethod
        def from_model_directory(
            cls,
            directory: Path,
            *,
            from_code: str,
            to_code: str,
        ) -> FakeTranslator:
            assert directory.name in {"pt_en", "en_pt"}
            assert (from_code, to_code) in {("pt", "en"), ("en", "pt")}
            return cls()

    monkeypatch.setattr(application, "LocalOpusTranslator", FakeTranslator)
    monkeypatch.setattr(
        application,
        "load_finetuned_chat_model",
        Mock(return_value=(None, "test-model")),
    )
    monkeypatch.setattr(
        application,
        "AssistantWorkflow",
        Mock(return_value=SimpleNamespace()),
    )

    runtime = application.build_assistant_runtime(
        tmp_path,
        config_path=config_path,
        approved_protocols_path=tmp_path / "protocols.jsonl",
        approved_patient_records_path=tmp_path / "patients.jsonl",
    )

    assert runtime.medquad_record_count == 1
    assert len(indexed_documents) == 1
    assert indexed_documents[0].source_id == medquad_record.source_id
    assert indexed_documents[0].license == "CC BY 4.0"
