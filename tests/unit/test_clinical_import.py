import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from medical_assistant.datasets.clinical_import import (
    convert_authorized_patient_records,
    convert_authorized_protocols,
)
from medical_assistant.audit import AuditActor, AuditEvent
from medical_assistant.patient_data import (
    AccessContext,
    AccessDeniedError,
    AuthorizedPatientRecordStore,
    CompositePatientStore,
    PatientRole,
)
from medical_assistant.retrieval import (
    DocumentIndex,
    load_approved_protocol_documents,
)
from medical_assistant.workflows import AssistantWorkflow


class _RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class _CapturingAnswerer:
    def __init__(self) -> None:
        self.documents: tuple[Document, ...] = ()

    def answer(self, question: str, documents: tuple[Document, ...]) -> str:
        self.documents = documents
        return "Resposta de teste."


def _schema() -> dict[str, object]:
    protocol = {
        "type": "object",
        "required": [
            "protocolo_id",
            "titulo",
            "especialidade",
            "versao",
            "cid10_relacionados",
            "criterios_inclusao",
            "etapas",
            "prescricao_padrao",
        ],
        "properties": {
            "protocolo_id": {"type": "string"},
            "titulo": {"type": "string"},
            "especialidade": {"type": "string"},
            "versao": {"type": "string"},
            "cid10_relacionados": {
                "type": "array",
                "items": {"type": "string"},
            },
            "criterios_inclusao": {
                "type": "array",
                "items": {"type": "string"},
            },
            "etapas": {"type": "array"},
            "prescricao_padrao": {"type": "array"},
        },
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["protocolos_clinicos", "prontuarios"],
        "properties": {
            "protocolos_clinicos": {
                "type": "array",
                "items": protocol,
            },
            "prontuarios": {"type": "array"},
        },
    }


def _source_data() -> dict[str, object]:
    return {
        "protocolos_clinicos": [
            {
                "protocolo_id": "PROTO-SYNTH-001",
                "titulo": "Protocolo sintético de teste",
                "especialidade": "demonstração",
                "versao": "0.1",
                "cid10_relacionados": ["Z00"],
                "criterios_inclusao": ["caso fictício"],
                "etapas": [{"ordem": 1, "acao": "revisar fonte"}],
                "prescricao_padrao": [],
            }
        ],
        "prontuarios": [
            {
                "prontuario_id": "RECORD-SYNTH-001",
                "paciente": {
                    "id": "PERSON-SYNTH-001",
                    "nome_completo": "Pessoa Fictícia",
                    "cpf_mascarado": "***.000.000-**",
                    "cns": "CNS-SYNTH-001",
                    "data_nascimento": "1980-01-01",
                    "idade": 45,
                    "sexo_biologico": "F",
                    "tipo_sanguineo": "O+",
                    "alergias": ["alergia fictícia"],
                    "comorbidades_previas": ["condição fictícia"],
                },
                "atendimentos": [
                    {
                        "atendimento_id": "VISIT-SYNTH-001",
                        "data_entrada": "2026-06-01T10:00:00",
                        "data_alta": None,
                        "status": "Em observação",
                        "classificacao_manchester": "Verde",
                        "protocolo_aplicado_id": "PROTO-SYNTH-001",
                        "profissional_responsavel": {
                            "nome": "Profissional Fictício",
                            "conselho": "CRM-SYNTH-001",
                            "especialidade": "demonstração",
                        },
                        "triagem_sinais_vitais": {
                            "pa_mmhg": "120/80",
                            "fc_bpm": 70,
                            "fr_irpm": 16,
                            "temperatura_c": 36.5,
                            "spo2_pct": 98,
                            "glasgow": 15,
                            "glicemia_mg_dl": 90,
                        },
                        "evolucao_soap": {
                            "subjetivo": "Relato fictício; contato pessoa@example.com",
                            "objetivo": "Achado fictício",
                            "avaliacao": "Avaliação fictícia",
                            "plano": "Plano fictício",
                        },
                        "exames_solicitados": [
                            {
                                "tuss": "TUSS-SYNTH-001",
                                "nome": "Exame fictício",
                                "resultado": "Resultado fictício",
                                "status": "Em análise",
                            }
                        ],
                        "prescricoes_ativas": [
                            {
                                "medicamento": "Medicamento fictício",
                                "via": "oral",
                                "posologia": "uso fictício",
                                "horario_administracao": "10:00",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_conversion_exports_protocols_and_pseudonymized_patient_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "processed" / "protocols.jsonl"
    patient_output_path = tmp_path / "processed" / "patients.jsonl"
    source_path.write_text(
        json.dumps(_source_data(), ensure_ascii=False),
        encoding="utf-8",
    )
    schema_path.write_text(json.dumps(_schema()), encoding="utf-8")

    report = convert_authorized_protocols(
        source_path,
        schema_path,
        output_path,
        institutional_approval_reference="approval-123",
        institutional_approved_at="2026-07-02",
        privacy_review_reference="privacy-456",
        privacy_reviewed_at="2026-07-10",
    )

    assert report["protocols_converted"] == 1
    assert report["patient_records_not_exported"] == 1
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["training_eligible"] is False
    assert records[0]["institutional_approval_reference"] == "approval-123"
    assert "prontuario_id" not in output_path.read_text(encoding="utf-8")

    documents = load_approved_protocol_documents(output_path)

    assert len(documents) == 1
    assert documents[0].source_id == "PROTO-SYNTH-001"
    assert documents[0].version == "0.1"
    assert documents[0].synthetic is False
    results = DocumentIndex(documents).search("Protocolo sintético de teste")
    assert results
    assert results[0].chunk.source_dataset == "hospital_protocol"
    assert results[0].chunk.source_id == "PROTO-SYNTH-001"

    records[0]["contains_patient_data"] = True
    output_path.write_text(
        json.dumps(records[0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approval metadata is invalid"):
        load_approved_protocol_documents(output_path)

    patient_report = convert_authorized_patient_records(
        source_path,
        schema_path,
        patient_output_path,
        institutional_approval_reference="approval-123",
        institutional_approved_at="2026-07-02",
        privacy_review_reference="privacy-456",
        privacy_reviewed_at="2026-07-10",
    )
    assert patient_report["patient_records_converted"] == 1
    patient_jsonl = patient_output_path.read_text(encoding="utf-8")
    assert "Pessoa Fictícia" not in patient_jsonl
    assert "Profissional Fictício" not in patient_jsonl
    assert "data_nascimento" not in patient_jsonl
    assert "cpf_mascarado" not in patient_jsonl
    assert "CNS-SYNTH-001" not in patient_jsonl
    assert "pessoa@example.com" not in patient_jsonl

    patient_store = AuthorizedPatientRecordStore(patient_output_path)
    assert patient_store.available_patient_ids == ("CASE-001",)
    composite_patient_store = CompositePatientStore([patient_store])
    assert composite_patient_store.available_patient_ids == ("CASE-001",)
    record = patient_store.read_patient(
        "CASE-001",
        AccessContext(
            user_id="demo-clinician",
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({"CASE-001"}),
        ),
    )
    assert record.source == "local:authorized_patient_record"
    assert "Pessoa Fictícia" not in record.records[0].content
    assert "[EMAIL_REMOVIDO]" in record.records[1].content
    assert record.encounters[0].exams[0].status.value == "pending"
    with pytest.raises(AccessDeniedError):
        patient_store.read_patient(
            "CASE-001",
            AccessContext(
                user_id="demo-clinician",
                role=PatientRole.CLINICIAN,
                authorized_patient_ids=frozenset(),
            ),
        )

    audit_sink = _RecordingAuditSink()
    workflow = AssistantWorkflow(
        DocumentIndex([]),
        composite_patient_store,
        audit_sink,
        chat_model=FakeListChatModel(responses=["unused"]),
        model_version="fake-model",
    )
    answerer = _CapturingAnswerer()
    assert workflow._answerer is not None
    monkeypatch.setattr(workflow._answerer, "answer", answerer.answer)
    actor = AuditActor(user_id="demo-clinician", role="clinician")
    unselected_result = workflow.run("Consulta informativa", actor=actor)
    assert unselected_result.patient_record is None
    assert audit_sink.events[-1].patient_data_accessed is False

    selected_result = workflow.run(
        "Consulta sobre o registro selecionado",
        actor=actor,
        patient_id="CASE-001",
        access_context=AccessContext(
            user_id=actor.user_id,
            role=PatientRole.CLINICIAN,
            authorized_patient_ids=frozenset({"CASE-001"}),
        ),
    )
    assert selected_result.patient_record is not None
    assert selected_result.patient_record.patient_id == "CASE-001"
    assert answerer.documents
    assert any("Evolução clínica" in doc.page_content for doc in answerer.documents)
    assert all("Pessoa Fictícia" not in doc.page_content for doc in answerer.documents)
    assert audit_sink.events[-1].patient_data_accessed is True
    assert "CASE-001" not in audit_sink.events[-1].to_json()
    assert audit_sink.events[-1].source_refs == ("local:authorized_patient_record",)

    patient_document = json.loads(patient_jsonl)
    patient_document["patient_profile"]["nome_completo"] = "Pessoa Fictícia"
    patient_output_path.write_text(
        json.dumps(patient_document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid patient profile"):
        AuthorizedPatientRecordStore(patient_output_path)

    patient_document["patient_profile"].pop("nome_completo")
    patient_document["training_eligible"] = True
    patient_output_path.write_text(
        json.dumps(patient_document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Patient approval metadata is invalid"):
        AuthorizedPatientRecordStore(patient_output_path)


def test_conversion_rejects_source_that_does_not_match_schema(
    tmp_path: Path,
) -> None:
    source = _source_data()
    source["protocolos_clinicos"] = [{}]
    source_path = tmp_path / "source.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "protocols.jsonl"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    schema_path.write_text(json.dumps(_schema()), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the supplied schema"):
        convert_authorized_protocols(
            source_path,
            schema_path,
            output_path,
            institutional_approval_reference="approval-123",
            institutional_approved_at="2026-07-02",
            privacy_review_reference="privacy-456",
            privacy_reviewed_at="2026-07-10",
        )

    assert not output_path.exists()
