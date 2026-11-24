import json
from dataclasses import dataclass, replace
from pathlib import Path

from medical_assistant.audit import AuditActor, JsonlAuditLogger
from medical_assistant.datasets import QARecord, load_medquad, load_pubmedqa
from medical_assistant.patient_data import (
    AuthorizedPatientRecordStore,
    CompositePatientStore,
    PatientStore,
    SyntheticPatientStore,
    SyntheticStaffMember,
    SyntheticStaffStore,
)
from medical_assistant.retrieval import (
    DocumentIndex,
    LocalE5Embedder,
    load_approved_protocol_documents,
    load_medquad_documents,
    load_pubmedqa_documents,
    load_synthetic_documents,
)
from medical_assistant.training.runtime import load_finetuned_chat_model
from medical_assistant.translation import LocalOpusTranslator
from medical_assistant.workflows import AssistantWorkflow, JsonlSimulatedAlertSink


@dataclass(frozen=True)
class AssistantRuntime:
    workflow: AssistantWorkflow
    actor: AuditActor
    staff_member: SyntheticStaffMember
    patient_store: PatientStore
    protocol_count: int
    patient_record_count: int
    pubmedqa_abstract_count: int
    medquad_record_count: int
    model_version: str
    retrieval_version: str
    audit_path: Path
    alerts_path: Path


def build_assistant_runtime(
    project_root: Path,
    *,
    config_path: Path | None = None,
    adapter_dir: Path | None = None,
    approved_protocols_path: Path | None = None,
    approved_patient_records_path: Path | None = None,
    audit_path: Path | None = None,
    alerts_path: Path | None = None,
) -> AssistantRuntime:
    def resolve(path: Path | None, default: str) -> Path:
        selected = path if path is not None else Path(default)
        return selected if selected.is_absolute() else project_root / selected

    resolved_config_path = resolve(
        config_path,
        "training/configs/qwen2_5_0_5b_pubmedqa_qlora.json",
    )
    resolved_adapter_dir = (
        None
        if adapter_dir is None
        else adapter_dir
        if adapter_dir.is_absolute()
        else project_root / adapter_dir
    )
    resolved_protocols_path = resolve(
        approved_protocols_path,
        "data/processed/protocolos_institucionais_autorizados.jsonl",
    )
    resolved_patient_records_path = resolve(
        approved_patient_records_path,
        "data/processed/prontuarios_autorizados.jsonl",
    )
    resolved_audit_path = resolve(
        audit_path,
        "artifacts/chatbot-assistente-audit.jsonl",
    )
    resolved_alerts_path = resolve(
        alerts_path,
        "artifacts/chatbot-assistente-alerts.jsonl",
    )

    training_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    data_config = training_config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("The selected model config must contain a data object")
    retrieval_source = data_config.get("retrieval_source")
    if not isinstance(retrieval_source, str) or not retrieval_source.strip():
        raise ValueError("The selected model config must declare data.retrieval_source")

    synthetic_dir = project_root / "data" / "synthetic"
    pubmedqa_path = project_root / retrieval_source
    if not pubmedqa_path.is_file():
        raise FileNotFoundError(f"PubMedQA retrieval data is missing: {pubmedqa_path}")
    pubmedqa_records = load_pubmedqa(pubmedqa_path, subset="PQA-L")
    documents = list(load_synthetic_documents(synthetic_dir))
    documents.extend(load_pubmedqa_documents(pubmedqa_records.records))
    medquad_source = data_config.get("medquad_source")
    medquad_records: tuple[QARecord, ...] = ()
    if medquad_source is not None:
        if not isinstance(medquad_source, str) or not medquad_source.strip():
            raise ValueError("The selected model config has an invalid data.medquad_source")
        medquad_result = load_medquad(
            resolve(Path(medquad_source), "data/raw/medquad")
        )
        if not medquad_result.records:
            raise ValueError("MedQuAD source has no eligible answered records")
        medquad_records = medquad_result.records
        documents.extend(load_medquad_documents(medquad_records))
    approved_protocols = load_approved_protocol_documents(resolved_protocols_path)
    documents.extend(approved_protocols)
    embedding_model = LocalE5Embedder(
        cache_dir=project_root / "artifacts" / "retrieval-embeddings"
    )
    document_index = DocumentIndex(
        documents,
        embedding_model=embedding_model,
        embedding_cache_path=(
            project_root / "artifacts" / "retrieval-index" / "embeddings.bin"
        ),
    )

    synthetic_patient_store = SyntheticPatientStore(synthetic_dir / "patients.json")
    approved_patient_store = AuthorizedPatientRecordStore(
        resolved_patient_records_path
    )
    patient_store = CompositePatientStore(
        [synthetic_patient_store, approved_patient_store]
    )
    staff_member = SyntheticStaffStore(synthetic_dir / "staff.json").read_staff(
        "demo-clinician"
    )
    staff_member = replace(
        staff_member,
        authorized_patient_ids=(
            staff_member.authorized_patient_ids
            | frozenset(approved_patient_store.available_patient_ids)
        ),
    )

    translation_dir = project_root / "artifacts" / "translation"
    query_translator = LocalOpusTranslator.from_model_directory(
        translation_dir / "pt_en",
        from_code="pt",
        to_code="en",
    )
    answer_translator = LocalOpusTranslator.from_model_directory(
        translation_dir / "en_pt",
        from_code="en",
        to_code="pt",
    )
    chat_model, model_version = load_finetuned_chat_model(
        resolved_config_path,
        adapter_dir=resolved_adapter_dir,
    )
    actor = AuditActor(user_id=staff_member.user_id, role=staff_member.role.value)
    workflow = AssistantWorkflow(
        document_index,
        patient_store,
        JsonlAuditLogger(resolved_audit_path),
        chat_model=chat_model,
        query_translator=query_translator,
        answer_translator=answer_translator,
        translation_version=f"{query_translator.version}; {answer_translator.version}",
        retrieval_version=embedding_model.version,
        model_version=model_version,
        simulated_alert_sink=JsonlSimulatedAlertSink(resolved_alerts_path),
    )
    return AssistantRuntime(
        workflow=workflow,
        actor=actor,
        staff_member=staff_member,
        patient_store=patient_store,
        protocol_count=len(approved_protocols),
        patient_record_count=len(approved_patient_store.available_patient_ids),
        pubmedqa_abstract_count=len(pubmedqa_records.records),
        medquad_record_count=len(medquad_records),
        model_version=model_version,
        retrieval_version=embedding_model.version,
        audit_path=resolved_audit_path,
        alerts_path=resolved_alerts_path,
    )
