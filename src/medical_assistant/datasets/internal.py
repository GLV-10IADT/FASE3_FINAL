import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from medical_assistant.datasets.models import DatasetLoadResult, QARecord
from medical_assistant.datasets.splits import DatasetSplits

INTERNAL_SOURCE_DATASETS = frozenset(
    {"hospital_protocol", "hospital_faq", "hospital_template"}
)
_FORBIDDEN_FIELDS = {
    "dob",
    "date_of_birth",
    "medical_record_number",
    "mrn",
    "national_id",
    "patient_id",
    "patient_identifier",
    "patient_name",
}
_PATIENT_RECORD_FIELDS = {
    "codigo_prontuario",
    "data_atendimento",
    "paciente_perfil",
    "soap",
}
_REQUIRED_TEXT_FIELDS = (
    "source_dataset",
    "source_id",
    "source_group_id",
    "source_version",
    "question",
    "answer",
    "usage_rights_reference",
    "institutional_approval_status",
    "institutional_approval_reference",
    "institutional_approved_at",
    "privacy_review_status",
    "privacy_review_reference",
    "privacy_reviewed_at",
    "preprocessing_version",
)
_REDACTION_PATTERNS = (
    (
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "[EMAIL_REMOVIDO]",
    ),
    (
        "cpf",
        re.compile(r"(?<!\d)(?:\d{3}\.?\d{3}\.?\d{3}-?\d{2})(?!\d)"),
        "[CPF_REMOVIDO]",
    ),
    (
        "phone",
        re.compile(
            r"\b(?:telefone|celular|fone)\s*[:=]\s*(?:\+?55\s*)?"
            r"(?:\(?\d{2}\)?\s*)?\d{4,5}[- ]?\d{4}\b",
            re.IGNORECASE,
        ),
        "[TELEFONE_REMOVIDO]",
    ),
    (
        "medical_record_id",
        re.compile(
            r"\bMRN\s*(?:[:=#]\s*)?[A-Z0-9][A-Z0-9/-]{3,}\b"
            r"|\b(?:prontu[aá]rio|registro\s+do\s+paciente|id\s+do\s+paciente)"
            r"\s*(?:[:=#]\s*|n[ºo.]?\s+)[A-Z0-9][A-Z0-9/-]{3,}\b",
            re.IGNORECASE,
        ),
        "[IDENTIFICADOR_REMOVIDO]",
    ),
    (
        "hospital_document_id",
        re.compile(
            r"\bHSP\s*[-:#/]?\s*\d{3,}(?:\s*[-/#.]\s*\d{2,4})?\b",
            re.IGNORECASE,
        ),
        "[DOCUMENTO_REMOVIDO]",
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:data\s+de\s+nascimento|nascimento|DOB)\s*[:=]\s*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            re.IGNORECASE,
        ),
        "[DATA_NASCIMENTO_REMOVIDA]",
    ),
    (
        "postal_code",
        re.compile(r"\bCEP\s*[:=]\s*\d{5}-?\d{3}\b", re.IGNORECASE),
        "[CEP_REMOVIDO]",
    ),
)
_PREPROCESSING_VERSION = "local-regex-v2"


def _reject_patient_identifier_fields(record: Mapping[str, Any]) -> None:
    if _contains_patient_record_fields(record):
        raise ValueError(
            "Internal records must not contain patient identifier fields or patient-record fields"
        )


def _contains_patient_record_fields(record: Mapping[str, Any]) -> bool:
    blocked_fields = {
        "".join(character for character in field if character.isalnum())
        for field in _FORBIDDEN_FIELDS | _PATIENT_RECORD_FIELDS
    }

    def inspect(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = "".join(
                    character for character in str(key).casefold() if character.isalnum()
                )
                if normalized_key in blocked_fields:
                    return True
                if str(key) == "preprocessing_redaction_counts":
                    continue
                if inspect(nested_value):
                    return True
        elif isinstance(value, list):
            for nested_value in value:
                if inspect(nested_value):
                    return True
        return False

    return inspect(record)


def validate_internal_training_record(record: Mapping[str, Any]) -> None:
    """Validate metadata attestations; this does not detect or remove personal data."""
    _reject_patient_identifier_fields(record)

    source_dataset = record.get("source_dataset")
    if not isinstance(source_dataset, str) or source_dataset not in INTERNAL_SOURCE_DATASETS:
        raise ValueError("Unsupported internal source dataset")
    if record.get("institutional_approval_status") != "approved":
        raise ValueError("Internal training source lacks institutional approval")
    if record.get("privacy_review_status") != "approved":
        raise ValueError("Internal training source lacks an approved privacy review")

    for field in _REQUIRED_TEXT_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Internal training record has invalid {field!r}")

    if record.get("contains_patient_data") is not False:
        raise ValueError("Internal training source must be reviewed as containing no patient data")
    if record.get("review_status") != "approved":
        raise ValueError("Internal training record has not completed content curation")
    if record.get("preprocessing_review_required") is not False:
        raise ValueError("Internal preprocessing output still requires human review")
    if record.get("synthetic") is True:
        raise ValueError("Internal training records must not be marked synthetic")

    redaction_counts = record.get("preprocessing_redaction_counts")
    if not isinstance(redaction_counts, dict) or any(
        not isinstance(category, str)
        or type(count) is not int
        or count < 0
        for category, count in redaction_counts.items()
    ):
        raise ValueError("Internal training record has invalid preprocessing_redaction_counts")

    for field in ("institutional_approved_at", "privacy_reviewed_at"):
        try:
            date.fromisoformat(record[field])
        except ValueError as exc:
            raise ValueError(f"Internal training record has invalid {field!r}") from exc


def _validate_preprocessing_candidate(record: Mapping[str, Any]) -> None:
    _reject_patient_identifier_fields(record)

    source_dataset = record.get("source_dataset")
    if not isinstance(source_dataset, str) or source_dataset not in INTERNAL_SOURCE_DATASETS:
        raise ValueError("Unsupported internal source dataset")
    candidate_fields = (
        "source_id",
        "source_group_id",
        "source_version",
        "question",
        "answer",
        "usage_rights_reference",
        "institutional_approval_status",
        "institutional_approval_reference",
        "institutional_approved_at",
    )
    for field in candidate_fields:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Internal preprocessing candidate has invalid {field!r}")
    if record["institutional_approval_status"] != "approved":
        raise ValueError("Internal preprocessing requires institutional approval")
    if record.get("contains_patient_data") is True:
        raise ValueError("Patient records are not eligible for internal knowledge training")
    if record.get("synthetic") is True:
        raise ValueError("Internal preprocessing input must not be marked synthetic")
    try:
        date.fromisoformat(record["institutional_approved_at"])
    except ValueError as exc:
        raise ValueError(
            "Internal preprocessing candidate has invalid 'institutional_approved_at'"
        ) from exc


def _redact_metadata(value: Any) -> tuple[Any, dict[str, int]]:
    if isinstance(value, str):
        return _normalize_and_redact(value)
    if isinstance(value, list):
        result: list[Any] = []
        total_counts = {category: 0 for category, _, _ in _REDACTION_PATTERNS}
        for item in value:
            redacted, counts = _redact_metadata(item)
            result.append(redacted)
            for category, count in counts.items():
                total_counts[category] += count
        return result, total_counts
    if isinstance(value, dict):
        result_dict: dict[str, Any] = {}
        total_counts = {category: 0 for category, _, _ in _REDACTION_PATTERNS}
        for key, item in value.items():
            redacted_key, key_counts = _normalize_and_redact(str(key))
            redacted_value, value_counts = _redact_metadata(item)
            result_dict[redacted_key] = redacted_value
            for category, count in key_counts.items():
                total_counts[category] += count
            for category, count in value_counts.items():
                total_counts[category] += count
        return result_dict, total_counts
    return value, {category: 0 for category, _, _ in _REDACTION_PATTERNS}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_text(record: Mapping[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Internal source record at line {line_number} has invalid {field!r}"
        )
    return value


def _normalize_source_record(
    record: dict[str, Any],
    *,
    line_number: int,
    file_version: str,
    institutional_approval_reference: str | None,
    institutional_approved_at: str | None,
    usage_rights_reference: str | None,
) -> dict[str, Any]:
    if "question" in record and "answer" in record:
        source_dataset = record.get("source_dataset")
        question = _source_text(record, "question", line_number)
        answer = _source_text(record, "answer", line_number)
    elif "pergunta" in record and "resposta" in record:
        source_dataset = "hospital_faq"
        question = _source_text(record, "pergunta", line_number)
        answer = _source_text(record, "resposta", line_number)
    elif all(field in record for field in ("titulo", "conteudo", "acao_ia", "risco")):
        source_dataset = "hospital_protocol"
        title = _source_text(record, "titulo", line_number)
        question = f"Qual orientação está registrada em {title}?"
        answer = "\n".join(
            (
                f"Conteúdo: {_source_text(record, 'conteudo', line_number)}",
                f"Ação do assistente: {_source_text(record, 'acao_ia', line_number)}",
                f"Risco registrado: {_source_text(record, 'risco', line_number)}",
            )
        )
    elif all(field in record for field in ("contexto_uso", "texto", "variaveis")):
        source_dataset = "hospital_template"
        question = f"Qual template se aplica ao contexto: {_source_text(record, 'contexto_uso', line_number)}?"
        template_text = _source_text(record, "texto", line_number)
        variables = json.dumps(record["variaveis"], ensure_ascii=False, sort_keys=True)
        answer = f"Modelo: {template_text}\nVariáveis: {variables}"
    else:
        raise ValueError(f"Unsupported internal source schema at line {line_number}")

    source_id_value = record.get("source_id", record.get("id"))
    if (
        isinstance(source_id_value, bool)
        or not isinstance(source_id_value, (str, int))
        or not str(source_id_value).strip()
    ):
        raise ValueError(f"Internal source record at line {line_number} has invalid source ID")
    source_id = str(source_id_value).strip()
    _, source_id_redactions = _normalize_and_redact(source_id)
    if any(source_id_redactions.values()):
        raise ValueError(f"Internal source ID at line {line_number} resembles personal data")
    source_group_value = record.get("source_group_id", source_id)
    if (
        isinstance(source_group_value, bool)
        or not isinstance(source_group_value, (str, int))
        or not str(source_group_value).strip()
    ):
        raise ValueError(
            f"Internal source record at line {line_number} has invalid source group ID"
        )
    source_group_id = str(source_group_value).strip()
    _, group_id_redactions = _normalize_and_redact(source_group_id)
    if any(group_id_redactions.values()):
        raise ValueError(
            f"Internal source group ID at line {line_number} resembles personal data"
        )

    approval_reference = (
        record.get("institutional_approval_reference") or institutional_approval_reference
    )
    approved_at = record.get("institutional_approved_at") or institutional_approved_at
    rights_reference = record.get("usage_rights_reference") or usage_rights_reference
    source_version = record.get("source_version") or file_version
    source_metadata = {
        key: value
        for key, value in record.items()
        if key
        in {
            "categoria",
            "especialidade",
            "tags",
            "titulo",
            "risco",
            "contexto_uso",
            "variaveis",
        }
    }
    if "source_metadata" in record:
        if not isinstance(record["source_metadata"], dict):
            raise ValueError(f"Invalid source metadata at line {line_number}")
        source_metadata.update(record["source_metadata"])

    normalized = {
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_group_id": source_group_id,
        "source_version": source_version,
        "question": question,
        "answer": answer,
        "source_metadata": source_metadata,
        "usage_rights_reference": rights_reference,
        "institutional_approval_status": record.get(
            "institutional_approval_status", "approved"
        ),
        "institutional_approval_reference": approval_reference,
        "institutional_approved_at": approved_at,
        "contains_patient_data": record.get("contains_patient_data"),
    }

    if "source_dataset" in record and (
        not isinstance(source_dataset, str)
        or source_dataset not in INTERNAL_SOURCE_DATASETS
    ):
        raise ValueError(f"Unsupported internal source dataset at line {line_number}")
    if record.get("synthetic") is True:
        raise ValueError(f"Synthetic records cannot be imported as institutional data at line {line_number}")
    return normalized


def _normalize_and_redact(text: str) -> tuple[str, dict[str, int]]:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    counts: dict[str, int] = {}
    for category, pattern, replacement in _REDACTION_PATTERNS:
        normalized, count = pattern.subn(replacement, normalized)
        counts[category] = count
    return normalized, counts


def normalize_and_redact_clinical_text(text: str) -> tuple[str, dict[str, int]]:
    """Normalize text and redact common identifier patterns; not full anonymization."""
    if not isinstance(text, str):
        raise ValueError("Clinical text must be a string")
    return _normalize_and_redact(text)


def preprocess_approved_internal_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    institutional_approval_reference: str | None = None,
    institutional_approved_at: str | None = None,
    usage_rights_reference: str | None = None,
) -> dict[str, Any]:
    """Normalize and redact common identifiers, leaving all records review-pending.

    This local regex pass is not complete anonymization. Names, addresses, rare
    facts, and identifiers in unrecognized formats require human review.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Internal preprocessing input does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Preprocessing output must not overwrite the source file")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite preprocessing output: {output_path}")

    file_version = f"sha256:{_file_sha256(input_path)}"
    output_records: list[str] = []
    total_redactions = {category: 0 for category, _, _ in _REDACTION_PATTERNS}
    seen_sources: set[tuple[str, str]] = set()
    excluded_patient_records = 0
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid internal preprocessing JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"Internal preprocessing line {line_number} must be an object")
        if _contains_patient_record_fields(record):
            excluded_patient_records += 1
            continue

        try:
            normalized = _normalize_source_record(
                record,
                line_number=line_number,
                file_version=file_version,
                institutional_approval_reference=institutional_approval_reference,
                institutional_approved_at=institutional_approved_at,
                usage_rights_reference=usage_rights_reference,
            )
            _validate_preprocessing_candidate(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid internal preprocessing candidate at line {line_number}: {exc}"
            ) from exc

        source_id = normalized["source_id"]
        source_dataset = normalized["source_dataset"]
        source_key = (source_dataset, source_id)
        if source_key in seen_sources:
            raise ValueError(f"Duplicate internal source at line {line_number}")
        seen_sources.add(source_key)

        question, question_counts = _normalize_and_redact(normalized["question"])
        answer, answer_counts = _normalize_and_redact(normalized["answer"])
        source_metadata, metadata_counts = _redact_metadata(normalized["source_metadata"])
        redaction_counts = {
            category: (
                question_counts[category]
                + answer_counts[category]
                + metadata_counts[category]
            )
            for category in total_redactions
        }
        for category, count in redaction_counts.items():
            total_redactions[category] += count

        processed = dict(normalized)
        processed["question"] = question
        processed["answer"] = answer
        processed["source_metadata"] = source_metadata
        processed["preprocessing_version"] = _PREPROCESSING_VERSION
        processed["preprocessing_redaction_counts"] = redaction_counts
        processed["preprocessing_review_required"] = True
        processed["review_status"] = "pending_privacy_review"
        processed["privacy_review_status"] = "pending"
        processed["privacy_review_reference"] = None
        processed["privacy_reviewed_at"] = None
        processed["contains_patient_data"] = None
        output_records.append(json.dumps(processed, ensure_ascii=False))

    if not output_records:
        raise ValueError(f"Internal preprocessing input is empty: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{record}\n" for record in output_records),
        encoding="utf-8",
    )
    return {
        "output_path": output_path,
        "records_processed": len(output_records),
        "records_excluded_patient_records": excluded_patient_records,
        "redactions_by_category": total_redactions,
        "human_review_required": True,
        "source_version": file_version,
    }


def load_approved_internal_jsonl(path: Path) -> DatasetLoadResult:
    """Load curated Q&A records with explicit institutional and privacy attestations.

    This loader validates the review metadata only. It does not anonymize, scan,
    or certify the question and answer text.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Internal dataset does not exist: {path}")

    records: list[QARecord] = []
    seen_sources: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid internal JSONL at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Internal JSONL line {line_number} must be an object")

        try:
            validate_internal_training_record(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid internal record at line {line_number}: {exc}") from exc

        source_key = (payload["source_dataset"], payload["source_id"])
        if source_key in seen_sources:
            raise ValueError(f"Duplicate internal source at line {line_number}")
        seen_sources.add(source_key)
        source_metadata = payload.get("source_metadata")
        if source_metadata is not None and not isinstance(source_metadata, dict):
            raise ValueError(f"Invalid source metadata at line {line_number}")

        records.append(
            QARecord(
                id=f"{payload['source_dataset']}:{payload['source_id']}",
                question=payload["question"],
                answer=payload["answer"],
                source_dataset=payload["source_dataset"],
                source_id=payload["source_id"],
                source_group_id=payload["source_group_id"],
                source_version=payload["source_version"],
                source_metadata=payload.get("source_metadata"),
                license=None,
                license_note=payload["usage_rights_reference"],
                usage_rights_reference=payload["usage_rights_reference"],
                institutional_approval_status=payload["institutional_approval_status"],
                institutional_approval_reference=payload["institutional_approval_reference"],
                institutional_approved_at=payload["institutional_approved_at"],
                privacy_review_status=payload["privacy_review_status"],
                privacy_review_reference=payload["privacy_review_reference"],
                privacy_reviewed_at=payload["privacy_reviewed_at"],
                contains_patient_data=False,
                review_status=payload["review_status"],
                preprocessing_version=payload["preprocessing_version"],
                preprocessing_redaction_counts=payload["preprocessing_redaction_counts"],
                preprocessing_review_required=payload["preprocessing_review_required"],
            )
        )

    if not records:
        raise ValueError(f"Internal dataset is empty: {path}")
    return DatasetLoadResult(records=tuple(records))


def export_training_splits(splits: DatasetSplits, output_dir: Path) -> tuple[Path, ...]:
    """Write normalized JSONL splits; callers should use ignored local storage."""
    split_records = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    empty_splits = [name for name, records in split_records.items() if not records]
    if empty_splits:
        raise ValueError(f"Training splits must not be empty: {', '.join(empty_splits)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{name}.jsonl" for name in split_records)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing training split files; choose an empty "
            "output directory"
        )

    payloads: list[str] = []
    for split_name, records in split_records.items():
        lines: list[str] = []
        for record in records:
            serialized = asdict(replace(record, split=split_name))
            if record.source_dataset in INTERNAL_SOURCE_DATASETS:
                validate_internal_training_record(serialized)
            if record.source_dataset == "synthetic":
                serialized["synthetic"] = True
            lines.append(json.dumps(serialized, ensure_ascii=False))
        payloads.append("".join(f"{line}\n" for line in lines))

    for path, content in zip(paths, payloads):
        path.write_text(content, encoding="utf-8")
    return paths
