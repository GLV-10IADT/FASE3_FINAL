from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from medical_assistant.datasets.internal import normalize_and_redact_clinical_text
from jsonschema import Draft7Validator, FormatChecker


_REQUIRED_SCHEMA_FIELDS = (
    "protocolos_clinicos",
    "prontuarios",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_validated_source(
    input_path: Path,
    schema_path: Path,
) -> tuple[dict[str, Any], str, str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Clinical data file does not exist: {input_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Clinical data schema does not exist: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    source = json.loads(input_path.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema, format_checker=FormatChecker())
    error = next(iter(validator.iter_errors(source)), None)
    if error is not None:
        error_path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"Clinical data does not match the supplied schema at {error_path}"
        )
    if not isinstance(source, dict) or any(
        not isinstance(source.get(field), list)
        for field in _REQUIRED_SCHEMA_FIELDS
    ):
        raise ValueError("Clinical data must contain protocol and patient-record lists")
    return source, _sha256(input_path), _sha256(schema_path)


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Clinical import has invalid {field!r}")
    return value.strip()


def _validate_date(value: str, *, field: str) -> str:
    normalized = _require_text(value, field=field)
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Clinical import has invalid {field!r}") from exc
    return normalized


def _protocol_text(protocol: dict[str, Any]) -> str:
    sections = [
        f"Especialidade: {protocol['especialidade']}",
        "CID-10 relacionados: "
        + (", ".join(protocol["cid10_relacionados"]) or "não informado"),
        "Critérios de inclusão: "
        + ("; ".join(protocol["criterios_inclusao"]) or "não informado"),
        "Etapas: "
        + json.dumps(protocol["etapas"], ensure_ascii=False, sort_keys=True),
        "Prescrição padrão: "
        + json.dumps(
            protocol["prescricao_padrao"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    ]
    return f"Protocolo: {protocol['titulo']}\n" + "\n".join(sections)


def _write_jsonl(output_path: Path, records: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            temporary.write("".join(f"{record}\n" for record in records))
        os.rename(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def convert_authorized_protocols(
    input_path: Path,
    schema_path: Path,
    output_path: Path,
    *,
    institutional_approval_reference: str,
    institutional_approved_at: str,
    privacy_review_reference: str,
    privacy_reviewed_at: str,
) -> dict[str, int | str | Path]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite clinical import: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Clinical import output must not overwrite its source")

    approval_reference = _require_text(
        institutional_approval_reference,
        field="institutional_approval_reference",
    )
    approval_date = _validate_date(
        institutional_approved_at,
        field="institutional_approved_at",
    )
    privacy_reference = _require_text(
        privacy_review_reference,
        field="privacy_review_reference",
    )
    privacy_date = _validate_date(
        privacy_reviewed_at,
        field="privacy_reviewed_at",
    )

    source, input_hash, schema_hash = _load_validated_source(
        input_path,
        schema_path,
    )

    protocols = source["protocolos_clinicos"]
    patients = source["prontuarios"]
    output_records: list[str] = []
    seen_ids: set[str] = set()
    for protocol in protocols:
        source_id = _require_text(protocol.get("protocolo_id"), field="protocolo_id")
        if source_id in seen_ids:
            raise ValueError("Clinical data contains duplicate protocol IDs")
        seen_ids.add(source_id)
        title = _require_text(protocol.get("titulo"), field="titulo")
        version = _require_text(protocol.get("versao"), field="versao")
        specialty = _require_text(protocol.get("especialidade"), field="especialidade")
        source_document = {
            "id": f"hospital-protocol:{source_id}",
            "text": _protocol_text(protocol),
            "source": f"{input_path.as_posix()}#{source_id}",
            "title": title,
            "section": "clinical_protocol",
            "source_dataset": "hospital_protocol",
            "source_id": source_id,
            "version": version,
            "license": "internal",
            "license_note": (
                f"Institutional approval {approval_reference}; "
                f"privacy review {privacy_reference}."
            ),
            "institutional_approval_status": "approved",
            "institutional_approval_reference": approval_reference,
            "institutional_approved_at": approval_date,
            "privacy_review_status": "approved",
            "privacy_review_reference": privacy_reference,
            "privacy_reviewed_at": privacy_date,
            "contains_patient_data": False,
            "training_eligible": False,
            "approval_attestation_source": "user_provided",
            "source_file_sha256": input_hash,
            "schema_file_sha256": schema_hash,
            "source_metadata": {
                "especialidade": specialty,
                "cid10_relacionados": protocol["cid10_relacionados"],
            },
        }
        output_records.append(json.dumps(source_document, ensure_ascii=False))

    if not output_records:
        raise ValueError("Clinical data contains no protocols to import")

    _write_jsonl(output_path, output_records)

    return {
        "output_path": output_path,
        "protocols_converted": len(output_records),
        "patient_records_not_exported": len(patients),
        "source_file_sha256": input_hash,
        "schema_file_sha256": schema_hash,
    }


def _redact_patient_text(
    value: Any,
    *,
    field: str,
    counts: dict[str, int],
) -> str:
    text = _require_text(value, field=field)
    normalized, redactions = normalize_and_redact_clinical_text(text)
    for category, count in redactions.items():
        counts[category] = counts.get(category, 0) + count
    return normalized


def _age_band(age: Any) -> str:
    if type(age) is not int or age < 0:
        raise ValueError("Clinical patient record has an invalid age")
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def _normalized_exam_status(value: Any, *, case_id: str) -> str:
    status = _require_text(value, field="exam_status").casefold()
    canonical = "".join(
        character
        for character in unicodedata.normalize("NFKD", status)
        if not unicodedata.combining(character)
    )
    if canonical in {"concluido", "completed"}:
        return "completed"
    if canonical in {"em analise", "pendente", "pending"}:
        return "pending"
    raise ValueError(f"Patient record {case_id!r} has an unsupported exam status")


def convert_authorized_patient_records(
    input_path: Path,
    schema_path: Path,
    output_path: Path,
    *,
    institutional_approval_reference: str,
    institutional_approved_at: str,
    privacy_review_reference: str,
    privacy_reviewed_at: str,
) -> dict[str, int | str | Path]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite patient import: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Patient import output must not overwrite its source")
    approval_reference = _require_text(
        institutional_approval_reference,
        field="institutional_approval_reference",
    )
    approval_date = _validate_date(
        institutional_approved_at,
        field="institutional_approved_at",
    )
    privacy_reference = _require_text(
        privacy_review_reference,
        field="privacy_review_reference",
    )
    privacy_date = _validate_date(
        privacy_reviewed_at,
        field="privacy_reviewed_at",
    )
    source, input_hash, schema_hash = _load_validated_source(
        input_path,
        schema_path,
    )

    output_records: list[str] = []
    total_redactions: dict[str, int] = {}
    for patient_number, raw_patient in enumerate(source["prontuarios"], 1):
        case_id = f"CASE-{patient_number:03d}"
        patient = raw_patient["paciente"]
        counts: dict[str, int] = {}
        profile = {
            "age_band": _age_band(patient["idade"]),
            "biological_sex": _redact_patient_text(
                patient["sexo_biologico"],
                field="biological_sex",
                counts=counts,
            ),
            "blood_type": _redact_patient_text(
                patient["tipo_sanguineo"],
                field="blood_type",
                counts=counts,
            ),
            "allergies": [
                _redact_patient_text(value, field="allergy", counts=counts)
                for value in patient["alergias"]
            ],
            "comorbidities": [
                _redact_patient_text(value, field="comorbidity", counts=counts)
                for value in patient["comorbidades_previas"]
            ],
        }

        encounters: list[dict[str, Any]] = []
        encounter_number = 0
        exam_number = 0
        for raw_encounter in raw_patient["atendimentos"]:
            encounter_number += 1
            encounter_id = f"{case_id}-ENCOUNTER-{encounter_number:03d}"
            provider = raw_encounter["profissional_responsavel"]
            soap = {
                field: _redact_patient_text(
                    raw_encounter["evolucao_soap"][field],
                    field=f"soap_{field}",
                    counts=counts,
                )
                for field in ("subjetivo", "objetivo", "avaliacao", "plano")
            }
            exams: list[dict[str, str]] = []
            for raw_exam in raw_encounter["exames_solicitados"]:
                exam_number += 1
                exams.append(
                    {
                        "exam_id": f"{case_id}-EXAM-{exam_number:03d}",
                        "tuss": _redact_patient_text(
                            raw_exam["tuss"],
                            field="exam_tuss",
                            counts=counts,
                        ),
                        "name": _redact_patient_text(
                            raw_exam["nome"],
                            field="exam_name",
                            counts=counts,
                        ),
                        "result": _redact_patient_text(
                            raw_exam["resultado"],
                            field="exam_result",
                            counts=counts,
                        ),
                        "status": _normalized_exam_status(
                            raw_exam["status"],
                            case_id=case_id,
                        ),
                    }
                )
            medications = [
                {
                    "name": _redact_patient_text(
                        prescription["medicamento"],
                        field="medication",
                        counts=counts,
                    ),
                    "route": _redact_patient_text(
                        prescription["via"],
                        field="medication_route",
                        counts=counts,
                    ),
                    "posology": _redact_patient_text(
                        prescription["posologia"],
                        field="medication_posology",
                        counts=counts,
                    ),
                    "administration_time": _redact_patient_text(
                        prescription["horario_administracao"],
                        field="medication_administration_time",
                        counts=counts,
                    ),
                }
                for prescription in raw_encounter["prescricoes_ativas"]
            ]
            vitals = {
                "blood_pressure": _redact_patient_text(
                    raw_encounter["triagem_sinais_vitais"]["pa_mmhg"],
                    field="blood_pressure",
                    counts=counts,
                ),
                "heart_rate_bpm": raw_encounter["triagem_sinais_vitais"]["fc_bpm"],
                "respiratory_rate_rpm": raw_encounter["triagem_sinais_vitais"]["fr_irpm"],
                "temperature_c": raw_encounter["triagem_sinais_vitais"]["temperatura_c"],
                "oxygen_saturation_pct": raw_encounter["triagem_sinais_vitais"]["spo2_pct"],
                "glasgow": raw_encounter["triagem_sinais_vitais"]["glasgow"],
                "glucose_mg_dl": raw_encounter["triagem_sinais_vitais"]["glicemia_mg_dl"],
            }
            encounters.append(
                {
                    "encounter_id": encounter_id,
                    "status": _redact_patient_text(
                        raw_encounter["status"],
                        field="encounter_status",
                        counts=counts,
                    ),
                    "classification": _redact_patient_text(
                        raw_encounter["classificacao_manchester"],
                        field="clinical_classification",
                        counts=counts,
                    ),
                    "protocol_id": _redact_patient_text(
                        raw_encounter["protocolo_aplicado_id"],
                        field="protocol_id",
                        counts=counts,
                    ),
                    "professional_specialty": _redact_patient_text(
                        provider["especialidade"],
                        field="professional_specialty",
                        counts=counts,
                    ),
                    "vitals": vitals,
                    "soap": soap,
                    "exams": exams,
                    "medications": medications,
                }
            )
        for category, count in counts.items():
            total_redactions[category] = total_redactions.get(category, 0) + count
        patient_document = {
            "id": f"hospital-patient:{case_id}",
            "source": "local:authorized_patient_record",
            "title": f"Prontuário local {case_id}",
            "section": "selected_patient_record",
            "source_dataset": "hospital_patient_record",
            "source_id": case_id,
            "version": "schema-v1",
            "license": "internal",
            "license_note": (
                f"Institutional approval {approval_reference}; "
                f"privacy review {privacy_reference}."
            ),
            "patient_case_id": case_id,
            "patient_profile": profile,
            "encounters": encounters,
            "institutional_approval_status": "approved",
            "institutional_approval_reference": approval_reference,
            "institutional_approved_at": approval_date,
            "privacy_review_status": "approved",
            "privacy_review_reference": privacy_reference,
            "privacy_reviewed_at": privacy_date,
            "contains_patient_data": True,
            "training_eligible": False,
            "approval_attestation_source": "user_provided",
            "source_file_sha256": input_hash,
            "schema_file_sha256": schema_hash,
            "preprocessing_redaction_counts": counts,
        }
        output_records.append(json.dumps(patient_document, ensure_ascii=False))

    if not output_records:
        raise ValueError("Clinical data contains no patient records to import")
    _write_jsonl(output_path, output_records)
    return {
        "output_path": output_path,
        "patient_records_converted": len(output_records),
        "redactions": sum(total_redactions.values()),
        "source_file_sha256": input_hash,
        "schema_file_sha256": schema_hash,
    }
