import json
import re
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from medical_assistant.patient_data.store import (
    AccessContext,
    AccessDeniedError,
    ExamStatus,
    PatientEncounter,
    PatientExam,
    PatientNotFoundError,
    PatientRecord,
    PatientRecordItem,
    PatientRole,
)

_CASE_ID_PATTERN = re.compile(r"CASE-\d{3}")
_PROFILE_FIELDS = {
    "age_band",
    "biological_sex",
    "blood_type",
    "allergies",
    "comorbidities",
}
_ENCOUNTER_FIELDS = {
    "encounter_id",
    "status",
    "classification",
    "protocol_id",
    "professional_specialty",
    "vitals",
    "soap",
    "exams",
    "medications",
}
_VITAL_FIELDS = {
    "blood_pressure",
    "heart_rate_bpm",
    "respiratory_rate_rpm",
    "temperature_c",
    "oxygen_saturation_pct",
    "glasgow",
    "glucose_mg_dl",
}
_SOAP_FIELDS = {"subjetivo", "objetivo", "avaliacao", "plano"}
_EXAM_FIELDS = {"exam_id", "tuss", "name", "result", "status"}
_MEDICATION_FIELDS = {"name", "route", "posology", "administration_time"}


def _require_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Authorized patient record has invalid {context}")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Authorized patient record has invalid {context}")
    return value.strip()


def _require_string_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Authorized patient record has invalid {context}")
    return tuple(_require_string(item, context) for item in value)


def _profile_summary(profile: dict[str, Any]) -> str:
    lines = [
        f"Faixa etária: {profile['age_band']}.",
        f"Sexo biológico: {profile['biological_sex']}.",
        f"Tipo sanguíneo: {profile['blood_type']}.",
        "Alergias: " + (", ".join(profile["allergies"]) or "não informado") + ".",
        "Comorbidades prévias: "
        + (", ".join(profile["comorbidities"]) or "não informado")
        + ".",
    ]
    return "\n".join(lines)


def _encounter_summary(encounter: dict[str, Any]) -> str:
    vitals = encounter["vitals"]
    soap = encounter["soap"]
    lines = [
        f"Atendimento: {encounter['status']}.",
        f"Classificação: {encounter['classification']}.",
        f"Protocolo aplicado: {encounter['protocol_id']}.",
        f"Especialidade responsável: {encounter['professional_specialty']}.",
        "Sinais vitais: "
        + json.dumps(vitals, ensure_ascii=False, sort_keys=True)
        + ".",
        "Evolução clínica: "
        + json.dumps(soap, ensure_ascii=False, sort_keys=True)
        + ".",
    ]
    for exam in encounter["exams"]:
        lines.append(
            "Exame: "
            f"{exam['name']}; TUSS {exam['tuss']}; "
            f"resultado: {exam['result']}; status: {exam['status']}."
        )
    for medication in encounter["medications"]:
        lines.append(
            "Prescrição ativa registrada: "
            f"{medication['name']}, via {medication['route']}, "
            f"posologia {medication['posology']}, "
            f"horário {medication['administration_time']}."
        )
    return "\n".join(lines)


class AuthorizedPatientRecordStore:
    """Reads pseudonymized, locally approved patient records with an access gate."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Authorized patient corpus does not exist: {path}")
        patients: dict[str, PatientRecord] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid authorized patient JSONL at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Authorized patient record at line {line_number} must be an object"
                )
            patient_id = _require_string(record.get("patient_case_id"), "case id")
            if (
                not _CASE_ID_PATTERN.fullmatch(patient_id)
                or record.get("id") != f"hospital-patient:{patient_id}"
                or record.get("source_id") != patient_id
                or record.get("source") != "local:authorized_patient_record"
                or record.get("source_dataset") != "hospital_patient_record"
            ):
                raise ValueError(
                    f"Authorized patient metadata is invalid at line {line_number}"
                )
            if patient_id in patients:
                raise ValueError(f"Duplicate authorized patient at line {line_number}")
            if (
                record.get("institutional_approval_status") != "approved"
                or record.get("privacy_review_status") != "approved"
                or record.get("contains_patient_data") is not True
                or record.get("training_eligible") is not False
                or record.get("approval_attestation_source") != "user_provided"
            ):
                raise ValueError(
                    f"Patient approval metadata is invalid at line {line_number}"
                )
            for date_field in ("institutional_approved_at", "privacy_reviewed_at"):
                date_value = _require_string(record.get(date_field), date_field)
                try:
                    date.fromisoformat(date_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Patient approval metadata has an invalid {date_field!r}"
                    ) from exc
            _require_string(record.get("institutional_approval_reference"), "approval reference")
            _require_string(record.get("privacy_review_reference"), "privacy reference")

            profile = _require_object(
                record.get("patient_profile"),
                _PROFILE_FIELDS,
                "patient profile",
            )
            profile["age_band"] = _require_string(profile.get("age_band"), "age band")
            profile["biological_sex"] = _require_string(
                profile.get("biological_sex"),
                "biological sex",
            )
            profile["blood_type"] = _require_string(
                profile.get("blood_type"),
                "blood type",
            )
            profile["allergies"] = _require_string_list(
                profile.get("allergies"),
                "allergies",
            )
            profile["comorbidities"] = _require_string_list(
                profile.get("comorbidities"),
                "comorbidities",
            )
            raw_encounters = record.get("encounters")
            if not isinstance(raw_encounters, list):
                raise ValueError(
                    f"Authorized patient record {patient_id!r} has invalid encounters"
                )

            encounters: list[PatientEncounter] = []
            record_items: list[PatientRecordItem] = [
                PatientRecordItem(
                    record_id=patient_id,
                    record_type="patient_profile",
                    content=_profile_summary(profile),
                )
            ]
            for raw_encounter in raw_encounters:
                encounter = _require_object(
                    raw_encounter,
                    _ENCOUNTER_FIELDS,
                    "encounter",
                )
                encounter_id = _require_string(
                    encounter.get("encounter_id"),
                    "encounter id",
                )
                if not encounter_id.startswith(f"{patient_id}-ENCOUNTER-"):
                    raise ValueError(
                        f"Authorized patient record {patient_id!r} has invalid encounter metadata"
                    )
                for field in (
                    "status",
                    "classification",
                    "protocol_id",
                    "professional_specialty",
                ):
                    encounter[field] = _require_string(encounter.get(field), field)
                vitals = _require_object(
                    encounter.get("vitals"),
                    _VITAL_FIELDS,
                    "vitals",
                )
                soap = _require_object(
                    encounter.get("soap"),
                    _SOAP_FIELDS,
                    "clinical note",
                )
                for field, value in (*vitals.items(), *soap.items()):
                    if not isinstance(value, (str, int, float)):
                        raise ValueError(
                            f"Authorized patient record {patient_id!r} has invalid {field!r}"
                        )
                raw_exams = encounter.get("exams")
                raw_medications = encounter.get("medications")
                if not isinstance(raw_exams, list) or not isinstance(raw_medications, list):
                    raise ValueError(
                        f"Authorized patient record {patient_id!r} has invalid encounter details"
                    )
                exams: list[PatientExam] = []
                for raw_exam in raw_exams:
                    exam = _require_object(raw_exam, _EXAM_FIELDS, "exam")
                    exam_id = _require_string(exam.get("exam_id"), "exam id")
                    if not exam_id.startswith(f"{patient_id}-EXAM-"):
                        raise ValueError(
                            f"Authorized patient record {patient_id!r} has invalid exam metadata"
                        )
                    display_name = _require_string(exam.get("name"), "exam name")
                    try:
                        status = ExamStatus(_require_string(exam.get("status"), "exam status"))
                    except ValueError as exc:
                        raise ValueError(
                            f"Authorized patient record {patient_id!r} has invalid exam status"
                        ) from exc
                    _require_string(exam.get("tuss"), "exam code")
                    _require_string(exam.get("result"), "exam result")
                    exams.append(
                        PatientExam(
                            exam_id=exam_id,
                            display_name=display_name,
                            status=status,
                        )
                    )
                for raw_medication in raw_medications:
                    medication = _require_object(
                        raw_medication,
                        _MEDICATION_FIELDS,
                        "medication",
                    )
                    for field in _MEDICATION_FIELDS:
                        _require_string(medication.get(field), field)
                encounters.append(
                    PatientEncounter(
                        encounter_id=encounter_id,
                        status=encounter["status"],
                        exams=tuple(exams),
                    )
                )
                record_items.append(
                    PatientRecordItem(
                        record_id=encounter_id,
                        record_type="clinical_encounter",
                        content=_encounter_summary(encounter),
                    )
                )

            redactions = record.get("preprocessing_redaction_counts")
            if not isinstance(redactions, dict) or any(
                not isinstance(category, str)
                or type(count) is not int
                or count < 0
                for category, count in redactions.items()
            ):
                raise ValueError(
                    f"Authorized patient record {patient_id!r} has invalid redaction metadata"
                )
            patients[patient_id] = PatientRecord(
                patient_id=patient_id,
                display_label=f"Prontuário local {patient_id}",
                records=tuple(record_items),
                encounters=tuple(encounters),
                source="local:authorized_patient_record",
            )
        if not patients:
            raise ValueError(f"Authorized patient corpus is empty: {path}")
        self._patients: Mapping[str, PatientRecord] = MappingProxyType(patients)

    @property
    def available_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._patients))

    def read_patient(
        self,
        patient_id: str,
        access_context: AccessContext,
    ) -> PatientRecord:
        if not isinstance(access_context, AccessContext):
            raise AccessDeniedError("A trusted access context is required")
        if (
            access_context.role is not PatientRole.CLINICIAN
            or patient_id not in access_context.authorized_patient_ids
        ):
            raise AccessDeniedError("The caller is not authorized to read this patient")
        try:
            return self._patients[patient_id]
        except KeyError as exc:
            raise PatientNotFoundError("Patient record was not found") from exc
