import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol


class PatientRole(StrEnum):
    CLINICIAN = "clinician"
    GUEST = "guest"


class ExamStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class AccessDeniedError(PermissionError):
    """Raised when a caller is not authorized to read a patient fixture."""


class PatientNotFoundError(LookupError):
    """Raised when an authorized caller requests an unknown synthetic patient."""


@dataclass(frozen=True)
class AccessContext:
    user_id: str
    role: PatientRole
    authorized_patient_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(self.role, PatientRole):
            raise ValueError("role must be a PatientRole value")
        if not isinstance(self.authorized_patient_ids, frozenset):
            object.__setattr__(
                self, "authorized_patient_ids", frozenset(self.authorized_patient_ids)
            )
        if any(not isinstance(patient_id, str) or not patient_id for patient_id in self.authorized_patient_ids):
            raise ValueError("authorized_patient_ids must contain non-empty strings")


@dataclass(frozen=True)
class PatientRecordItem:
    record_id: str
    record_type: str
    content: str


@dataclass(frozen=True)
class PatientExam:
    exam_id: str
    display_name: str
    status: ExamStatus


@dataclass(frozen=True)
class PatientEncounter:
    encounter_id: str
    status: str
    exams: tuple[PatientExam, ...]


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    display_label: str
    records: tuple[PatientRecordItem, ...]
    encounters: tuple[PatientEncounter, ...] = ()
    source: str = "data/synthetic/patients.json"


class PatientStore(Protocol):
    @property
    def available_patient_ids(self) -> tuple[str, ...]: ...

    def read_patient(
        self,
        patient_id: str,
        access_context: AccessContext,
    ) -> PatientRecord: ...


class CompositePatientStore:
    """Routes a patient read to one of several independently gated stores."""

    def __init__(self, stores: tuple[PatientStore, ...] | list[PatientStore]) -> None:
        self._stores_by_patient: dict[str, PatientStore] = {}
        for store in stores:
            for patient_id in store.available_patient_ids:
                if patient_id in self._stores_by_patient:
                    raise ValueError("Patient IDs must be unique across stores")
                self._stores_by_patient[patient_id] = store

    @property
    def available_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._stores_by_patient))

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
            store = self._stores_by_patient[patient_id]
        except KeyError as exc:
            raise PatientNotFoundError("Patient record was not found") from exc
        return store.read_patient(patient_id, access_context)


class SyntheticPatientStore:
    """Loads immutable synthetic records and exposes an authorization-gated read."""

    def __init__(self, path: Path) -> None:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise ValueError("Synthetic patient data must contain a list")

        patients: dict[str, PatientRecord] = {}
        for item in raw_data:
            if not isinstance(item, dict) or item.get("synthetic") is not True:
                raise ValueError("Every patient fixture must be explicitly synthetic")

            patient_id = item.get("patient_id")
            display_label = item.get("display_label")
            if not isinstance(patient_id, str) or not patient_id.startswith("PATIENT-SYNTH-"):
                raise ValueError("Synthetic patient fixture has an invalid patient_id")
            if not isinstance(display_label, str) or not display_label.strip():
                raise ValueError(f"Synthetic patient {patient_id!r} has no display label")
            raw_records = item.get("records")
            if not isinstance(raw_records, list):
                raise ValueError(f"Synthetic patient {patient_id!r} has invalid records")

            records: list[PatientRecordItem] = []
            seen_record_ids: set[str] = set()
            for raw_record in raw_records:
                if not isinstance(raw_record, dict):
                    raise ValueError(f"Synthetic patient {patient_id!r} has an invalid record")
                record_id = raw_record.get("record_id")
                record_type = raw_record.get("record_type")
                content = raw_record.get("content")
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (record_id, record_type, content)
                ):
                    raise ValueError(f"Synthetic patient {patient_id!r} has incomplete record data")
                if record_id in seen_record_ids:
                    raise ValueError(
                        f"Synthetic patient {patient_id!r} has duplicate record ID {record_id!r}"
                    )
                seen_record_ids.add(record_id)
                records.append(
                    PatientRecordItem(
                        record_id=record_id,
                        record_type=record_type,
                        content=content,
                    )
                )

            raw_encounters = item.get("encounters", [])
            if not isinstance(raw_encounters, list):
                raise ValueError(f"Synthetic patient {patient_id!r} has invalid encounters")
            encounters: list[PatientEncounter] = []
            seen_encounter_ids: set[str] = set()
            seen_exam_ids: set[str] = set()
            for raw_encounter in raw_encounters:
                if not isinstance(raw_encounter, dict):
                    raise ValueError(
                        f"Synthetic patient {patient_id!r} has an invalid encounter"
                    )
                encounter_id = raw_encounter.get("encounter_id")
                encounter_status = raw_encounter.get("status")
                raw_exams = raw_encounter.get("exams")
                if (
                    not isinstance(encounter_id, str)
                    or not encounter_id.startswith("ENCOUNTER-SYNTH-")
                    or not isinstance(encounter_status, str)
                    or not encounter_status.strip()
                    or not isinstance(raw_exams, list)
                ):
                    raise ValueError(
                        f"Synthetic patient {patient_id!r} has incomplete encounter data"
                    )
                if encounter_id in seen_encounter_ids:
                    raise ValueError(
                        f"Synthetic patient {patient_id!r} has duplicate encounter ID "
                        f"{encounter_id!r}"
                    )
                seen_encounter_ids.add(encounter_id)
                exams: list[PatientExam] = []
                for raw_exam in raw_exams:
                    if not isinstance(raw_exam, dict):
                        raise ValueError(
                            f"Synthetic encounter {encounter_id!r} has an invalid exam"
                        )
                    exam_id = raw_exam.get("exam_id")
                    display_name = raw_exam.get("display_name")
                    try:
                        exam_status = ExamStatus(raw_exam.get("status"))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Synthetic exam in {encounter_id!r} has an invalid status"
                        ) from exc
                    if (
                        not isinstance(exam_id, str)
                        or not exam_id.startswith("EXAM-SYNTH-")
                        or not isinstance(display_name, str)
                        or not display_name.strip()
                    ):
                        raise ValueError(
                            f"Synthetic exam in {encounter_id!r} has incomplete data"
                        )
                    if exam_id in seen_exam_ids:
                        raise ValueError(
                            f"Synthetic patient {patient_id!r} has duplicate exam ID "
                            f"{exam_id!r}"
                        )
                    seen_exam_ids.add(exam_id)
                    exams.append(
                        PatientExam(
                            exam_id=exam_id,
                            display_name=display_name.strip(),
                            status=exam_status,
                        )
                    )
                encounters.append(
                    PatientEncounter(
                        encounter_id=encounter_id,
                        status=encounter_status.strip(),
                        exams=tuple(exams),
                    )
                )

            if patient_id in patients:
                raise ValueError(f"Duplicate synthetic patient ID {patient_id!r}")
            patients[patient_id] = PatientRecord(
                patient_id=patient_id,
                display_label=display_label.strip(),
                records=tuple(records),
                encounters=tuple(encounters),
            )

        self._patients: Mapping[str, PatientRecord] = MappingProxyType(patients)

    @property
    def available_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._patients))

    def read_patient(
        self,
        patient_id: str,
        access_context: AccessContext,
    ) -> PatientRecord:
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a non-empty string")
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
            raise PatientNotFoundError(
                f"Synthetic patient {patient_id!r} was not found"
            ) from exc
