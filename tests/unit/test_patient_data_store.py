import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from medical_assistant.patient_data import (
    AccessContext,
    AccessDeniedError,
    ExamStatus,
    PatientNotFoundError,
    PatientRole,
    SyntheticStaffStore,
    SyntheticPatientStore,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "data" / "synthetic" / "patients.json"
PATIENT_ID = "PATIENT-SYNTH-001"


def make_context(
    *,
    role: PatientRole = PatientRole.CLINICIAN,
    authorized_patient_ids: frozenset[str] = frozenset({PATIENT_ID}),
) -> AccessContext:
    return AccessContext(
        user_id="demo-clinician",
        role=role,
        authorized_patient_ids=authorized_patient_ids,
    )


def test_authorized_clinician_can_read_synthetic_patient() -> None:
    store = SyntheticPatientStore(FIXTURE_PATH)

    patient = store.read_patient(PATIENT_ID, make_context())

    assert patient.patient_id == PATIENT_ID
    assert patient.records[0].record_id == "RECORD-SYNTH-001"


def test_read_is_denied_for_wrong_role_or_patient_scope() -> None:
    store = SyntheticPatientStore(FIXTURE_PATH)

    with pytest.raises(AccessDeniedError):
        store.read_patient(
            PATIENT_ID,
            make_context(role=PatientRole.GUEST),
        )
    with pytest.raises(AccessDeniedError):
        store.read_patient(
            PATIENT_ID,
            make_context(authorized_patient_ids=frozenset({"PATIENT-SYNTH-OTHER"})),
        )


def test_authorization_is_checked_before_patient_existence() -> None:
    store = SyntheticPatientStore(FIXTURE_PATH)

    with pytest.raises(AccessDeniedError):
        store.read_patient(
            "PATIENT-SYNTH-UNKNOWN",
            make_context(authorized_patient_ids=frozenset()),
        )
    with pytest.raises(PatientNotFoundError):
        store.read_patient(
            "PATIENT-SYNTH-UNKNOWN",
            make_context(authorized_patient_ids=frozenset({"PATIENT-SYNTH-UNKNOWN"})),
        )


def test_patient_fixture_objects_are_immutable() -> None:
    store = SyntheticPatientStore(FIXTURE_PATH)
    patient = store.read_patient(PATIENT_ID, make_context())

    with pytest.raises(FrozenInstanceError):
        patient.display_label = "Alterado"
    assert not hasattr(store, "write_patient")
    assert patient.encounters[0].exams[0].status is ExamStatus.PENDING
    with pytest.raises(FrozenInstanceError):
        patient.encounters[0].exams[0].status = ExamStatus.COMPLETED


def test_loader_rejects_non_synthetic_patients(tmp_path: Path) -> None:
    invalid_path = tmp_path / "patients.json"
    invalid_path.write_text(
        json.dumps(
            [
                {
                    "synthetic": False,
                    "patient_id": PATIENT_ID,
                    "display_label": "Not synthetic",
                    "records": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicitly synthetic"):
        SyntheticPatientStore(invalid_path)


def test_loader_rejects_invalid_synthetic_exam_status(tmp_path: Path) -> None:
    invalid_path = tmp_path / "patients.json"
    invalid_path.write_text(
        json.dumps(
            [
                {
                    "synthetic": True,
                    "patient_id": PATIENT_ID,
                    "display_label": "Synthetic",
                    "records": [],
                    "encounters": [
                        {
                            "encounter_id": "ENCOUNTER-SYNTH-001",
                            "status": "open",
                            "exams": [
                                {
                                    "exam_id": "EXAM-SYNTH-001",
                                    "display_name": "Synthetic exam",
                                    "status": "unknown",
                                }
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid status"):
        SyntheticPatientStore(invalid_path)


def test_synthetic_staff_store_provides_only_declared_access_scope() -> None:
    staff = SyntheticStaffStore(ROOT / "data" / "synthetic" / "staff.json")

    member = staff.read_staff("demo-clinician")

    assert member.role is PatientRole.CLINICIAN
    assert member.access_context().authorized_patient_ids == frozenset(
        {PATIENT_ID, "PATIENT-SYNTH-002"}
    )
    with pytest.raises(AccessDeniedError, match="identity"):
        staff.read_staff("unknown-demo-user")


def test_access_context_rejects_invalid_user_and_role() -> None:
    with pytest.raises(ValueError, match="user_id"):
        AccessContext("", PatientRole.CLINICIAN, frozenset())
    with pytest.raises(ValueError, match="PatientRole"):
        AccessContext("user", "clinician", frozenset())  # type: ignore[arg-type]
