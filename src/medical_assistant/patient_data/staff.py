import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from medical_assistant.patient_data.store import (
    AccessContext,
    AccessDeniedError,
    PatientRole,
)


@dataclass(frozen=True)
class SyntheticStaffMember:
    user_id: str
    role: PatientRole
    authorized_patient_ids: frozenset[str]

    def access_context(self) -> AccessContext:
        return AccessContext(
            user_id=self.user_id,
            role=self.role,
            authorized_patient_ids=self.authorized_patient_ids,
        )


class SyntheticStaffStore:
    """Loads demo identities only; it is not an authentication mechanism."""

    def __init__(self, path: Path) -> None:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise ValueError("Synthetic staff data must contain a list")

        members: dict[str, SyntheticStaffMember] = {}
        for item in raw_data:
            if not isinstance(item, dict) or item.get("synthetic") is not True:
                raise ValueError("Every staff fixture must be explicitly synthetic")
            user_id = item.get("user_id")
            raw_role = item.get("role")
            raw_patient_ids = item.get("authorized_patient_ids")
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("Synthetic staff fixture has an invalid user_id")
            if not isinstance(raw_patient_ids, list) or not all(
                isinstance(patient_id, str)
                and patient_id.startswith("PATIENT-SYNTH-")
                for patient_id in raw_patient_ids
            ):
                raise ValueError(
                    f"Synthetic staff member {user_id!r} has invalid patient scope"
                )
            try:
                role = PatientRole(raw_role)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Synthetic staff member {user_id!r} has an invalid role"
                ) from exc
            if user_id in members:
                raise ValueError(f"Duplicate synthetic staff user {user_id!r}")
            members[user_id] = SyntheticStaffMember(
                user_id=user_id,
                role=role,
                authorized_patient_ids=frozenset(raw_patient_ids),
            )
        self._members: Mapping[str, SyntheticStaffMember] = MappingProxyType(members)

    def read_staff(self, user_id: str) -> SyntheticStaffMember:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        try:
            return self._members[user_id]
        except KeyError as exc:
            raise AccessDeniedError("The synthetic demo identity is not available") from exc
