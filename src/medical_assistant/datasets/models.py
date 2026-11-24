from dataclasses import dataclass


@dataclass(frozen=True)
class QARecord:
    id: str
    question: str
    answer: str
    source_dataset: str
    source_id: str
    license: str | None
    license_note: str | None = None
    source_group_id: str | None = None
    source_url: str | None = None
    context: str | None = None
    label: str | None = None
    source_subset: str | None = None
    label_provenance: str | None = None
    evaluation_eligible: bool = False
    split: str | None = None
    question_type: str | None = None
    focus: str | None = None
    source_origin: str | None = None
    review_status: str = "unreviewed"
    source_version: str | None = None
    source_metadata: dict[str, object] | None = None
    usage_rights_reference: str | None = None
    institutional_approval_status: str | None = None
    institutional_approval_reference: str | None = None
    institutional_approved_at: str | None = None
    privacy_review_status: str | None = None
    privacy_review_reference: str | None = None
    privacy_reviewed_at: str | None = None
    contains_patient_data: bool | None = None
    preprocessing_version: str | None = None
    preprocessing_redaction_counts: dict[str, int] | None = None
    preprocessing_review_required: bool | None = None


@dataclass(frozen=True)
class DatasetLoadResult:
    records: tuple[QARecord, ...]
    skipped_missing_answers: int = 0
    skipped_excluded_files: int = 0
    skipped_empty_documents: int = 0
