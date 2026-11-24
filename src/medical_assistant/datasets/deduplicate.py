from dataclasses import dataclass

from medical_assistant.datasets.models import QARecord


@dataclass(frozen=True)
class DeduplicationResult:
    records: tuple[QARecord, ...]
    duplicate_sources: tuple[tuple[str, str], ...]


def _canonical(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def deduplicate_records(records: list[QARecord] | tuple[QARecord, ...]) -> DeduplicationResult:
    """Keep the first record for each ID and exact normalized QA/context tuple."""
    unique_records: list[QARecord] = []
    duplicate_sources: list[tuple[str, str]] = []
    seen_ids: dict[str, tuple[QARecord, str]] = {}
    seen_content: dict[tuple[str, str, str, str], QARecord] = {}

    for record in records:
        previous_id_record = seen_ids.get(record.id)
        if previous_id_record is not None:
            previous_record, retained_id = previous_id_record
            if previous_record != record:
                raise ValueError(f"Conflicting records share id {record.id!r}")
            duplicate_sources.append((record.id, retained_id))
            continue

        content_key = (
            _canonical(record.question),
            _canonical(record.answer),
            _canonical(record.context),
            _canonical(record.label),
        )
        previous_content_record = seen_content.get(content_key)
        if previous_content_record is not None:
            duplicate_sources.append((record.id, previous_content_record.id))
            seen_ids[record.id] = (record, previous_content_record.id)
            continue

        seen_ids[record.id] = (record, record.id)
        seen_content[content_key] = record
        unique_records.append(record)

    return DeduplicationResult(
        records=tuple(unique_records),
        duplicate_sources=tuple(duplicate_sources),
    )
