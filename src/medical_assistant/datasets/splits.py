import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from medical_assistant.datasets.models import QARecord


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[QARecord, ...]
    validation: tuple[QARecord, ...]
    test: tuple[QARecord, ...]
    excluded: tuple[QARecord, ...] = ()


def load_pubmedqa_test_labels(path: Path) -> dict[str, str]:
    """Load the official PubMedQA PMID-to-label test mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("PubMedQA test ground truth must be an object")

    labels: dict[str, str] = {}
    for source_id, label in data.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("PubMedQA test ground truth contains an invalid PMID")
        if not isinstance(label, str) or label not in {"yes", "no", "maybe"}:
            raise ValueError(f"PubMedQA test ground truth has invalid label for {source_id!r}")
        labels[source_id] = label
    return labels


def split_pubmedqa_pqal(
    records: list[QARecord] | tuple[QARecord, ...],
    *,
    official_seed: int = 0,
    validation_seed: int = 42,
) -> DatasetSplits:
    """Reproduce PubMedQA's 500/500 PQA-L holdout and split its CV half.

    The record order must match the pinned source JSON order. The official
    test half is selected with the upstream seed and remains evaluation-only;
    10% of the remaining cross-validation half is reserved for validation.
    """
    if not records:
        raise ValueError("PQA-L records must not be empty")

    ids_by_label: dict[str, list[str]] = {"yes": [], "no": [], "maybe": []}
    records_by_id: dict[str, QARecord] = {}
    for record in records:
        if (
            record.source_dataset != "pubmedqa"
            or record.source_subset != "PQA-L"
            or not record.evaluation_eligible
        ):
            raise ValueError("Official PubMedQA splitting requires evaluation-eligible PQA-L")
        label = record.label
        if not isinstance(label, str) or label not in ids_by_label:
            raise ValueError("PQA-L record has an invalid expert label")
        if record.source_id in records_by_id:
            raise ValueError(f"Duplicate PubMedQA source ID {record.source_id!r}")
        records_by_id[record.source_id] = record
        ids_by_label[label].append(record.source_id)

    rng = random.Random(official_seed)
    label_parts: dict[str, tuple[list[str], list[str]]] = {}
    for label, source_ids in ids_by_label.items():
        rng.shuffle(source_ids)
        first_part_size = math.ceil(len(source_ids) / 2)
        label_parts[label] = (
            source_ids[:first_part_size],
            source_ids[first_part_size:],
        )

    cv_ids = [
        source_id
        for label in ids_by_label
        for source_id in label_parts[label][0]
    ]
    official_test_ids = [
        source_id
        for label in ids_by_label
        for source_id in label_parts[label][1]
    ]
    if len(official_test_ids) != len(cv_ids):
        source_id = rng.choice(cv_ids)
        cv_ids.remove(source_id)
        official_test_ids.append(source_id)

    if len(official_test_ids) != len(cv_ids):
        raise ValueError("Unable to reproduce PubMedQA's balanced official holdout")

    official_test_labels = {
        source_id: records_by_id[source_id].label
        for source_id in official_test_ids
    }
    return split_records(
        records,
        seed=validation_seed,
        validation_fraction=0.05,
        test_fraction=0.5,
        official_test_labels=official_test_labels,
    )


def _content_key(record: QARecord) -> tuple[str, str, str]:
    return tuple(
        " ".join(value.casefold().split())
        for value in (record.question, record.answer, record.context or "")
    )


def _group_records(records: list[QARecord]) -> list[list[QARecord]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    key_owners: dict[tuple[str, ...], int] = {}
    for index, record in enumerate(records):
        source_group_id = record.source_group_id or record.source_id
        keys = (
            ("source", record.source_dataset, source_group_id),
            ("content", *_content_key(record)),
        )
        for key in keys:
            previous_index = key_owners.setdefault(key, index)
            union(index, previous_index)

    groups_by_root: dict[int, list[QARecord]] = defaultdict(list)
    for index, record in enumerate(records):
        groups_by_root[find(index)].append(record)
    return sorted(
        (sorted(group, key=lambda record: record.id) for group in groups_by_root.values()),
        key=lambda group: group[0].id,
    )


def _take_count(group_count: int, fraction: float) -> int:
    if group_count < 2 or fraction <= 0:
        return 0
    return min(group_count - 1, max(1, int(group_count * fraction + 0.5)))


def _partition_groups(
    groups: list[list[QARecord]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[QARecord], list[QARecord], list[QARecord]]:
    strata: dict[str | None, list[list[QARecord]]] = defaultdict(list)
    for group in groups:
        labels = {record.label for record in group}
        stratum = next(iter(labels)) if len(labels) == 1 else None
        strata[stratum].append(group)

    rng = random.Random(seed)
    train: list[QARecord] = []
    validation: list[QARecord] = []
    test: list[QARecord] = []

    for stratum_groups in strata.values():
        rng.shuffle(stratum_groups)
        validation_count = _take_count(len(stratum_groups), validation_fraction)
        test_count = _take_count(
            len(stratum_groups) - validation_count,
            test_fraction / (1 - validation_fraction),
        )
        validation_groups = stratum_groups[:validation_count]
        test_groups = stratum_groups[
            validation_count : validation_count + test_count
        ]
        train_groups = stratum_groups[validation_count + test_count :]

        validation.extend(record for group in validation_groups for record in group)
        test.extend(record for group in test_groups for record in group)
        train.extend(record for group in train_groups for record in group)

    return train, validation, test


def split_records(
    records: list[QARecord] | tuple[QARecord, ...],
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    official_test_labels: dict[str, str] | None = None,
) -> DatasetSplits:
    """Create deterministic, content-grouped splits.

    When official PubMedQA test labels are supplied, those PMIDs remain in the
    test set unchanged. Other records duplicating their content are excluded
    from training/validation to prevent test leakage.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    all_records = list(records)
    seen_ids: set[str] = set()
    for record in all_records:
        if record.id in seen_ids:
            raise ValueError(f"Duplicate record id {record.id!r}")
        seen_ids.add(record.id)

    official_test: list[QARecord] = []
    remaining = all_records
    excluded: list[QARecord] = []

    if official_test_labels is not None:
        by_source_id: dict[str, QARecord] = {}
        for record in all_records:
            if record.source_dataset == "pubmedqa" and record.source_subset == "PQA-L":
                if not record.evaluation_eligible:
                    raise ValueError(
                        f"PubMedQA record {record.source_id!r} is not evaluation-eligible"
                    )
                if record.source_id in by_source_id:
                    raise ValueError(f"Duplicate PubMedQA source id {record.source_id!r}")
                by_source_id[record.source_id] = record

        if any(
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(label, str)
            or label not in {"yes", "no", "maybe"}
            for source_id, label in official_test_labels.items()
        ):
            raise ValueError("Official PubMedQA test labels contain invalid IDs or labels")

        missing_ids = set(official_test_labels) - by_source_id.keys()
        if missing_ids:
            raise ValueError(
                f"Official test IDs are missing from PubMedQA records: {sorted(missing_ids)[:5]}"
            )

        for source_id, expected_label in official_test_labels.items():
            record = by_source_id[source_id]
            if record.label != expected_label:
                raise ValueError(
                    f"Official label mismatch for PubMedQA record {source_id!r}"
                )
            official_test.append(record)

        official_test_ids = set(official_test_labels)
        remaining = [
            record
            for record in all_records
            if not (
                record.source_dataset == "pubmedqa"
                and record.source_subset == "PQA-L"
                and record.source_id in official_test_ids
            )
        ]
        test_group_keys = {
            (record.source_dataset, record.source_group_id or record.source_id)
            for record in official_test
        }
        test_content = {_content_key(record) for record in official_test}
        retained: list[QARecord] = []
        for record in remaining:
            source_group_key = (
                record.source_dataset,
                record.source_group_id or record.source_id,
            )
            if source_group_key in test_group_keys or _content_key(record) in test_content:
                excluded.append(record)
            else:
                retained.append(record)
        remaining = retained

    groups = _group_records(remaining)
    if official_test_labels is None:
        train, validation, test = _partition_groups(
            groups,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        test.extend(official_test)
    else:
        validation_ratio = (
            validation_fraction / (1 - test_fraction) if test_fraction < 1 else 0
        )
        train, validation, _ = _partition_groups(
            groups,
            seed=seed,
            validation_fraction=validation_ratio,
            test_fraction=0,
        )
        test = official_test

    return DatasetSplits(
        train=tuple(sorted(train, key=lambda record: record.id)),
        validation=tuple(sorted(validation, key=lambda record: record.id)),
        test=tuple(sorted(test, key=lambda record: record.id)),
        excluded=tuple(sorted(excluded, key=lambda record: record.id)),
    )
