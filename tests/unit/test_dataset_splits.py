import json
from pathlib import Path

import pytest

from medical_assistant.datasets import (
    QARecord,
    load_pubmedqa_test_labels,
    split_pubmedqa_pqal,
    split_records,
)


def make_record(
    record_id: str,
    *,
    source_dataset: str = "pubmedqa",
    source_id: str | None = None,
    question: str | None = None,
    label: str | None = "yes",
) -> QARecord:
    return QARecord(
        id=record_id,
        question=question or f"Pergunta {record_id}?",
        answer=f"Resposta {record_id}.",
        source_dataset=source_dataset,
        source_id=source_id or record_id,
        license="synthetic",
        label=label,
        source_subset="PQA-L" if source_dataset == "pubmedqa" else None,
        evaluation_eligible=source_dataset == "pubmedqa",
    )


def test_official_pubmedqa_test_ids_and_labels_are_preserved() -> None:
    records = [
        make_record("p1", source_id="1001", label="yes"),
        make_record("p2", source_id="1002", label="no"),
        make_record("p3", source_id="1003", label="maybe"),
        make_record("p4", source_id="1004", label="yes"),
    ]

    splits = split_records(
        records,
        validation_fraction=0.25,
        test_fraction=0.25,
        official_test_labels={"1001": "yes", "1003": "maybe"},
    )

    assert {record.source_id for record in splits.test} == {"1001", "1003"}
    assert {record.source_id for record in splits.train + splits.validation} == {
        "1002",
        "1004",
    }


def test_pubmedqa_pqal_split_reproduces_official_holdout_deterministically() -> None:
    labels = ["yes"] * 552 + ["no"] * 338 + ["maybe"] * 110
    records = [
        make_record(
            f"pqal-{index:04}",
            source_id=f"PMID-{index:04}",
            label=label,
        )
        for index, label in enumerate(labels)
    ]

    first = split_pubmedqa_pqal(records)
    second = split_pubmedqa_pqal(records)

    assert len(first.test) == 500
    assert len(first.validation) == 51
    assert len(first.train) == 449
    assert [
        [record.id for record in split]
        for split in (first.train, first.validation, first.test)
    ] == [
        [record.id for record in split]
        for split in (second.train, second.validation, second.test)
    ]


def test_pubmedqa_pqal_official_split_rejects_artificial_subset() -> None:
    record = QARecord(
        id="artificial",
        question="Synthetic question?",
        answer="Synthetic answer.",
        source_dataset="pubmedqa",
        source_id="PMID-1",
        license="MIT",
        label="yes",
        source_subset="PQA-A",
        evaluation_eligible=False,
    )

    with pytest.raises(ValueError, match="PQA-L"):
        split_pubmedqa_pqal([record])


def test_official_test_rejects_missing_pmids_and_label_mismatches() -> None:
    records = [make_record("p1", source_id="1001", label="yes")]

    with pytest.raises(ValueError, match="missing from PubMedQA"):
        split_records(records, official_test_labels={"9999": "yes"})
    with pytest.raises(ValueError, match="label mismatch"):
        split_records(records, official_test_labels={"1001": "no"})


def test_official_test_requires_expert_labeled_pqal_records() -> None:
    record = QARecord(
        id="artificial",
        question="Pergunta artificial?",
        answer="Resposta artificial.",
        source_dataset="pubmedqa",
        source_id="1001",
        license="synthetic",
        label="yes",
        source_subset="PQA-A",
        evaluation_eligible=False,
    )

    with pytest.raises(ValueError, match="missing from PubMedQA"):
        split_records([record], official_test_labels={"1001": "yes"})


def test_official_test_content_duplicates_are_excluded_from_train() -> None:
    official = make_record(
        "test",
        source_id="test-pmid",
        question="Same question?",
        label="yes",
    )
    duplicate = QARecord(
        id="duplicate-in-train",
        question=" same   question? ",
        answer=official.answer,
        source_dataset="pubmedqa",
        source_id="train-pmid",
        license="synthetic",
        label="yes",
    )
    other = make_record("other", source_id="other-pmid", label="no")

    splits = split_records(
        [official, duplicate, other],
        official_test_labels={"test-pmid": "yes"},
        validation_fraction=0,
        test_fraction=0.2,
    )

    assert [record.id for record in splits.test] == ["test"]
    assert [record.id for record in splits.excluded] == ["duplicate-in-train"]
    assert [record.id for record in splits.train] == ["other"]


def test_content_duplicates_never_cross_generated_splits() -> None:
    records = [
        make_record("one", source_dataset="medquad", label=None, question="Same?"),
        QARecord(
            id="two",
            question=" same? ",
            answer="Resposta one.",
            source_dataset="medquad",
            source_id="two",
            license="synthetic",
            label=None,
        ),
    ]
    records.extend(
        make_record(f"other-{index}", source_dataset="medquad", label=None)
        for index in range(20)
    )

    splits = split_records(
        records,
        seed=11,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    assignments = {
        record.id: split_name
        for split_name, split_records_ in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
        for record in split_records_
    }

    assert assignments["one"] == assignments["two"]
    assert set(assignments) == {record.id for record in records}


def test_source_group_records_never_cross_generated_splits() -> None:
    records = [
        QARecord(
            id=f"doc-a-{index}",
            question=f"Question {index}?",
            answer=f"Answer {index}.",
            source_dataset="medquad",
            source_id=f"doc-a-q-{index}",
            source_group_id="doc-a",
            license="synthetic",
        )
        for index in range(3)
    ]
    records.extend(
        make_record(f"other-{index}", source_dataset="medquad", label=None)
        for index in range(20)
    )

    splits = split_records(records, seed=3, validation_fraction=0.2, test_fraction=0.2)
    assignments = {
        record.id: split_name
        for split_name, split in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
        for record in split
    }

    assert len({assignments[f"doc-a-{index}"] for index in range(3)}) == 1


def test_generated_splits_are_deterministic_and_disjoint() -> None:
    records = [
        make_record(f"record-{index}", source_dataset="medquad", label=None)
        for index in range(30)
    ]
    first = split_records(records, seed=9)
    second = split_records(records, seed=9)

    assert [[record.id for record in split] for split in (first.train, first.validation, first.test)] == [
        [record.id for record in split]
        for split in (second.train, second.validation, second.test)
    ]
    split_ids = [
        {record.id for record in split}
        for split in (first.train, first.validation, first.test)
    ]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])


def test_load_pubmedqa_test_labels_validates_mapping(tmp_path: Path) -> None:
    labels_path = tmp_path / "test_ground_truth.json"
    labels_path.write_text(json.dumps({"123": "maybe"}), encoding="utf-8")

    assert load_pubmedqa_test_labels(labels_path) == {"123": "maybe"}

    labels_path.write_text(json.dumps({"123": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid label"):
        load_pubmedqa_test_labels(labels_path)
