import argparse
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from medical_assistant.datasets import (
    QARecord,
    deduplicate_records,
    load_medquad,
    load_pubmedqa,
)
from medical_assistant.datasets.splits import (
    DatasetSplits,
    split_pubmedqa_pqal,
    split_records,
)
from medical_assistant.training.runner import load_jsonl, validate_splits


PUBMEDQA_REVISION = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
MEDQUAD_REVISION = "577bd37b96c02d1833b2c9eed2de9f96964e96cb"
MEDQUAD_ARCHIVE_SHA256 = "161d948f8ce8f82accca1f5769a512dff2a01af2b3004d48f2b3da2758a38a09"
TRAIN_PER_PUBLIC_DATASET = 254
MEDQUAD_EVALUATION_RECORDS_PER_SPLIT = 64
SYNTHETIC_TRAIN_DIR = Path("training") / "synthetic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qa_key(record: QARecord) -> tuple[str, str]:
    return (
        " ".join(record.question.casefold().split()),
        " ".join(record.answer.casefold().split()),
    )


def _sample_records(
    records: tuple[QARecord, ...],
    count: int,
    *,
    seed: int,
) -> tuple[QARecord, ...]:
    if len(records) < count:
        raise ValueError(
            f"Need {count} records for the balanced sample, found {len(records)}"
        )
    return tuple(random.Random(seed).sample(list(records), count))


def _sample_medquad_documents(
    records: tuple[QARecord, ...],
    count: int,
    *,
    seed: int,
) -> tuple[QARecord, ...]:
    grouped: dict[str, list[QARecord]] = {}
    for record in records:
        group_id = record.source_group_id or record.source_id
        grouped.setdefault(group_id, []).append(record)

    if len(grouped) < count:
        raise ValueError(
            f"Need {count} MedQuAD documents, found {len(grouped)}"
        )

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    rng = random.Random(seed + 1)
    selected = [
        rng.choice(grouped[group_id])
        for group_id in group_ids[:count]
    ]
    return tuple(sorted(selected, key=lambda record: record.id))


def _synthetic_records(
    project_root: Path,
    split_name: str,
) -> tuple[tuple[QARecord, ...], dict[tuple[str, str], dict[str, Any]]]:
    path = project_root / SYNTHETIC_TRAIN_DIR / f"{split_name}.jsonl"
    rows = load_jsonl(path)
    records: list[QARecord] = []
    extras: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_id"]
        if row.get("synthetic") is not True:
            raise ValueError(f"Synthetic record {source_id!r} is not marked synthetic")
        records.append(
            QARecord(
                id=f"synthetic:{source_id}",
                question=row["question"],
                answer=row["answer"],
                source_dataset="synthetic",
                source_id=source_id,
                source_group_id=row.get("source_group_id") or source_id,
                license="synthetic",
            )
        )
        extras[("synthetic", source_id)] = {
            key: row[key]
            for key in ("expected_behavior", "expected_source_id")
            if key in row
        }
    return tuple(records), extras


def _serialize_splits(
    splits: DatasetSplits,
    extras: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        split_records = getattr(splits, split_name)
        rows: list[dict[str, Any]] = []
        for record in split_records:
            row = asdict(replace(record, split=split_name))
            if record.source_dataset == "synthetic":
                row["synthetic"] = True
            row.update(extras.get((record.source_dataset, record.source_id), {}))
            rows.append(row)
        result[split_name] = sorted(rows, key=lambda row: row["id"])
    return result


def prepare_hybrid_training(
    project_root: Path,
    *,
    pubmedqa_path: Path,
    medquad_path: Path,
    medquad_archive_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if _sha256(medquad_archive_path) != MEDQUAD_ARCHIVE_SHA256:
        raise ValueError("MedQuAD archive hash does not match the pinned revision")

    pubmedqa = load_pubmedqa(pubmedqa_path, subset="PQA-L")
    pubmedqa_splits = split_pubmedqa_pqal(pubmedqa.records)
    pubmedqa_keys = {_qa_key(record) for record in pubmedqa.records}

    medquad = load_medquad(medquad_path)
    eligible_medquad = tuple(
        record for record in medquad.records if _qa_key(record) not in pubmedqa_keys
    )
    medquad_deduplication = deduplicate_records(eligible_medquad)
    medquad_splits = split_records(
        list(medquad_deduplication.records),
        seed=42,
        validation_fraction=0.1,
        test_fraction=0.2,
    )

    synthetic_train, train_extras = _synthetic_records(project_root, "train")
    synthetic_validation, validation_extras = _synthetic_records(
        project_root, "validation"
    )
    synthetic_test, test_extras = _synthetic_records(project_root, "test")
    extras = {**train_extras, **validation_extras, **test_extras}

    selected_splits = DatasetSplits(
        train=(
            *_sample_records(
                pubmedqa_splits.train,
                TRAIN_PER_PUBLIC_DATASET,
                seed=42,
            ),
            *_sample_medquad_documents(
                medquad_splits.train,
                TRAIN_PER_PUBLIC_DATASET,
                seed=42,
            ),
            *synthetic_train,
        ),
        validation=(
            *pubmedqa_splits.validation,
            *_sample_medquad_documents(
                medquad_splits.validation,
                MEDQUAD_EVALUATION_RECORDS_PER_SPLIT,
                seed=43,
            ),
            *synthetic_validation,
        ),
        test=(
            *pubmedqa_splits.test,
            *_sample_medquad_documents(
                medquad_splits.test,
                MEDQUAD_EVALUATION_RECORDS_PER_SPLIT,
                seed=44,
            ),
            *synthetic_test,
        ),
    )
    serialized_splits = _serialize_splits(selected_splits, extras)
    validate_splits(serialized_splits, synthetic_only=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split_name: output_dir / f"{split_name}.jsonl"
        for split_name in serialized_splits
    }
    manifest_path = output_dir / "manifest.json"
    existing_paths = [
        path for path in (*split_paths.values(), manifest_path) if path.exists()
    ]
    if existing_paths:
        raise FileExistsError(
            f"Refusing to overwrite hybrid training data in {output_dir}"
        )

    for split_name, rows in serialized_splits.items():
        split_paths[split_name].write_text(
            "".join(
                f"{json.dumps(row, ensure_ascii=False)}\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    source_counts = {
        split_name: {
            dataset: sum(row["source_dataset"] == dataset for row in rows)
            for dataset in ("pubmedqa", "medquad", "synthetic")
        }
        for split_name, rows in serialized_splits.items()
    }
    manifest = {
        "experiment_id": "qwen2.5-0.5b-hybrid-public-qlora-v2",
        "seed": 42,
        "sources": {
            "PubMedQA": {
                "revision": PUBMEDQA_REVISION,
                "source_sha256": _sha256(pubmedqa_path),
                "subset": "PQA-L",
                "repository_license": "MIT; underlying article rights require separate review",
                "records_loaded": len(pubmedqa.records),
            },
            "MedQuAD": {
                "revision": MEDQUAD_REVISION,
                "archive_sha256": _sha256(medquad_archive_path),
                "repository_license": "CC BY 4.0",
                "records_loaded": len(medquad.records),
                "records_missing_answers_skipped": medquad.skipped_missing_answers,
                "empty_documents_skipped": medquad.skipped_empty_documents,
                "excluded_xml_files": medquad.skipped_excluded_files,
                "duplicate_records_skipped": len(medquad_deduplication.duplicate_sources),
                "cross_dataset_question_answer_pairs_skipped": (
                    len(medquad.records) - len(eligible_medquad)
                ),
                "excluded_subsets": [
                    "10_MPlus_ADAM_QA",
                    "11_MPlusDrugs_QA",
                    "12_MPlusHerbsSupplements_QA",
                ],
            },
            "Hospital protocols": {
                "role": "RAG only; current protocol records are not marked training-eligible"
            },
            "Patient records": {
                "role": "selected-record RAG and structured exam lookup; never fine-tuned"
            },
            "Synthetic safety examples": {
                "role": "train/validation/test behavior examples",
                "train_records": len(synthetic_train),
                "validation_records": len(synthetic_validation),
                "test_records": len(synthetic_test),
            },
        },
        "counts": {
            "train": len(selected_splits.train),
            "validation": len(selected_splits.validation),
            "test": len(selected_splits.test),
            "by_source": source_counts,
        },
        "selection": {
            "train": f"{TRAIN_PER_PUBLIC_DATASET} records per public dataset, sampled with seed 42",
            "medquad_evaluation": (
                f"{MEDQUAD_EVALUATION_RECORDS_PER_SPLIT} unique source documents "
                "per validation/test split"
            ),
            "pubmedqa_test": "The official 500-record PQA-L holdout remains evaluation-only.",
            "patient_records_in_training": False,
            "protocol_contents_in_training": False,
        },
        "limitations": [
            "The local GTX 1650 experiment uses a balanced subset, not the full MedQuAD corpus.",
            "Token F1 and dataset metrics do not establish clinical accuracy or safety.",
            "Public educational QA is not a hospital protocol or a patient record.",
        ],
        "split_sha256": {
            split_name: _sha256(path)
            for split_name, path in split_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a privacy-safe hybrid QLoRA dataset"
    )
    parser.add_argument(
        "--pubmedqa",
        type=Path,
        default=Path("data") / "raw" / "pubmedqa" / "ori_pqal.json",
    )
    parser.add_argument(
        "--medquad",
        type=Path,
        default=(
            Path("data")
            / "raw"
            / "medquad"
            / f"MedQuAD-{MEDQUAD_REVISION}"
        ),
    )
    parser.add_argument(
        "--medquad-archive",
        type=Path,
        default=(
            Path("data")
            / "raw"
            / "medquad"
            / f"medquad-{MEDQUAD_REVISION}.zip"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed" / "hybrid_training",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    manifest = prepare_hybrid_training(
        project_root,
        pubmedqa_path=resolve(args.pubmedqa),
        medquad_path=resolve(args.medquad),
        medquad_archive_path=resolve(args.medquad_archive),
        output_dir=resolve(args.output_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
