import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from medical_assistant.datasets.loaders import load_pubmedqa
from medical_assistant.datasets.splits import split_pubmedqa_pqal
from medical_assistant.datasets.internal import export_training_splits
from medical_assistant.training.runner import validate_splits


PUBMEDQA_REVISION = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
PUBMEDQA_PQAL_SHA256 = "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pubmedqa(source_path: Path, output_dir: Path) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"PubMedQA PQA-L source does not exist: {source_path}")
    source_sha256 = _sha256(source_path)
    if source_sha256 != PUBMEDQA_PQAL_SHA256:
        raise ValueError(
            "PubMedQA source hash does not match the pinned PQA-L revision"
        )
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite PubMedQA preparation manifest: {output_dir}"
        )

    loaded = load_pubmedqa(source_path, subset="PQA-L")
    records = tuple(
        replace(record, source_version=PUBMEDQA_REVISION)
        for record in loaded.records
    )
    split_result = split_pubmedqa_pqal(records)
    named_splits = {
        "train": split_result.train,
        "validation": split_result.validation,
        "test": split_result.test,
    }
    split_payload = {
        split_name: [
            asdict(replace(record, split=split_name))
            for record in split_records
        ]
        for split_name, split_records in named_splits.items()
    }
    validate_splits(split_payload, synthetic_only=False)

    output_paths = export_training_splits(split_result, output_dir)
    split_hashes = {
        path.stem: _sha256(path)
        for path in output_paths
    }
    label_counts = {
        split_name: {
            label: sum(record.label == label for record in split_records)
            for label in ("yes", "no", "maybe")
        }
        for split_name, split_records in named_splits.items()
    }
    manifest = {
        "dataset": "PubMedQA",
        "subset": "PQA-L",
        "source_url": (
            "https://raw.githubusercontent.com/pubmedqa/pubmedqa/"
            f"{PUBMEDQA_REVISION}/data/ori_pqal.json"
        ),
        "source_revision": PUBMEDQA_REVISION,
        "source_sha256": source_sha256,
        "repository_license": "MIT",
        "underlying_article_rights_note": (
            "The repository license does not establish rights for every underlying "
            "abstract or source record."
        ),
        "split_method": (
            "Reproduced upstream seed-0 PQA-L 500/500 holdout; split the "
            "cross-validation half by source/content with seed 42, reserving "
            "approximately 10% for validation."
        ),
        "counts": {
            split_name: len(split_records)
            for split_name, split_records in named_splits.items()
        },
        "excluded_duplicate_records": len(split_result.excluded),
        "label_counts": label_counts,
        "split_sha256": split_hashes,
        "pqa_u_used": False,
        "pqa_a_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pinned PubMedQA PQA-L grouped training splits"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/pubmedqa/ori_pqal.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/pubmedqa"),
    )
    args = parser.parse_args()
    manifest = prepare_pubmedqa(args.source, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
