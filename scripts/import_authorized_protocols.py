from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medical_assistant.datasets.clinical_import import convert_authorized_protocols


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate and convert approved clinical protocols for local RAG"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/massa_dados_protocolos_prontuarios.json"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/schema_protocolos_prontuarios.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/protocolos_institucionais_autorizados.jsonl"),
    )
    parser.add_argument("--institutional-approval-reference", required=True)
    parser.add_argument("--institutional-approved-at", required=True)
    parser.add_argument("--privacy-review-reference", required=True)
    parser.add_argument("--privacy-reviewed-at", required=True)
    args = parser.parse_args()

    paths = (args.input, args.schema, args.output)
    resolved = tuple(
        path if path.is_absolute() else project_root / path
        for path in paths
    )
    report = convert_authorized_protocols(
        resolved[0],
        resolved[1],
        resolved[2],
        institutional_approval_reference=args.institutional_approval_reference,
        institutional_approved_at=args.institutional_approved_at,
        privacy_review_reference=args.privacy_review_reference,
        privacy_reviewed_at=args.privacy_reviewed_at,
    )
    print("Protocolos convertidos:", report["protocols_converted"])
    print(
        "Prontuários não exportados:",
        report["patient_records_not_exported"],
    )
    print("Destino local:", report["output_path"])
    print("SHA-256 da origem:", report["source_file_sha256"])
    print("SHA-256 do schema:", report["schema_file_sha256"])


if __name__ == "__main__":
    main()
