import argparse
import json
import sys
from pathlib import Path

from medical_assistant.retrieval import (
    DocumentIndex,
    LocalE5Embedder,
    RetrievalEvalCase,
    evaluate_retrieval,
    load_approved_protocol_documents,
)


def _load_cases(path: Path) -> tuple[RetrievalEvalCase, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval evaluation cases do not exist: {path}")
    cases: list[RetrievalEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid retrieval evaluation JSONL at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Retrieval evaluation case at line {line_number} must be an object"
            )
        query = record.get("query")
        expected_source_ids = record.get("expected_source_ids")
        if (
            not isinstance(query, str)
            or not isinstance(expected_source_ids, list)
            or not all(
                isinstance(source_id, str) and source_id.strip()
                for source_id in expected_source_ids
            )
        ):
            raise ValueError(
                f"Retrieval evaluation case at line {line_number} has invalid fields"
            )
        cases.append(
            RetrievalEvalCase(
                query=query,
                expected_source_ids=tuple(expected_source_ids),
            )
        )
    if not cases:
        raise ValueError(f"Retrieval evaluation cases are empty: {path}")
    return tuple(cases)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local hybrid retrieval against human-reviewed protocol cases."
        )
    )
    parser.add_argument(
        "--protocols",
        type=Path,
        default=Path("data/processed/protocolos_institucionais_autorizados.jsonl"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/evaluation/protocol_retrieval_cases.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-lexical-score", type=float, default=0.5)
    parser.add_argument("--min-semantic-score", type=float, default=0.55)
    args = parser.parse_args()
    protocols_path = (
        args.protocols if args.protocols.is_absolute() else project_root / args.protocols
    )
    cases_path = args.cases if args.cases.is_absolute() else project_root / args.cases
    documents = load_approved_protocol_documents(protocols_path)
    cases = _load_cases(cases_path)
    expected_source_ids = {
        source_id
        for case in cases
        for source_id in case.expected_source_ids
    }
    protocol_source_ids = {document.source_id for document in documents}
    if expected_source_ids != protocol_source_ids:
        raise ValueError(
            "Evaluation cases must cover every protocol source and no other source"
        )
    lexical_metrics = evaluate_retrieval(
        DocumentIndex(documents),
        cases,
        top_k=args.top_k,
        min_lexical_score=args.min_lexical_score,
        min_semantic_score=args.min_semantic_score,
    )
    embedding_model = LocalE5Embedder(
        cache_dir=project_root / "artifacts" / "retrieval-embeddings"
    )
    hybrid_metrics = evaluate_retrieval(
        DocumentIndex(documents, embedding_model=embedding_model),
        cases,
        top_k=args.top_k,
        min_lexical_score=args.min_lexical_score,
        min_semantic_score=args.min_semantic_score,
    )
    report = {
        "embedding_model": embedding_model.version,
        "top_k": args.top_k,
        "min_lexical_score": args.min_lexical_score,
        "min_semantic_score": args.min_semantic_score,
        "lexical": lexical_metrics.__dict__,
        "hybrid": hybrid_metrics.__dict__,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
