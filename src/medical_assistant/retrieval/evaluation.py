from dataclasses import dataclass

from medical_assistant.retrieval.index import DocumentIndex


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalMetrics:
    cases: int
    top_k: int
    recall_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank_at_k: float


def evaluate_retrieval(
    index: DocumentIndex,
    cases: tuple[RetrievalEvalCase, ...] | list[RetrievalEvalCase],
    *,
    top_k: int = 5,
    min_lexical_score: float = 0.5,
    min_semantic_score: float = 0.55,
) -> RetrievalMetrics:
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if not 0 <= min_lexical_score <= 1:
        raise ValueError("min_lexical_score must be between zero and one")
    if not -1 <= min_semantic_score <= 1:
        raise ValueError("min_semantic_score must be between minus one and one")
    if not cases:
        raise ValueError("At least one retrieval evaluation case is required")

    available_source_ids = {chunk.source_id for chunk in index.chunks}
    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    hits = 0

    for case in cases:
        if not case.query.strip():
            raise ValueError("Evaluation queries must not be empty")
        if not case.expected_source_ids:
            raise ValueError("Evaluation cases must declare expected source IDs")
        unknown_source_ids = set(case.expected_source_ids) - available_source_ids
        if unknown_source_ids:
            raise ValueError(
                "Evaluation case references sources not present in the index"
            )

        results = index.search(
            case.query,
            limit=top_k,
            min_score=min_lexical_score,
            min_semantic_score=min_semantic_score,
        )
        retrieved_source_ids = tuple(
            dict.fromkeys(result.chunk.source_id for result in results)
        )
        expected_source_ids = set(case.expected_source_ids)
        relevant_count = len(expected_source_ids & set(retrieved_source_ids))
        recall_scores.append(relevant_count / len(expected_source_ids))
        if relevant_count:
            hits += 1
            reciprocal_rank = next(
                1 / rank
                for rank, source_id in enumerate(retrieved_source_ids, start=1)
                if source_id in expected_source_ids
            )
        else:
            reciprocal_rank = 0.0
        reciprocal_ranks.append(reciprocal_rank)

    case_count = len(cases)
    return RetrievalMetrics(
        cases=case_count,
        top_k=top_k,
        recall_at_k=sum(recall_scores) / case_count,
        hit_rate_at_k=hits / case_count,
        mean_reciprocal_rank_at_k=sum(reciprocal_ranks) / case_count,
    )
