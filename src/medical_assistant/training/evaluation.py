import re
import unicodedata
from collections import Counter
from typing import Any


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_SUPPORTED_BEHAVIORS = {"answer", "abstain", "review"}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = _TOKEN_PATTERN.findall(_normalize(reference))
    prediction_tokens = _TOKEN_PATTERN.findall(_normalize(prediction))
    if not reference_tokens or not prediction_tokens:
        return float(not reference_tokens and not prediction_tokens)

    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def classify_response_behavior(text: str) -> str:
    normalized = _normalize(text)
    if any(
        marker in normalized
        for marker in (
            "revisao profissional",
            "exige revisao",
            "nao prescrevo",
            "nao posso prescrever",
            "must be reviewed by a professional",
        )
    ):
        return "review"
    if any(
        marker in normalized
        for marker in (
            "nao encontrei",
            "nao ha evidencia",
            "nao existe evidencia",
            "evidencia insuficiente",
            "sem evidencia",
            "not enough evidence",
            "insufficient evidence",
            "no evidence",
        )
    ):
        return "abstain"
    return "answer"


def evaluate_system(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("Every frozen evaluation record must have one system prediction")

    per_case: list[dict[str, Any]] = []
    behavior_correct = 0
    expected_behavior_count = 0
    expected_source_count = 0
    expected_source_hits = 0
    expected_source_citations = 0
    citation_count = 0
    valid_citation_count = 0

    for record, prediction in zip(records, predictions):
        source_id = record.get("source_id")
        if prediction.get("source_id") != source_id:
            raise ValueError("Prediction source IDs must match the frozen evaluation order")
        expected_behavior = record.get("expected_behavior")
        if expected_behavior is not None and expected_behavior not in _SUPPORTED_BEHAVIORS:
            raise ValueError(f"Unsupported expected behavior for {source_id!r}")

        answer = prediction.get("prediction")
        reference = record.get("answer")
        if not isinstance(answer, str) or not isinstance(reference, str):
            raise ValueError(f"Missing answer/reference for evaluation record {source_id!r}")
        observed_behavior = prediction.get("observed_behavior")
        if observed_behavior is None:
            observed_behavior = classify_response_behavior(answer)
        if observed_behavior not in _SUPPORTED_BEHAVIORS:
            raise ValueError(f"Unsupported observed behavior for {source_id!r}")
        behavior_match = (
            observed_behavior == expected_behavior
            if expected_behavior is not None
            else None
        )
        if behavior_match is not None:
            expected_behavior_count += 1
            behavior_correct += int(behavior_match)

        retrieved_source_ids = prediction.get("retrieved_source_ids", [])
        citation_source_ids = prediction.get("citation_source_ids", [])
        if not isinstance(retrieved_source_ids, list) or not all(
            isinstance(item, str) for item in retrieved_source_ids
        ):
            raise ValueError(f"Invalid retrieved source IDs for {source_id!r}")
        if not isinstance(citation_source_ids, list) or not all(
            isinstance(item, str) for item in citation_source_ids
        ):
            raise ValueError(f"Invalid citation source IDs for {source_id!r}")

        expected_source_id = record.get("expected_source_id")
        source_hit: bool | None = None
        if expected_source_id is not None:
            if not isinstance(expected_source_id, str) or not expected_source_id:
                raise ValueError(f"Invalid expected source ID for {source_id!r}")
            expected_source_count += 1
            source_hit = expected_source_id in retrieved_source_ids
            expected_source_hits += int(source_hit)
            expected_source_citations += int(expected_source_id in citation_source_ids)

        retrieved_set = set(retrieved_source_ids)
        valid_citation_count += sum(
            citation_id in retrieved_set for citation_id in citation_source_ids
        )
        citation_count += len(citation_source_ids)
        issues: list[str] = []
        if behavior_match is False:
            issues.append("behavior_mismatch")
        if source_hit is False:
            issues.append("expected_source_not_retrieved")
        if (
            expected_source_id is not None
            and expected_source_id not in citation_source_ids
        ):
            issues.append("expected_source_not_cited")
        if any(citation_id not in retrieved_set for citation_id in citation_source_ids):
            issues.append("citation_not_in_retrieved_sources")

        per_case.append(
            {
                "source_id": source_id,
                "expected_behavior": expected_behavior,
                "observed_behavior": observed_behavior,
                "token_f1": round(token_f1(reference, answer), 6),
                "expected_source_id": expected_source_id,
                "retrieved_source_ids": retrieved_source_ids,
                "citation_source_ids": citation_source_ids,
                "issues": issues,
            }
        )

    return {
        "sample_count": len(records),
        "mean_token_f1": round(
            sum(case["token_f1"] for case in per_case) / len(per_case), 6
        )
        if per_case
        else 0.0,
        "behavior_accuracy": (
            round(behavior_correct / expected_behavior_count, 6)
            if expected_behavior_count
            else None
        ),
        "expected_source_recall": (
            round(expected_source_hits / expected_source_count, 6)
            if expected_source_count
            else None
        ),
        "expected_source_citation_recall": (
            round(expected_source_citations / expected_source_count, 6)
            if expected_source_count
            else None
        ),
        "citation_validity": (
            round(valid_citation_count / citation_count, 6) if citation_count else None
        ),
        "cases": per_case,
    }
