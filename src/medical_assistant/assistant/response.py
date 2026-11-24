from dataclasses import dataclass
from enum import StrEnum
from typing import Collection


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SourceCitation:
    source: str
    title: str
    section: str
    source_id: str

    def __post_init__(self) -> None:
        for field_name in ("source", "title", "section", "source_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Citation {field_name} must be a non-empty string")


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    sources: tuple[SourceCitation, ...]
    uncertainty: str
    review_status: ReviewStatus

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("Answer must be a non-empty string")
        if not isinstance(self.uncertainty, str) or not self.uncertainty.strip():
            raise ValueError("Uncertainty must be a non-empty explanation")
        if not isinstance(self.review_status, ReviewStatus):
            raise ValueError("review_status must be a ReviewStatus value")
        if not isinstance(self.sources, tuple):
            object.__setattr__(self, "sources", tuple(self.sources))
        if not all(isinstance(source, SourceCitation) for source in self.sources):
            raise ValueError("sources must contain only SourceCitation values")


def validate_source_citations(
    response: AssistantResponse,
    available_sources: Collection[SourceCitation],
) -> None:
    available = set(available_sources)
    invalid_sources = [source for source in response.sources if source not in available]
    if invalid_sources:
        invalid_ids = ", ".join(source.source_id for source in invalid_sources)
        raise ValueError(f"Response cites unavailable sources: {invalid_ids}")
