"""Document ingestion, chunking, and retrieval."""

from medical_assistant.retrieval.index import (
    DocumentChunk,
    DocumentIndex,
    SearchResult,
    SourceDocument,
    load_approved_protocol_documents,
    load_medquad_documents,
    load_pubmedqa_documents,
    load_synthetic_documents,
)
from medical_assistant.retrieval.embeddings import LocalE5Embedder, TextEmbedder
from medical_assistant.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalMetrics,
    evaluate_retrieval,
)

__all__ = [
    "DocumentChunk",
    "DocumentIndex",
    "LocalE5Embedder",
    "RetrievalEvalCase",
    "RetrievalMetrics",
    "SearchResult",
    "SourceDocument",
    "TextEmbedder",
    "evaluate_retrieval",
    "load_approved_protocol_documents",
    "load_medquad_documents",
    "load_pubmedqa_documents",
    "load_synthetic_documents",
]
