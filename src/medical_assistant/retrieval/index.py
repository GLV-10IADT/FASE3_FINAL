import hashlib
import json
import math
import os
import re
import struct
import sys
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from medical_assistant.datasets.models import QARecord
from medical_assistant.retrieval.embeddings import TextEmbedder

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_EMBEDDING_CACHE_MAGIC = b"MAIDX001"
_TERM_REPLACEMENTS = (
    (re.compile(r"\baquatic urticaria\b", re.IGNORECASE), "aquagenic urticaria"),
    (re.compile(r"\bwater[- ]induced urticaria\b", re.IGNORECASE), "aquagenic urticaria"),
    (re.compile(r"\bbabies?\b", re.IGNORECASE), "infants"),
    (re.compile(r"\bfainting\b", re.IGNORECASE), "syncope"),
    (re.compile(r"\bbathing\b", re.IGNORECASE), "bath"),
)
_STOPWORDS = {
    "a", "as", "ao", "aos", "com", "da", "das", "de", "do", "dos", "e", "em",
    "essa", "esse", "esta", "este", "na", "nas", "no", "nos", "o", "os", "para",
    "por", "que", "um", "uma", "the", "a", "an", "and", "are", "be", "can", "for",
    "in", "is", "of", "on", "or", "to", "what", "which", "during", "related",
}


@dataclass(frozen=True)
class SourceDocument:
    id: str
    text: str
    source: str
    title: str
    section: str
    source_dataset: str
    source_id: str
    source_url: str | None = None
    version: str | None = None
    document_date: str | None = None
    license: str | None = None
    license_note: str | None = None
    related_source: str | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    title: str
    section: str
    source_dataset: str
    source_id: str
    source_url: str | None = None
    version: str | None = None
    document_date: str | None = None
    license: str | None = None
    license_note: str | None = None
    related_source: str | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None


def _required_string(record: dict[str, object], key: str, record_id: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Synthetic document {record_id!r} has invalid {key!r}")
    return value.strip()


def _required_protocol_string(
    record: dict[str, object],
    key: str,
    record_id: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Approved protocol {record_id!r} has invalid {key!r}")
    return value.strip()


def _load_synthetic_json(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Synthetic document file must contain a list of objects: {path}")
    return data


def load_synthetic_documents(directory: Path) -> tuple[SourceDocument, ...]:
    """Load protocol and FAQ fixtures while requiring explicit synthetic markers."""
    protocol_path = directory / "protocols.json"
    faq_path = directory / "faq.json"
    documents: list[SourceDocument] = []

    for record in _load_synthetic_json(protocol_path):
        record_id = _required_string(record, "id", str(protocol_path))
        if record.get("synthetic") is not True:
            raise ValueError(f"Synthetic protocol {record_id!r} is not marked synthetic")
        documents.append(
            SourceDocument(
                id=f"synthetic:protocol:{record_id}",
                text=_required_string(record, "text", record_id),
                source=f"data/synthetic/{protocol_path.name}#{record_id}",
                title=_required_string(record, "title", record_id),
                section=_required_string(record, "section", record_id),
                source_dataset="synthetic",
                source_id=record_id,
                version=_required_string(record, "version", record_id),
                document_date=(
                    record["date"] if isinstance(record.get("date"), str) else None
                ),
                license="synthetic",
                synthetic=True,
            )
        )

    for record in _load_synthetic_json(faq_path):
        record_id = _required_string(record, "id", str(faq_path))
        if record.get("synthetic") is not True:
            raise ValueError(f"Synthetic FAQ {record_id!r} is not marked synthetic")
        question = _required_string(record, "question", record_id)
        answer = _required_string(record, "answer", record_id)
        related_source = _required_string(record, "source", record_id)
        documents.append(
            SourceDocument(
                id=f"synthetic:faq:{record_id}",
                text=f"Question: {question}\nAnswer: {answer}",
                source=f"data/synthetic/{faq_path.name}#{record_id}",
                title=question,
                section="faq",
                source_dataset="synthetic",
                source_id=record_id,
                license="synthetic",
                related_source=related_source,
                synthetic=True,
            )
        )

    return tuple(documents)


def load_approved_protocol_documents(path: Path) -> tuple[SourceDocument, ...]:
    """Load approved protocol-only JSONL documents; patient records are excluded."""
    if not path.is_file():
        raise FileNotFoundError(f"Approved protocol corpus does not exist: {path}")

    documents: list[SourceDocument] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid approved protocol JSONL at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Approved protocol record at line {line_number} must be an object"
            )

        record_id = _required_protocol_string(record, "id", str(line_number))
        source_id = _required_protocol_string(record, "source_id", record_id)
        if record_id in seen_ids or record.get("source_dataset") != "hospital_protocol":
            raise ValueError(f"Invalid or duplicate approved protocol at line {line_number}")
        seen_ids.add(record_id)
        if (
            record.get("institutional_approval_status") != "approved"
            or record.get("privacy_review_status") != "approved"
            or record.get("contains_patient_data") is not False
            or record.get("training_eligible") is not False
            or record.get("approval_attestation_source") != "user_provided"
        ):
            raise ValueError(f"Protocol approval metadata is invalid at line {line_number}")
        for date_field in ("institutional_approved_at", "privacy_reviewed_at"):
            date_value = _required_protocol_string(record, date_field, record_id)
            try:
                date.fromisoformat(date_value)
            except ValueError as exc:
                raise ValueError(
                    f"Protocol approval metadata has an invalid {date_field!r}"
                ) from exc
        _required_protocol_string(
            record,
            "institutional_approval_reference",
            record_id,
        )
        _required_protocol_string(record, "privacy_review_reference", record_id)
        source_metadata = record.get("source_metadata")
        if not isinstance(source_metadata, dict):
            raise ValueError(f"Protocol metadata is invalid at line {line_number}")
        documents.append(
            SourceDocument(
                id=record_id,
                text=_required_protocol_string(record, "text", record_id),
                source=_required_protocol_string(record, "source", record_id),
                title=_required_protocol_string(record, "title", record_id),
                section=_required_protocol_string(record, "section", record_id),
                source_dataset="hospital_protocol",
                source_id=source_id,
                version=_required_protocol_string(record, "version", record_id),
                license="internal",
                license_note=_required_protocol_string(
                    record,
                    "license_note",
                    record_id,
                ),
                synthetic=False,
            )
        )
    if not documents:
        raise ValueError(f"Approved protocol corpus is empty: {path}")
    return tuple(documents)


def load_medquad_documents(records: tuple[QARecord, ...] | list[QARecord]) -> tuple[SourceDocument, ...]:
    documents: list[SourceDocument] = []
    for record in records:
        if record.source_dataset != "medquad":
            raise ValueError(f"Expected MedQuAD record, got {record.source_dataset!r}")
        if not record.answer.strip():
            continue
        documents.append(
            SourceDocument(
                id=record.id,
                text=f"Question: {record.question}\nAnswer: {record.answer}",
                source=record.source_url or f"medquad:{record.source_id}",
                title=record.focus or record.question,
                section=record.question_type or "QAPair",
                source_dataset=record.source_dataset,
                source_id=record.source_id,
                source_url=record.source_url,
                license=record.license,
                license_note=record.license_note,
                synthetic=False,
            )
        )
    return tuple(documents)


def load_pubmedqa_documents(records: tuple[QARecord, ...] | list[QARecord]) -> tuple[SourceDocument, ...]:
    documents: list[SourceDocument] = []
    for record in records:
        if record.source_dataset != "pubmedqa" or record.source_subset != "PQA-L":
            raise ValueError("PubMedQA retrieval requires expert-labeled PQA-L records")
        if not record.context or not record.context.strip():
            raise ValueError(f"PubMedQA record {record.source_id!r} has no abstract context")
        documents.append(
            SourceDocument(
                id=f"pubmedqa:{record.source_id}",
                text=record.context,
                source=record.source_url or f"https://pubmed.ncbi.nlm.nih.gov/{record.source_id}/",
                title=f"PubMedQA abstract context for PMID {record.source_id}",
                section="abstract_context",
                source_dataset=record.source_dataset,
                source_id=record.source_id,
                source_url=record.source_url,
                license=record.license,
                license_note=record.license_note,
                synthetic=False,
            )
        )
    return tuple(documents)


def _chunk_document(
    document: SourceDocument,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[DocumentChunk, ...]:
    text = " ".join(document.text.split())
    if not text:
        return ()

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_number = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + 1, end + 1)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    id=f"{document.id}#chunk-{chunk_number:04d}",
                    text=chunk_text,
                    source=document.source,
                    title=document.title,
                    section=document.section,
                    source_dataset=document.source_dataset,
                    source_id=document.source_id,
                    source_url=document.source_url,
                    version=document.version,
                    document_date=document.document_date,
                    license=document.license,
                    license_note=document.license_note,
                    related_source=document.related_source,
                    synthetic=document.synthetic,
                )
            )
            chunk_number += 1

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return tuple(chunks)


def _tokens(text: str) -> set[str]:
    normalized_text = text
    for pattern, replacement in _TERM_REPLACEMENTS:
        normalized_text = pattern.sub(replacement, normalized_text)
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(normalized_text)
        if token.casefold() not in _STOPWORDS
    }


def _normalize_embedding(
    embedding: Sequence[float],
    *,
    label: str,
) -> tuple[float, ...]:
    if not embedding:
        raise ValueError(f"{label} embedding must not be empty")
    values = tuple(float(value) for value in embedding)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} embedding must contain only finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError(f"{label} embedding must not be a zero vector")
    return tuple(value / norm for value in values)


def _embedding_cache_fingerprint(
    embedding_model: TextEmbedder,
    texts: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(embedding_model.version.encode("utf-8"))
    digest.update(b"\0")
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_embedding_cache(
    path: Path,
    fingerprint: str,
    expected_count: int,
) -> tuple[tuple[float, ...], ...] | None:
    with path.open("rb") as stream:
        if stream.read(len(_EMBEDDING_CACHE_MAGIC)) != _EMBEDDING_CACHE_MAGIC:
            raise ValueError("Embedding cache has an invalid header")
        header_length = stream.read(4)
        if len(header_length) != 4:
            raise ValueError("Embedding cache header is incomplete")
        metadata_length = struct.unpack("<I", header_length)[0]
        if not 1 <= metadata_length <= 65536:
            raise ValueError("Embedding cache metadata has an invalid size")
        metadata_bytes = stream.read(metadata_length)
        if len(metadata_bytes) != metadata_length:
            raise ValueError("Embedding cache metadata is incomplete")
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Embedding cache metadata must be an object")
        cached_fingerprint = metadata.get("fingerprint")
        count = metadata.get("count")
        dimension = metadata.get("dimension")
        if (
            not isinstance(cached_fingerprint, str)
            or not isinstance(count, int)
            or not isinstance(dimension, int)
        ):
            raise ValueError("Embedding cache metadata is invalid")
        if cached_fingerprint != fingerprint:
            return None
        if count != expected_count or dimension < 1:
            raise ValueError("Embedding cache contains invalid vectors")
        if metadata.get("byteorder") != "little":
            raise ValueError("Embedding cache has an unsupported byte order")
        if array("f").itemsize != 4:
            raise RuntimeError("Embedding cache requires 32-bit float support")
        expected_bytes = count * dimension * 4
        vector_bytes = stream.read(expected_bytes)
        if len(vector_bytes) != expected_bytes or stream.read(1):
            raise ValueError("Embedding cache contains invalid vectors")
        cached_embeddings = array("f")
        cached_embeddings.frombytes(vector_bytes)
        if sys.byteorder != "little":
            cached_embeddings.byteswap()
        return tuple(
            tuple(float(value) for value in cached_embeddings[start:start + dimension])
            for start in range(0, len(cached_embeddings), dimension)
        )


def _write_embedding_cache(
    path: Path,
    fingerprint: str,
    embeddings: Sequence[Sequence[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if not embeddings:
        raise ValueError("Cannot cache an empty embedding collection")
    dimension = len(embeddings[0])
    if array("f").itemsize != 4:
        raise RuntimeError("Embedding cache requires 32-bit float support")
    flattened = array(
        "f",
        (float(value) for embedding in embeddings for value in embedding),
    )
    if sys.byteorder != "little":
        flattened.byteswap()
    metadata = json.dumps(
        {
            "fingerprint": fingerprint,
            "count": len(embeddings),
            "dimension": dimension,
            "byteorder": "little",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with temporary_path.open("wb") as stream:
            stream.write(_EMBEDDING_CACHE_MAGIC)
            stream.write(struct.pack("<I", len(metadata)))
            stream.write(metadata)
            stream.write(flattened.tobytes())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class DocumentIndex:
    """In-memory lexical index with optional local semantic retrieval."""

    def __init__(
        self,
        documents: tuple[SourceDocument, ...] | list[SourceDocument],
        *,
        chunk_size: int = 800,
        overlap: int = 100,
        embedding_model: TextEmbedder | None = None,
        embedding_cache_path: Path | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if embedding_cache_path is not None and embedding_model is None:
            raise ValueError("Embedding cache requires an embedding model")

        chunks = [
            chunk
            for document in documents
            for chunk in _chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        ]
        chunk_ids = [chunk.id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Document chunk IDs must be unique")
        self._chunks = tuple(chunks)
        self._tokens_by_chunk = tuple(_tokens(chunk.text) for chunk in chunks)
        self._embedding_model = embedding_model
        self._embeddings: tuple[tuple[float, ...], ...] = ()
        self._embedding_dimension: int | None = None
        if embedding_model is not None and self._chunks:
            embedding_texts = tuple(
                f"{chunk.title}. {chunk.section}. {chunk.text}"
                for chunk in self._chunks
            )
            fingerprint = _embedding_cache_fingerprint(
                embedding_model,
                embedding_texts,
            )
            raw_embeddings = None
            if (
                embedding_cache_path is not None
                and embedding_cache_path.is_file()
            ):
                raw_embeddings = _read_embedding_cache(
                    embedding_cache_path,
                    fingerprint,
                    len(self._chunks),
                )
            cache_miss = raw_embeddings is None
            if raw_embeddings is None:
                raw_embeddings = embedding_model.embed_documents(embedding_texts)
            if len(raw_embeddings) != len(self._chunks):
                raise ValueError(
                    "Embedding model must return one vector per document chunk"
                )
            self._embeddings = tuple(
                _normalize_embedding(embedding, label=f"Document {index}")
                for index, embedding in enumerate(raw_embeddings)
            )
            self._embedding_dimension = len(self._embeddings[0])
            if any(
                len(embedding) != self._embedding_dimension
                for embedding in self._embeddings
            ):
                raise ValueError("Document embeddings must have a consistent dimension")
            if cache_miss and embedding_cache_path is not None:
                _write_embedding_cache(
                    embedding_cache_path,
                    fingerprint,
                    self._embeddings,
                )

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return self._chunks

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.5,
        min_semantic_score: float = 0.55,
    ) -> tuple[SearchResult, ...]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        if not -1 <= min_semantic_score <= 1:
            raise ValueError("min_semantic_score must be between minus one and one")
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()

        lexical_scores = tuple(
            len(query_tokens & chunk_tokens) / len(query_tokens)
            for chunk_tokens in self._tokens_by_chunk
        )
        if self._embedding_model is None:
            results = [
                SearchResult(
                    chunk=chunk,
                    score=score,
                    lexical_score=score,
                )
                for chunk, score in zip(self._chunks, lexical_scores)
                if score >= min_score and score > 0
            ]
        else:
            query_embedding = _normalize_embedding(
                self._embedding_model.embed_query(query),
                label="Query",
            )
            if len(query_embedding) != self._embedding_dimension:
                raise ValueError("Query and document embeddings must have the same dimension")
            results = []
            for chunk, lexical_score, document_embedding in zip(
                self._chunks,
                lexical_scores,
                self._embeddings,
            ):
                semantic_score = sum(
                    query_value * document_value
                    for query_value, document_value in zip(
                        query_embedding,
                        document_embedding,
                    )
                )
                if (
                    lexical_score < min_score
                    and semantic_score < min_semantic_score
                ):
                    continue
                score = (lexical_score + max(0.0, semantic_score)) / 2
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        lexical_score=lexical_score,
                        semantic_score=semantic_score,
                    )
                )
        results.sort(key=lambda result: (-result.score, result.chunk.id))
        return tuple(results[:limit])
