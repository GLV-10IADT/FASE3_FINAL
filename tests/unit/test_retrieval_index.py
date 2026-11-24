from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from medical_assistant.datasets import QARecord, load_medquad
from medical_assistant.retrieval import (
    DocumentIndex,
    SourceDocument,
    load_medquad_documents,
    load_pubmedqa_documents,
    load_synthetic_documents,
)


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"


class _FakeEmbedder:
    @property
    def version(self) -> str:
        return "fake-embedder-v1"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0) if index == 0 else (0.0, 1.0)
            for index in range(len(texts))
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (0.99, 0.1) if "chest" in text else (0.0, 1.0)


class _CountingEmbedder(_FakeEmbedder):
    def __init__(self) -> None:
        self.document_embedding_calls = 0

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        self.document_embedding_calls += 1
        return super().embed_documents(texts)


def test_synthetic_ingestion_preserves_source_title_section_and_synthetic_flag() -> None:
    documents = load_synthetic_documents(SYNTHETIC_DIR)
    index = DocumentIndex(documents)

    results = index.search("identificador origem documento", limit=3)

    assert results
    for result in results:
        assert result.chunk.source.startswith("data/synthetic/")
        assert result.chunk.title
        assert result.chunk.section
        assert result.chunk.source_id
        assert result.chunk.synthetic is True


def test_medquad_ingestion_preserves_published_provenance() -> None:
    records = load_medquad(SYNTHETIC_DIR / "medquad_sample.xml").records
    documents = load_medquad_documents(records)
    index = DocumentIndex(documents)

    results = index.search("finalidade arquivo fictício", limit=2)

    assert results
    chunk = results[0].chunk
    assert chunk.source == "https://example.invalid/demo"
    assert chunk.title == "Fixture de demonstração"
    assert chunk.section == "information"
    assert chunk.source_id == "Q-SYNTH-001"
    assert chunk.source_dataset == "medquad"
    assert chunk.license == "CC BY 4.0"
    assert chunk.synthetic is False


def test_pubmedqa_ingestion_indexes_only_abstract_context_with_provenance() -> None:
    record = QARecord(
        id="pubmedqa:pqal:123",
        question="Question?",
        answer="Reference answer.",
        source_dataset="pubmedqa",
        source_id="123",
        source_subset="PQA-L",
        context="Abstract evidence.",
        license="MIT",
        license_note="Check underlying article rights.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123/",
    )

    documents = load_pubmedqa_documents([record])
    index = DocumentIndex(documents)

    assert len(documents) == 1
    assert documents[0].text == "Abstract evidence."
    assert documents[0].source == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert documents[0].source_id == "123"
    assert documents[0].license_note == "Check underlying article rights."
    assert "Reference answer" not in index.chunks[0].text


def test_curated_queries_retrieve_expected_source_records() -> None:
    medquad_records = load_medquad(SYNTHETIC_DIR / "medquad_sample.xml").records
    documents = (
        *load_synthetic_documents(SYNTHETIC_DIR),
        *load_medquad_documents(medquad_records),
    )
    index = DocumentIndex(documents)
    expected_source_ids = {
        "recuperação citação fontes": "DOC-SYNTH-001",
        "versão seção": "FAQ-SYNTH-001",
        "finalidade arquivo fictício": "Q-SYNTH-001",
    }

    for query, expected_source_id in expected_source_ids.items():
        results = index.search(query)
        assert results
        assert results[0].chunk.source_id == expected_source_id


def test_search_abstains_when_corpus_has_no_sufficient_evidence() -> None:
    index = DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR))

    assert index.search("xylophytic zephyr nanoparticle") == ()
    assert index.search("recuperação xylophytic zephyr") == ()


def test_search_normalizes_common_translation_variants_for_pubmedqa_terms() -> None:
    document = SourceDocument(
        id="pubmedqa-synthetic:aquagenic",
        text=(
            "Infants had events in the bath; an equivalent of aquagenic "
            "urticaria was considered."
        ),
        source="synthetic-pubmedqa:aquagenic",
        title="Synthetic abstract",
        section="abstract_context",
        source_dataset="synthetic",
        source_id="PUBMEDQA-SYNTH-AQUAGENIC",
        synthetic=True,
    )
    index = DocumentIndex([document])

    results = index.search(
        "Can baby fainting during bathing be related to aquatic urticaria?"
    )

    assert results
    assert results[0].chunk.source_id == "PUBMEDQA-SYNTH-AQUAGENIC"


def test_hybrid_search_finds_a_semantic_match_without_shared_query_terms() -> None:
    documents = [
        SourceDocument(
            id="semantic-target",
            text="Cardiac symptoms and relevant source information.",
            source="fixture://target",
            title="Target",
            section="demo",
            source_dataset="synthetic",
            source_id="TARGET",
            synthetic=True,
        ),
        SourceDocument(
            id="semantic-distractor",
            text="Unrelated topic and information.",
            source="fixture://distractor",
            title="Distractor",
            section="demo",
            source_dataset="synthetic",
            source_id="DISTRACTOR",
            synthetic=True,
        ),
    ]
    index = DocumentIndex(documents, embedding_model=_FakeEmbedder())

    results = index.search("chest discomfort", min_semantic_score=0.8)

    assert results
    assert results[0].chunk.source_id == "TARGET"
    assert results[0].lexical_score == 0
    assert results[0].semantic_score == pytest.approx(0.9949, abs=0.001)


def test_hybrid_search_rejects_inconsistent_embedding_dimensions() -> None:
    class _InconsistentEmbedder(_FakeEmbedder):
        def embed_documents(
            self,
            texts: Sequence[str],
        ) -> tuple[tuple[float, ...], ...]:
            return ((1.0, 0.0), (0.0, 1.0, 0.0))[:len(texts)]

    documents = [
        SourceDocument(
            id="one",
            text="first",
            source="fixture://one",
            title="One",
            section="demo",
            source_dataset="synthetic",
            source_id="ONE",
            synthetic=True,
        ),
        SourceDocument(
            id="two",
            text="second",
            source="fixture://two",
            title="Two",
            section="demo",
            source_dataset="synthetic",
            source_id="TWO",
            synthetic=True,
        ),
    ]

    with pytest.raises(ValueError, match="consistent dimension"):
        DocumentIndex(documents, embedding_model=_InconsistentEmbedder())


def test_document_index_reuses_persisted_embeddings(tmp_path: Path) -> None:
    document = SourceDocument(
        id="cached-document",
        text="Heart health evidence.",
        source="fixture://cached",
        title="Heart",
        section="demo",
        source_dataset="synthetic",
        source_id="HEART",
        synthetic=True,
    )
    cache_path = tmp_path / "retrieval" / "embeddings.bin"
    first_embedder = _CountingEmbedder()
    DocumentIndex(
        [document],
        embedding_model=first_embedder,
        embedding_cache_path=cache_path,
    )

    second_embedder = _CountingEmbedder()
    cached_index = DocumentIndex(
        [document],
        embedding_model=second_embedder,
        embedding_cache_path=cache_path,
    )

    assert first_embedder.document_embedding_calls == 1
    assert second_embedder.document_embedding_calls == 0
    assert cached_index.search("heart", min_semantic_score=0.0)


def test_document_index_rebuilds_cache_when_embedding_text_changes(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "embeddings.bin"
    original = SourceDocument(
        id="versioned-document",
        text="Original evidence.",
        source="fixture://versioned",
        title="Evidence",
        section="demo",
        source_dataset="synthetic",
        source_id="VERSIONED",
        synthetic=True,
    )
    first_embedder = _CountingEmbedder()
    DocumentIndex(
        [original],
        embedding_model=first_embedder,
        embedding_cache_path=cache_path,
    )

    second_embedder = _CountingEmbedder()
    DocumentIndex(
        [replace(original, text="Updated evidence.")],
        embedding_model=second_embedder,
        embedding_cache_path=cache_path,
    )

    assert first_embedder.document_embedding_calls == 1
    assert second_embedder.document_embedding_calls == 1


def test_search_validates_minimum_score() -> None:
    index = DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR))

    with pytest.raises(ValueError, match="min_score must be between"):
        index.search("documento", min_score=1.1)


def test_chunking_preserves_metadata_on_every_chunk() -> None:
    document = SourceDocument(
        id="synthetic:long",
        text="one two three four five six seven eight nine ten eleven twelve",
        source="fixture.json#long",
        title="Long fixture",
        section="demo",
        source_dataset="synthetic",
        source_id="long",
        synthetic=True,
    )
    index = DocumentIndex([document], chunk_size=20, overlap=5)

    assert len(index.chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in index.chunks)
    assert all(
        chunk.source == document.source
        and chunk.title == document.title
        and chunk.section == document.section
        and chunk.source_id == document.source_id
        for chunk in index.chunks
    )


def test_index_rejects_duplicate_chunk_ids() -> None:
    document = SourceDocument(
        id="same",
        text="some text",
        source="fixture.json#same",
        title="Fixture",
        section="demo",
        source_dataset="synthetic",
        source_id="same",
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        DocumentIndex([document, document])


def test_search_handles_empty_queries_and_validates_limit() -> None:
    index = DocumentIndex(load_synthetic_documents(SYNTHETIC_DIR))

    assert index.search("   ") == ()
    with pytest.raises(ValueError, match="limit must be greater"):
        index.search("document", limit=0)


def test_index_validates_chunk_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        DocumentIndex([], chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        DocumentIndex([], chunk_size=10, overlap=10)
