import json
from pathlib import Path

import pytest

from medical_assistant.datasets import (
    QARecord,
    deduplicate_records,
    load_medquad,
    load_pubmedqa,
)


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = ROOT / "data" / "synthetic"


def test_load_pubmedqa_normalizes_records_and_tracks_label_provenance() -> None:
    result = load_pubmedqa(SYNTHETIC / "pubmedqa_sample.json", subset="PQA-L")

    assert len(result.records) == 2
    assert result.records[0].source_dataset == "pubmedqa"
    assert result.records[0].source_subset == "PQA-L"
    assert result.records[0].label == "yes"
    assert result.records[0].label_provenance == "expert_labeled"
    assert result.records[0].evaluation_eligible is True
    assert "underlying abstract" in result.records[0].license_note
    assert result.records[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/PMID-SYNTH-001/"
    assert "\n\n" not in result.records[0].context


def test_load_pubmedqa_preserves_source_order(tmp_path: Path) -> None:
    path = tmp_path / "source-order.json"
    records = {
        "PMID-Z": {
            "QUESTION": "Pergunta Z?",
            "CONTEXTS": ["Contexto Z."],
            "LONG_ANSWER": "Resposta Z.",
            "final_decision": "yes",
        },
        "PMID-A": {
            "QUESTION": "Pergunta A?",
            "CONTEXTS": ["Contexto A."],
            "LONG_ANSWER": "Resposta A.",
            "final_decision": "no",
        },
    }
    path.write_text(json.dumps(records), encoding="utf-8")

    result = load_pubmedqa(path)

    assert [record.source_id for record in result.records] == ["PMID-Z", "PMID-A"]


def test_pubmedqa_artificial_subset_is_explicitly_marked() -> None:
    result = load_pubmedqa(SYNTHETIC / "pubmedqa_sample.json", subset="PQA-A")

    assert {record.label_provenance for record in result.records} == {
        "artificially_generated"
    }
    assert all(record.evaluation_eligible is False for record in result.records)


def test_pubmedqa_rejects_unknown_labels(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps(
            {
                "PMID-SYNTH-003": {
                    "QUESTION": "Pergunta fictícia?",
                    "CONTEXTS": ["Contexto fictício."],
                    "LONG_ANSWER": "Resposta fictícia.",
                    "final_decision": "unknown",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported label"):
        load_pubmedqa(invalid_path)


def test_load_medquad_preserves_metadata_and_reports_empty_answers() -> None:
    result = load_medquad(SYNTHETIC / "medquad_sample.xml")

    assert len(result.records) == 1
    assert result.skipped_missing_answers == 1
    record = result.records[0]
    assert record.source_dataset == "medquad"
    assert record.source_id == "Q-SYNTH-001"
    assert record.source_origin == "Synthetic"
    assert record.question_type == "information"
    assert record.focus == "Fixture de demonstração"
    assert record.license == "CC BY 4.0"
    assert record.source_url == "https://example.invalid/demo"


def test_medquad_loads_xml_files_in_a_directory(tmp_path: Path) -> None:
    source_file = SYNTHETIC / "medquad_sample.xml"
    (tmp_path / "sample.xml").write_bytes(source_file.read_bytes())
    excluded_dir = tmp_path / "11_MPlusDrugs_QA"
    excluded_dir.mkdir()
    (excluded_dir / "excluded.xml").write_bytes(source_file.read_bytes())

    result = load_medquad(tmp_path)

    assert len(result.records) == 1
    assert result.skipped_missing_answers == 1
    assert result.skipped_excluded_files == 1


def test_medquad_rejects_direct_file_from_excluded_subset(tmp_path: Path) -> None:
    excluded_dir = tmp_path / "12_MPlusHerbsSupplements_QA"
    excluded_dir.mkdir()
    excluded_file = excluded_dir / "excluded.xml"
    excluded_file.write_bytes(
        (SYNTHETIC / "medquad_sample.xml").read_bytes()
    )

    with pytest.raises(ValueError, match="excluded by its copyright notice"):
        load_medquad(excluded_file)


def test_medquad_counts_documents_without_question_answer_pairs(tmp_path: Path) -> None:
    empty_document = tmp_path / "empty.xml"
    empty_document.write_text(
        '<Document id="empty" source="NIDDK"><QAPairs /></Document>',
        encoding="utf-8",
    )

    result = load_medquad(empty_document)

    assert result.records == ()
    assert result.skipped_empty_documents == 1


@pytest.mark.parametrize(
    ("xml", "expected_source", "expected_focus"),
    [
        (
            '<doc docid="doc-1" corpus="NINDS" url="https://example.invalid">'
            "<doctitle-focus>Focus lowercase</doctitle-focus>"
            "<qaPairs><pair><question qid=\"q-1\" qtype=\"information\">Question?</question>"
            "<answer>Answer.</answer></pair></qaPairs></doc>",
            "NINDS",
            "Focus lowercase",
        ),
        (
            '<DiseaseFile fid="doc-2" source="CDC" url="https://example.invalid">'
            "<Focus>Focus disease</Focus>"
            "<QAPairs><QAPair><Question qid=\"q-2\" qtype=\"treatment\">Question?</Question>"
            "<Answer>Answer.</Answer></QAPair></QAPairs></DiseaseFile>",
            "CDC",
            "Focus disease",
        ),
    ],
)
def test_medquad_loads_supported_xml_variants(
    tmp_path: Path,
    xml: str,
    expected_source: str,
    expected_focus: str,
) -> None:
    source_file = tmp_path / "variant.xml"
    source_file.write_text(xml, encoding="utf-8")

    result = load_medquad(source_file)

    assert len(result.records) == 1
    assert result.records[0].source_origin == expected_source
    assert result.records[0].focus == expected_focus


def test_medquad_rejects_dtd_declarations(tmp_path: Path) -> None:
    malicious_xml = tmp_path / "unsafe.xml"
    malicious_xml.write_text(
        '<!DOCTYPE Document [<!ENTITY x "x">]><Document id="x"><QAPairs /></Document>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DTD or entities"):
        load_medquad(malicious_xml)


def test_deduplicate_records_is_deterministic_and_keeps_distinct_contexts() -> None:
    first = QARecord(
        id="one",
        question=" Pergunta   Fictícia? ",
        answer="Resposta.",
        context="Contexto A",
        source_dataset="synthetic",
        source_id="one",
        license="synthetic",
    )
    duplicate = QARecord(
        id="two",
        question="pergunta fictícia?",
        answer="Resposta.",
        context="Contexto A",
        source_dataset="synthetic",
        source_id="two",
        license="synthetic",
    )
    distinct_context = QARecord(
        id="three",
        question="Pergunta Fictícia?",
        answer="Resposta.",
        context="Contexto B",
        source_dataset="synthetic",
        source_id="three",
        license="synthetic",
    )

    result = deduplicate_records([first, duplicate, distinct_context])

    assert [record.id for record in result.records] == ["one", "three"]
    assert result.duplicate_sources == (("two", "one"),)


def test_deduplicate_records_maps_repeated_duplicate_to_original_record() -> None:
    first = QARecord(
        id="one",
        question="Pergunta?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="one",
        license="synthetic",
    )
    duplicate = QARecord(
        id="two",
        question="Pergunta?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="two",
        license="synthetic",
    )

    result = deduplicate_records([first, duplicate, duplicate])

    assert result.duplicate_sources == (("two", "one"), ("two", "one"))


def test_deduplicate_records_keeps_conflicting_labels_separate() -> None:
    yes_record = QARecord(
        id="yes",
        question="Pergunta?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="yes",
        license="synthetic",
        label="yes",
    )
    no_record = QARecord(
        id="no",
        question="Pergunta?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="no",
        license="synthetic",
        label="no",
    )

    result = deduplicate_records([yes_record, no_record])

    assert [record.id for record in result.records] == ["yes", "no"]
    assert result.duplicate_sources == ()


def test_deduplicate_records_rejects_conflicting_duplicate_ids() -> None:
    first = QARecord(
        id="same",
        question="Pergunta A?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="same",
        license="synthetic",
    )
    conflicting = QARecord(
        id="same",
        question="Pergunta B?",
        answer="Resposta.",
        source_dataset="synthetic",
        source_id="same",
        license="synthetic",
    )

    with pytest.raises(ValueError, match="Conflicting records"):
        deduplicate_records([first, conflicting])
