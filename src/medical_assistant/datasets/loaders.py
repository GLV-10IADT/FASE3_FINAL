import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from medical_assistant.datasets.models import DatasetLoadResult, QARecord

PubMedQASubset = Literal["PQA-L", "PQA-A"]
_PUBMEDQA_LABELS = {"yes", "no", "maybe"}
_WHITESPACE = re.compile(r"\s+")
_MEDQUAD_EXCLUDED_SUBSETS = {
    "10_mplus_adam_qa",
    "11_mplusdrugs_qa",
    "12_mplusherbssupplements_qa",
}


def _clean_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _required_string(value: object, field: str, record_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {record_id!r} has a missing or invalid {field!r}")
    return _clean_text(value)


def load_pubmedqa(
    path: Path,
    subset: PubMedQASubset = "PQA-L",
) -> DatasetLoadResult:
    """Load PubMedQA in source order without downloading it."""
    if subset not in ("PQA-L", "PQA-A"):
        raise ValueError(f"Unsupported PubMedQA subset: {subset!r}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("PubMedQA JSON must contain a mapping keyed by PMID")

    records: list[QARecord] = []
    for pmid, item in raw_data.items():
        if not isinstance(item, dict):
            raise ValueError(f"PubMedQA record {pmid!r} must be an object")

        question = _required_string(item.get("QUESTION"), "QUESTION", pmid)
        answer = _required_string(item.get("LONG_ANSWER"), "LONG_ANSWER", pmid)
        label = _required_string(item.get("final_decision"), "final_decision", pmid).lower()
        if label not in _PUBMEDQA_LABELS:
            raise ValueError(
                f"PubMedQA record {pmid!r} has unsupported label {label!r}"
            )

        contexts = item.get("CONTEXTS")
        if (
            not isinstance(contexts, list)
            or not contexts
            or any(not isinstance(context, str) or not context.strip() for context in contexts)
        ):
            raise ValueError(f"PubMedQA record {pmid!r} has invalid CONTEXTS")

        records.append(
            QARecord(
                id=f"pubmedqa:{subset.casefold()}:{pmid}",
                question=question,
                answer=answer,
                source_dataset="pubmedqa",
                source_id=pmid,
                license="MIT",
                license_note=(
                    "Repository license; verify rights for underlying abstract/source records."
                ),
                source_group_id=pmid,
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                context="\n\n".join(_clean_text(context) for context in contexts),
                label=label,
                source_subset=subset,
                label_provenance=(
                    "expert_labeled" if subset == "PQA-L" else "artificially_generated"
                ),
                evaluation_eligible=subset == "PQA-L",
            )
        )

    return DatasetLoadResult(records=tuple(records))


def _element_text(element: ET.Element | None, field: str, record_id: str) -> str:
    if element is None:
        raise ValueError(f"MedQuAD record {record_id!r} is missing {field!r}")
    return _clean_text(" ".join(element.itertext()))


def _parse_medquad_file(path: Path) -> tuple[list[QARecord], int, int]:
    content = path.read_bytes()
    upper_content = content.upper()
    if b"<!DOCTYPE" in upper_content or b"<!ENTITY" in upper_content:
        raise ValueError(f"MedQuAD XML must not declare a DTD or entities: {path}")

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid MedQuAD XML in {path}: {exc}") from exc

    if root.tag == "Document":
        document_id = root.attrib.get("id")
        focus_element = root.find("Focus")
        qa_pairs = root.findall("./QAPairs/QAPair")
        question_tag = "Question"
        answer_tag = "Answer"
    elif root.tag == "DiseaseFile":
        document_id = root.attrib.get("fid")
        focus_element = root.find("Focus")
        qa_pairs = root.findall("./QAPairs/QAPair")
        question_tag = "Question"
        answer_tag = "Answer"
    elif root.tag == "doc":
        document_id = root.attrib.get("docid")
        focus_element = root.find("doctitle-focus")
        qa_pairs = root.findall("./qaPairs/pair")
        question_tag = "question"
        answer_tag = "answer"
    else:
        raise ValueError(f"Unsupported MedQuAD XML root {root.tag!r}: {path}")
    if not document_id:
        raise ValueError(f"MedQuAD document has no id: {path}")

    if not qa_pairs:
        return [], 0, 1

    focus = _clean_text(" ".join(focus_element.itertext())) if focus_element is not None else None
    source_origin = root.attrib.get("source") or root.attrib.get("corpus")
    source_url = root.attrib.get("url")
    source_scope = (
        None
        if source_origin == "Synthetic"
        else ":".join(
            (
                source_origin or "unknown",
                path.parent.name,
                path.stem,
            )
        )
    )
    records: list[QARecord] = []
    skipped_missing_answers = 0

    for qa_pair in qa_pairs:
        question_element = qa_pair.find(question_tag)
        question_id = question_element.attrib.get("qid") if question_element is not None else None
        if not question_id:
            raise ValueError(f"MedQuAD document {document_id!r} has a question without qid")

        question = _element_text(question_element, "Question", question_id)
        answer_element = qa_pair.find(answer_tag)
        answer = (
            _clean_text(" ".join(answer_element.itertext()))
            if answer_element is not None
            else ""
        )
        if not answer:
            skipped_missing_answers += 1
            continue

        records.append(
            QARecord(
                id=(
                    f"medquad:{document_id}:{question_id}"
                    if source_scope is None
                    else f"medquad:{source_scope}:{document_id}:{question_id}"
                ),
                question=question,
                answer=answer,
                source_dataset="medquad",
                source_id=(
                    question_id
                    if source_scope is None
                    else f"{source_scope}:{question_id}"
                ),
                license="CC BY 4.0",
                source_group_id=(
                    document_id
                    if source_scope is None
                    else f"{source_scope}:{document_id}"
                ),
                source_url=source_url,
                question_type=question_element.attrib.get("qtype"),
                focus=focus,
                source_origin=source_origin,
                source_metadata=(
                    None
                    if source_scope is None
                    else {
                        "source_directory": path.parent.name,
                        "original_document_id": document_id,
                        "original_question_id": question_id,
                    }
                ),
            )
        )

    return records, skipped_missing_answers, 0


def _is_excluded_medquad_path(path: Path) -> bool:
    return any(part.casefold() in _MEDQUAD_EXCLUDED_SUBSETS for part in path.parts)


def load_medquad(path: Path) -> DatasetLoadResult:
    """Load one MedQuAD XML file or all XML files under a directory."""
    if path.is_file():
        if _is_excluded_medquad_path(path):
            raise ValueError(f"MedQuAD subset is excluded by its copyright notice: {path}")
        xml_paths = [path]
        skipped_excluded_files = 0
    elif path.is_dir():
        all_xml_paths = sorted(path.rglob("*.xml"))
        excluded_paths = [
            xml_path for xml_path in all_xml_paths if _is_excluded_medquad_path(xml_path)
        ]
        xml_paths = [
            xml_path
            for xml_path in all_xml_paths
            if not _is_excluded_medquad_path(xml_path)
        ]
        skipped_excluded_files = len(excluded_paths)
        if not xml_paths:
            raise ValueError(f"No eligible MedQuAD XML files found in {path}")
    else:
        raise FileNotFoundError(path)

    records: list[QARecord] = []
    skipped_missing_answers = 0
    skipped_empty_documents = 0
    for xml_path in xml_paths:
        file_records, skipped, empty_documents = _parse_medquad_file(xml_path)
        records.extend(file_records)
        skipped_missing_answers += skipped
        skipped_empty_documents += empty_documents

    return DatasetLoadResult(
        records=tuple(records),
        skipped_missing_answers=skipped_missing_answers,
        skipped_excluded_files=skipped_excluded_files,
        skipped_empty_documents=skipped_empty_documents,
    )
