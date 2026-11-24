import json
from pathlib import Path

import medical_assistant


ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_with_version() -> None:
    assert medical_assistant.__version__ == "0.1.0"


def test_source_manifest_records_pubmedqa_download_and_medquad_status() -> None:
    manifest = json.loads((ROOT / "data" / "source_manifest.json").read_text(encoding="utf-8"))

    datasets = {dataset["name"]: dataset for dataset in manifest["datasets"]}

    assert set(datasets) == {"PubMedQA", "MedQuAD"}
    assert datasets["PubMedQA"]["status"] == "downloaded_local_ignored"
    assert datasets["PubMedQA"]["planned_subsets"] == ["PQA-L"]
    assert datasets["PubMedQA"]["revision"]
    assert datasets["PubMedQA"]["sha256"]
    assert datasets["MedQuAD"]["status"] == "downloaded_local_ignored"
    assert datasets["MedQuAD"]["revision"]
    assert datasets["MedQuAD"]["sha256"]
    assert datasets["MedQuAD"]["records_loaded"] == 16423
    assert datasets["MedQuAD"]["excluded_xml_files"] == 5777
    assert datasets["MedQuAD"]["excluded_subsets"] == [
        "10_MPlus_ADAM_QA",
        "11_MPlusDrugs_QA",
        "12_MPlusHerbsSupplements_QA",
    ]


def test_demo_fixtures_are_explicitly_synthetic() -> None:
    synthetic_dir = ROOT / "data" / "synthetic"
    for path in synthetic_dir.glob("*.json"):
        records = json.loads(path.read_text(encoding="utf-8"))
        assert records
        if isinstance(records, dict):
            records = records.values()
        assert all(record["synthetic"] is True for record in records)


def test_patient_fixture_has_no_real_patient_identifier() -> None:
    patients = json.loads(
        (ROOT / "data" / "synthetic" / "patients.json").read_text(encoding="utf-8")
    )

    assert all(patient["patient_id"].startswith("PATIENT-SYNTH-") for patient in patients)
