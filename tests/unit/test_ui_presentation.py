from medical_assistant.assistant import (
    AssistantResponse,
    ReviewStatus,
    SourceCitation,
)
from medical_assistant.patient_data import PatientRecord, PatientRecordItem
from medical_assistant.ui.presentation import to_ui_result
from medical_assistant.workflows import (
    RequestIntent,
    SourceDetail,
    WorkflowResult,
    WorkflowStatus,
)


def test_ui_result_hides_patient_record_content_and_metadata() -> None:
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETE,
        intent=RequestIntent.PATIENT_QUERY,
        response=AssistantResponse(
            answer="Resposta baseada no registro selecionado.",
            sources=(
                SourceCitation(
                    source="private/path.jsonl",
                    title="PRIVATE-TITLE",
                    section="PRIVATE-SECTION",
                    source_id="PRIVATE-SOURCE-ID",
                ),
            ),
            uncertainty="Revisão humana necessária.",
            review_status=ReviewStatus.PENDING,
        ),
        source_details=(
            SourceDetail(
                citation=SourceCitation(
                    source="private/path.jsonl",
                    title="PRIVATE-TITLE",
                    section="PRIVATE-SECTION",
                    source_id="PRIVATE-SOURCE-ID",
                ),
                patient_record=True,
            ),
        ),
        patient_record=PatientRecord(
            patient_id="PRIVATE-PATIENT-ID",
            display_label="PRIVATE-LABEL",
            records=(
                PatientRecordItem(
                    record_id="PRIVATE-SOURCE-ID",
                    record_type="PRIVATE-SECTION",
                    content="PRIVATE-CLINICAL-CONTENT",
                ),
            ),
            source="private/path.jsonl",
        ),
    )

    ui_result = to_ui_result(result)
    serialized = repr(ui_result)

    assert ui_result.sources[0].patient_record
    assert ui_result.sources[0].excerpt is None
    assert ui_result.sources[0].lexical_score is None
    assert ui_result.sources[0].semantic_score is None
    for private_value in (
        "PRIVATE-TITLE",
        "PRIVATE-SECTION",
        "PRIVATE-SOURCE-ID",
        "PRIVATE-PATIENT-ID",
        "PRIVATE-LABEL",
        "PRIVATE-CLINICAL-CONTENT",
        "private/path.jsonl",
    ):
        assert private_value not in serialized
