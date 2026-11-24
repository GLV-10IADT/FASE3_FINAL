import pytest

from medical_assistant.safety import SafetyPolicy, SafetyReason


@pytest.mark.parametrize(
    ("question", "expected_reason"),
    [
        ("Qual medicamento devo prescrever?", SafetyReason.CLINICAL_ACTION),
        ("Defina a dose para este paciente.", SafetyReason.CLINICAL_ACTION),
        ("Recomende um procedimento.", SafetyReason.CLINICAL_ACTION),
        ("Can I change this medication?", SafetyReason.CLINICAL_ACTION),
        ("Diagnostique esta condição.", SafetyReason.CLINICAL_ACTION),
        ("Estou com dor no peito.", SafetyReason.URGENT_SIGNAL),
        ("Patient has shortness of breath.", SafetyReason.URGENT_SIGNAL),
    ],
)
def test_policy_requires_review_for_actions_and_urgent_signals(
    question: str,
    expected_reason: SafetyReason,
) -> None:
    assessment = SafetyPolicy().assess(question)

    assert assessment.requires_human_review is True
    assert assessment.reason is expected_reason


def test_policy_allows_informational_definition_without_action_cues() -> None:
    assessment = SafetyPolicy().assess("O que significa tratamento em um protocolo?")

    assert assessment.requires_human_review is False
    assert assessment.reason is None


def test_policy_handles_empty_and_non_string_input_explicitly() -> None:
    assert SafetyPolicy().assess("").requires_human_review is False
    with pytest.raises(TypeError):
        SafetyPolicy().assess(None)  # type: ignore[arg-type]
