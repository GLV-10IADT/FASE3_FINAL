from medical_assistant.ui.intake import (
    IntakeStep,
    SyntheticCaseInterview,
)


def test_synthetic_interview_requires_consent_and_can_be_canceled() -> None:
    interview = SyntheticCaseInterview()

    invalid = interview.advance("talvez")
    assert invalid.state.step is IntakeStep.CONSENT
    assert not invalid.ready_to_review

    canceled = interview.advance("não")
    assert canceled.state.step is IntakeStep.CANCELED
    assert not canceled.ready_to_review
    assert "nenhum caso foi processado" in canceled.message


def test_synthetic_interview_collects_answers_and_requires_final_confirmation() -> None:
    interview = SyntheticCaseInterview()
    answers = (
        "SIM",
        "adulto",
        "dor de cabeça",
        "náusea",
        "há dois dias",
        "PULAR",
        "PULAR",
    )

    for answer in answers:
        transition = interview.advance(answer)
        interview = transition.state

    assert interview.step is IntakeStep.FINAL_CONFIRMATION
    assert "Digite SIM" in transition.message
    assert "Histórico relevante" not in interview.summary()
    assert interview.retrieval_query() == "dor de cabeça náusea há dois dias"

    confirmation = interview.advance("sim")

    assert confirmation.ready_to_review
    assert confirmation.state.step is IntakeStep.COMPLETE
    assert "Queixa principal: dor de cabeça" in confirmation.state.summary()


def test_synthetic_interview_retries_invalid_values_without_advancing() -> None:
    interview = SyntheticCaseInterview()
    interview = interview.advance("sim").state

    invalid_age = interview.advance("50 anos")
    assert invalid_age.state.step is IntakeStep.AGE_RANGE

    interview = interview.advance("adulto").state
    invalid_complaint = interview.advance("x" * 241)
    assert invalid_complaint.state.step is IntakeStep.COMPLAINT
    assert "Limite de 240 caracteres" in invalid_complaint.message
