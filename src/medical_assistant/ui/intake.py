from dataclasses import dataclass, replace
from enum import StrEnum


class IntakeStep(StrEnum):
    CONSENT = "consent"
    AGE_RANGE = "age_range"
    COMPLAINT = "complaint"
    SYMPTOMS = "symptoms"
    DURATION = "duration"
    HISTORY = "history"
    EXAMS = "exams"
    FINAL_CONFIRMATION = "final_confirmation"
    COMPLETE = "complete"
    CANCELED = "canceled"


@dataclass(frozen=True)
class IntakeTransition:
    state: "SyntheticCaseInterview"
    message: str
    ready_to_review: bool = False


@dataclass(frozen=True)
class SyntheticCaseInterview:
    step: IntakeStep = IntakeStep.CONSENT
    age_range: str = ""
    complaint: str = ""
    symptoms: str = ""
    duration: str = ""
    history: str = ""
    exams: str = ""

    @staticmethod
    def opening_message() -> str:
        return (
            "Entrevista acadêmica: use somente informações inventadas. Não "
            "informe nomes, documentos ou dados de pacientes reais. "
            "Digite SIM para confirmar o uso exclusivo de um caso sintético."
        )

    def advance(self, answer: str) -> IntakeTransition:
        value = answer.strip()
        normalized = value.casefold()
        if normalized in {"/cancelar", "/cancel"}:
            return IntakeTransition(
                replace(self, step=IntakeStep.CANCELED),
                "Entrevista cancelada; nenhum caso foi processado. "
                "Use “Reiniciar entrevista” para começar novamente.",
            )

        if self.step is IntakeStep.CONSENT:
            if normalized == "sim":
                return IntakeTransition(
                    replace(self, step=IntakeStep.AGE_RANGE),
                    "Qual é a faixa etária? Responda criança, adolescente, "
                    "adulto ou idoso; não informe data de nascimento.",
                )
            if normalized in {"não", "nao"}:
                return IntakeTransition(
                    replace(self, step=IntakeStep.CANCELED),
                    "Entrevista cancelada; nenhum caso foi processado.",
                )
            return IntakeTransition(
                self,
                "Digite SIM para confirmar apenas um caso inventado ou NÃO "
                "para cancelar.",
            )

        if self.step is IntakeStep.AGE_RANGE:
            age_ranges = {
                "criança": "criança",
                "crianca": "criança",
                "adolescente": "adolescente",
                "adulto": "adulto",
                "idoso": "idoso",
            }
            age_range = age_ranges.get(normalized)
            if age_range is None:
                return IntakeTransition(
                    self,
                    "Escolha uma faixa etária: criança, adolescente, adulto "
                    "ou idoso.",
                )
            return IntakeTransition(
                replace(
                    self,
                    step=IntakeStep.COMPLAINT,
                    age_range=age_range,
                ),
                "Qual é a queixa principal? (até 240 caracteres)",
            )

        if self.step is IntakeStep.COMPLAINT:
            if not value:
                return IntakeTransition(self, "A queixa principal é obrigatória.")
            if len(value) > 240:
                return IntakeTransition(
                    self,
                    "Limite de 240 caracteres. Resuma a queixa principal.",
                )
            return IntakeTransition(
                replace(
                    self,
                    step=IntakeStep.SYMPTOMS,
                    complaint=value,
                ),
                "Há sintomas associados? Responda PULAR se não houver.",
            )

        optional_steps = {
            IntakeStep.SYMPTOMS: (
                "symptoms",
                400,
                IntakeStep.DURATION,
                "Quando começou ou há quanto tempo? Responda PULAR se não "
                "souber.",
            ),
            IntakeStep.DURATION: (
                "duration",
                160,
                IntakeStep.HISTORY,
                "Há algum histórico relevante? Responda PULAR se não houver.",
            ),
            IntakeStep.HISTORY: (
                "history",
                300,
                IntakeStep.EXAMS,
                "Quais exames já foram realizados? Responda PULAR se nenhum.",
            ),
            IntakeStep.EXAMS: (
                "exams",
                240,
                IntakeStep.FINAL_CONFIRMATION,
                "",
            ),
        }
        optional_step = optional_steps.get(self.step)
        if optional_step is not None:
            field_name, max_length, next_step, next_prompt = optional_step
            optional_value = "" if normalized in {"pular", "/pular"} else value
            if len(optional_value) > max_length:
                return IntakeTransition(
                    self,
                    f"Limite de {max_length} caracteres. Resuma ou responda PULAR.",
                )
            updated = replace(
                self,
                step=next_step,
                **{field_name: optional_value},
            )
            if next_step is IntakeStep.FINAL_CONFIRMATION:
                return IntakeTransition(
                    updated,
                    "Resumo do caso sintético (mantido somente nesta sessão):\n\n"
                    f"{updated.summary()}\n\n"
                    "Digite SIM para consultar as fontes ou NÃO para descartar.",
                )
            return IntakeTransition(updated, next_prompt)

        if self.step is IntakeStep.FINAL_CONFIRMATION:
            if normalized == "sim":
                return IntakeTransition(
                    replace(self, step=IntakeStep.COMPLETE),
                    "Consulta iniciada. O resultado precisa de revisão profissional.",
                    ready_to_review=True,
                )
            if normalized in {"não", "nao"}:
                return IntakeTransition(
                    replace(self, step=IntakeStep.CANCELED),
                    "Caso descartado; nenhuma consulta foi realizada.",
                )
            return IntakeTransition(
                self,
                "Digite SIM para consultar as fontes ou NÃO para descartar o caso.",
            )

        return IntakeTransition(
            self,
            "Esta entrevista foi encerrada. Use “Reiniciar entrevista” "
            "para começar novamente.",
        )

    def summary(self) -> str:
        fields = (
            ("Faixa etária", self.age_range),
            ("Queixa principal", self.complaint),
            ("Sintomas associados", self.symptoms),
            ("Tempo de evolução", self.duration),
            ("Histórico relevante", self.history),
            ("Exames já realizados", self.exams),
        )
        return "\n".join(f"{label}: {value}" for label, value in fields if value)

    def retrieval_query(self) -> str:
        return " ".join(
            value for value in (self.complaint, self.symptoms, self.duration) if value
        )
