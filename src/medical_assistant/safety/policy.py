import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class SafetyReason(StrEnum):
    CLINICAL_ACTION = "clinical_action"
    URGENT_SIGNAL = "urgent_signal"


@dataclass(frozen=True)
class SafetyAssessment:
    requires_human_review: bool
    reason: SafetyReason | None = None


_ACTION_VERBS = re.compile(
    r"\b(?:prescrev\w*|prescrib\w*|receit\w*|diagnostic\w*|diagnos\w*|"
    r"recomend\w*|recommend\w*|sugir\w*|suggest\w*|indiqu\w*|administ\w*|"
    r"administer\w*|inicie\w*|iniciar\w*|start\w*|interromp\w*|suspend\w*|"
    r"stop\w*|ajust\w*|adjust\w*|altere\w*|alter\w*|defin\w*|aplique\w*|apply\w*|"
    r"execute\w*|execut\w*)\b"
)
_DECISION_CUES = re.compile(
    r"\b(?:devo|deveria|posso|deve|should|must|can i|what should|which|qual)\b"
)
_CLINICAL_ACTION_NOUNS = re.compile(
    r"\b(?:dose|dosagem|dosage|medicamento|medicacao|medication|tratamento|"
    r"treatment|procedimento|procedure|conduta|prescricao|diagnostico)\b"
)
_URGENT_SIGNALS = re.compile(
    r"\b(?:urgente|urgencia|emergencia|emergency|urgent|dor no peito|chest pain|"
    r"falta de ar|shortness of breath|sangramento intenso|heavy bleeding|"
    r"convulsao|seizure|desmaio|fainting)\b"
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


class SafetyPolicy:
    """Conservative keyword policy; it is not a clinical triage system."""

    def assess(self, question: str) -> SafetyAssessment:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        normalized = _normalize(question)
        if _URGENT_SIGNALS.search(normalized):
            return SafetyAssessment(
                requires_human_review=True,
                reason=SafetyReason.URGENT_SIGNAL,
            )
        if _ACTION_VERBS.search(normalized) or (
            _DECISION_CUES.search(normalized) and _CLINICAL_ACTION_NOUNS.search(normalized)
        ):
            return SafetyAssessment(
                requires_human_review=True,
                reason=SafetyReason.CLINICAL_ACTION,
            )
        return SafetyAssessment(requires_human_review=False)
