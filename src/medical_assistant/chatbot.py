from dataclasses import dataclass
from collections.abc import Callable

from medical_assistant.audit import AuditActor
from medical_assistant.patient_data import (
    PatientStore,
    SyntheticStaffMember,
)
from medical_assistant.workflows import (
    AssistantWorkflow,
    WorkflowOperation,
    WorkflowResult,
)


@dataclass(frozen=True)
class SyntheticCaseIntake:
    summary: str
    retrieval_query: str


def _show_result(write_line: Callable[[str], None], result: WorkflowResult) -> None:
    write_line(f"Status: {result.status.value}")
    if result.error_code is not None:
        write_line(f"Erro: {result.error_code}")
    if result.response is not None:
        write_line(f"Resposta: {result.response.answer}")
        write_line(
            "Fontes: "
            + ", ".join(source.source_id for source in result.response.sources)
        )
        write_line(f"Revisão humana: {result.response.review_status.value}")
        write_line(f"Limitação: {result.response.uncertainty}")
    if result.simulated_alert is not None:
        write_line(f"Alerta simulado: {result.simulated_alert.status}")


def _read_case_field(
    label: str,
    read_line: Callable[[], str],
    write_line: Callable[[str], None],
    *,
    required: bool,
    max_length: int,
) -> str | None:
    while True:
        write_line(label)
        value = read_line().strip()
        if value.casefold() in {"/cancelar", "/cancel"}:
            return None
        if len(value) > max_length:
            write_line(f"Limite de {max_length} caracteres. Tente novamente.")
            continue
        if not value:
            if required:
                write_line("Este campo é obrigatório; digite /cancelar para sair.")
                continue
            return "Não informado"
        return value


def _collect_synthetic_case_summary(
    read_line: Callable[[], str],
    write_line: Callable[[str], None],
) -> SyntheticCaseIntake | None:
    write_line(
        "Entrevista acadêmica: use somente informações inventadas. "
        "Não informe nomes, documentos ou dados de pacientes reais."
    )
    write_line("Digite SIM para confirmar o uso exclusivo de um caso sintético:")
    if read_line().strip().casefold() != "sim":
        write_line("Entrevista cancelada; nenhum caso foi processado.")
        return None

    fields = (
        (
            "Faixa etária (criança/adolescente/adulto/idoso; sem data de nascimento):",
            "Faixa etária:",
            True,
            80,
        ),
        ("Queixa principal:", "Queixa principal:", True, 240),
        ("Sintomas associados (opcional):", "Sintomas associados:", False, 400),
        (
            "Quando começou ou há quanto tempo (opcional):",
            "Tempo de evolução:",
            False,
            160,
        ),
        ("Histórico relevante (opcional):", "Histórico relevante:", False, 300),
        ("Exames já realizados (opcional):", "Exames já realizados:", False, 240),
    )
    collected: list[tuple[str, str]] = []
    for prompt, label, required, max_length in fields:
        value = _read_case_field(
            prompt,
            read_line,
            write_line,
            required=required,
            max_length=max_length,
        )
        if value is None:
            write_line("Entrevista cancelada; nenhum caso foi processado.")
            return None
        if value != "Não informado":
            collected.append((label, value))

    summary = "\n".join(f"{label} {value}" for label, value in collected)
    retrieval_query = " ".join(
        value
        for label, value in collected
        if label in {
            "Queixa principal:",
            "Sintomas associados:",
            "Tempo de evolução:",
        }
    )
    write_line("Resumo do caso sintético (não é gravado na auditoria):")
    for label, value in collected:
        write_line(f"{label} {value}")
    write_line(
        "Gerar revisão acadêmica de possibilidades e exames citados nas fontes? "
        "Digite SIM para continuar:"
    )
    if read_line().strip().casefold() != "sim":
        write_line("Revisão cancelada; nenhum caso foi processado.")
        return None
    return SyntheticCaseIntake(summary, retrieval_query)


def run_chat_session(
    workflow: AssistantWorkflow,
    actor: AuditActor,
    staff_member: SyntheticStaffMember,
    patient_store: PatientStore,
    *,
    read_line: Callable[[], str],
    write_line: Callable[[str], None],
) -> None:
    authorized_patient_ids = staff_member.authorized_patient_ids
    available_patient_ids = set(patient_store.available_patient_ids)
    selected_patient_id: str | None = None
    access_context = staff_member.access_context()
    write_line(
        "Modo local de operador único, sem autenticação real. Prontuários "
        "pseudonimizados só entram no contexto após seleção explícita; "
        "não exponha este protótipo na rede."
    )
    while True:
        try:
            question = read_line().strip()
        except EOFError:
            break
        if not question:
            continue
        if question.casefold() in {"/sair", "/exit", "sair"}:
            write_line("Sessão encerrada.")
            break
        if question.casefold() in {"/ajuda", "/help"}:
            write_line(
                "Comandos: /atendimento inicia uma entrevista acadêmica sintética; "
                "/prontuarios lista aliases locais; /paciente <ID> seleciona um "
                "registro autorizado; /paciente limpar remove o contexto; "
                "/exames consulta exames pendentes; /sair encerra."
            )
            write_line(
                "Para perguntas informativas, digite sua pergunta em português. "
                "Não trate respostas como decisão clínica."
            )
            continue
        if question.casefold() == "/atendimento":
            try:
                case_intake = _collect_synthetic_case_summary(read_line, write_line)
            except EOFError:
                write_line("Entrevista interrompida; nenhum caso foi processado.")
                return
            if case_intake is None:
                continue
            write_line(
                "Consultando fontes acadêmicas locais. O resultado será apenas "
                "possibilidade para revisão profissional, nunca diagnóstico ou ordem."
            )
            result = workflow.run(
                case_intake.summary,
                actor=actor,
                operation=WorkflowOperation.REVIEW_SYNTHETIC_CASE,
                retrieval_query=case_intake.retrieval_query,
            )
            _show_result(write_line, result)
            continue
        if question.casefold() == "/paciente limpar":
            selected_patient_id = None
            write_line("Contexto do paciente removido.")
            continue
        if question.casefold() == "/prontuarios":
            case_ids = sorted(
                patient_id
                for patient_id in available_patient_ids
                if patient_id.startswith("CASE-")
            )
            if case_ids:
                write_line("Aliases locais disponíveis: " + ", ".join(case_ids))
            else:
                write_line("Nenhum prontuário local carregado.")
            continue
        if question.casefold().startswith("/paciente "):
            requested_id = question.partition(" ")[2].strip()
            if (
                requested_id not in available_patient_ids
                or requested_id not in authorized_patient_ids
            ):
                write_line("Paciente inexistente ou não autorizado.")
                continue
            selected_patient_id = requested_id
            write_line("Contexto local selecionado.")
            continue
        if question.casefold() == "/exames":
            if selected_patient_id is None:
                write_line("Selecione primeiro um registro autorizado.")
                continue
            result = workflow.run(
                "Verifique exames pendentes",
                actor=actor,
                patient_id=selected_patient_id,
                access_context=access_context,
                operation=WorkflowOperation.CHECK_PENDING_EXAMS,
            )
            _show_result(write_line, result)
            continue
        if question.startswith("/"):
            write_line(
                "Comando desconhecido. Use /atendimento, /prontuarios, "
                "/paciente ID, /exames ou /sair."
            )
            continue

        result = workflow.run(
            question,
            actor=actor,
            patient_id=selected_patient_id,
            access_context=access_context if selected_patient_id is not None else None,
        )
        _show_result(write_line, result)
