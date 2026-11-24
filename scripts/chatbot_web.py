from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from medical_assistant.application import AssistantRuntime, build_assistant_runtime
from medical_assistant.ui.intake import SyntheticCaseInterview
from medical_assistant.ui.presentation import UiResult, to_ui_result
from medical_assistant.workflows import WorkflowOperation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def _load_runtime() -> AssistantRuntime:
    return build_assistant_runtime(PROJECT_ROOT)


def _payload(result: UiResult) -> dict[str, Any]:
    return asdict(result)


def _render_result(result: dict[str, Any]) -> None:
    status = result["status"]
    if status == "complete":
        st.success("Consulta concluída")
    elif status == "insufficient_evidence":
        st.warning("Evidência insuficiente")
    elif status == "review_required":
        st.warning("Encaminhado para revisão humana")
    else:
        st.error("Não foi possível concluir a operação")

    if result["error_code"]:
        st.error(f"Erro: {result['error_code']}")
    if result["answer"]:
        st.markdown(result["answer"])
    if result["review_status"] == "pending":
        st.warning("Revisão profissional necessária; não é diagnóstico nem ordem clínica.")
    if result["uncertainty"]:
        st.caption(f"Limites: {result['uncertainty']}")
    if result["alert_status"]:
        st.info(f"Alerta local simulado: {result['alert_status']}")

    sources = result["sources"]
    with st.expander(f"Fontes e evidências ({len(sources)})"):
        if not sources:
            st.caption("Nenhuma fonte foi recuperada.")
        for source in sources:
            label = (
                "Prontuário selecionado"
                if source["patient_record"]
                else f"{source['title']} · {source['source_id']}"
            )
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(source["section"])
                if source["patient_record"]:
                    st.info(
                        "A fonte foi usada no contexto selecionado. Conteúdo e "
                        "metadados clínicos foram ocultados neste painel."
                    )
                elif source["excerpt"]:
                    st.text(source["excerpt"])
                score_columns = st.columns(2)
                if source["lexical_score"] is not None:
                    score_columns[0].metric(
                        "Relevância lexical",
                        f"{source['lexical_score']:.2f}",
                    )
                if source["semantic_score"] is not None:
                    score_columns[1].metric(
                        "Similaridade semântica",
                        f"{source['semantic_score']:.2f}",
                    )
                st.caption(f"Referência: {source['source']}")


def _clear_patient_context() -> None:
    st.session_state["patient_selector"] = ""
    st.session_state.pop("exam_result", None)


def _chat_tab(runtime: AssistantRuntime, patient_id: str | None) -> None:
    history = st.session_state["chat_history"]
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_result(message["result"])

    prompt = st.chat_input("Escreva sua pergunta informativa")
    if prompt is None:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando as fontes locais..."):
            result = runtime.workflow.run(
                prompt,
                actor=runtime.actor,
                patient_id=patient_id,
                access_context=(
                    runtime.staff_member.access_context()
                    if patient_id is not None
                    else None
                ),
            )
        rendered = _payload(to_ui_result(result))
        _render_result(rendered)
    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "result": rendered},
        ]
    )


def _synthetic_intake_tab(runtime: AssistantRuntime) -> None:
    st.info(
        "Entrevista somente para caso inventado. Ela não usa o prontuário "
        "selecionado na barra lateral. Responda PULAR para campos opcionais."
    )
    state = st.session_state.get("synthetic_interview_state")
    if not isinstance(state, SyntheticCaseInterview):
        state = SyntheticCaseInterview()
        st.session_state["synthetic_interview_state"] = state
    history = st.session_state.setdefault(
        "synthetic_interview_history",
        [{"role": "assistant", "content": state.opening_message()}],
    )

    for message in history:
        with st.chat_message(message["role"]):
            if "result" in message:
                _render_result(message["result"])
            else:
                st.markdown(message["content"])

    if st.button("Reiniciar entrevista", key="reset_synthetic_interview"):
        state = SyntheticCaseInterview()
        st.session_state["synthetic_interview_state"] = state
        st.session_state["synthetic_interview_history"] = [
            {"role": "assistant", "content": state.opening_message()}
        ]
        st.rerun()

    answer = st.chat_input(
        "Digite sua resposta",
        key="synthetic_interview_input",
    )
    if answer is None:
        return

    history.append({"role": "user", "content": answer})
    transition = state.advance(answer)
    st.session_state["synthetic_interview_state"] = transition.state
    history.append(
        {"role": "assistant", "content": transition.message}
    )

    with st.chat_message("user"):
        st.markdown(answer)
    with st.chat_message("assistant"):
        st.markdown(transition.message)
        if transition.ready_to_review:
            with st.spinner("Consultando as fontes locais..."):
                result = runtime.workflow.run(
                    transition.state.summary(),
                    actor=runtime.actor,
                    operation=WorkflowOperation.REVIEW_SYNTHETIC_CASE,
                    retrieval_query=transition.state.retrieval_query(),
                )
            rendered = _payload(to_ui_result(result))
            _render_result(rendered)
            history.append({"role": "assistant", "result": rendered})


def _exam_tab(runtime: AssistantRuntime, patient_id: str | None) -> None:
    if patient_id is None:
        st.info("Selecione um registro na barra lateral para consultar exames.")
        return

    st.caption(
        "A consulta usa somente o registro explicitamente selecionado. "
        "O resultado é local e não envia alertas à equipe."
    )
    if st.button("Verificar exames", type="primary"):
        with st.spinner("Verificando o registro local..."):
            result = runtime.workflow.run(
                "Verifique exames pendentes",
                actor=runtime.actor,
                patient_id=patient_id,
                access_context=runtime.staff_member.access_context(),
                operation=WorkflowOperation.CHECK_PENDING_EXAMS,
            )
        st.session_state["exam_result"] = {
            "patient_id": patient_id,
            "result": _payload(to_ui_result(result)),
        }

    stored_result = st.session_state.get("exam_result")
    if (
        stored_result is not None
        and stored_result["patient_id"] == patient_id
    ):
        _render_result(stored_result["result"])


def main() -> None:
    st.set_page_config(
        page_title="Assistente médico acadêmico",
        layout="wide",
    )
    st.title("Assistente médico")
    st.caption("Protótipo acadêmico local com fontes rastreáveis.")
    st.warning(
        "Não use para diagnóstico, prescrição ou decisões assistenciais. "
        "A revisão profissional é obrigatória."
    )

    for key, initial_value in (
        ("chat_history", []),
        ("patient_selector", ""),
    ):
        if key not in st.session_state:
            st.session_state[key] = initial_value

    runtime_loaded = st.session_state.get("runtime_loaded", False)
    load_requested = False
    with st.sidebar:
        st.header("Sessão local")
        st.info(
            "Operador único, sem autenticação real. Não exponha esta interface "
            "na rede."
        )
        if not runtime_loaded:
            load_requested = st.button(
                "Carregar assistente local",
                type="primary",
            )

    if not runtime_loaded and not load_requested:
        st.subheader("Bem-vindo")
        st.write(
            "Carregue o modelo local para habilitar perguntas, atendimento "
            "sintético e consulta de exames."
        )
        st.caption(
            "A primeira carga pode demorar enquanto inicializa o modelo e os "
            "embeddings locais."
        )
        return

    with st.spinner(
        "Carregando modelo, embeddings e bases locais; a primeira execução pode demorar..."
    ):
        runtime = _load_runtime()
    st.session_state["runtime_loaded"] = True

    with st.sidebar:
        patient_options = [""] + sorted(runtime.staff_member.authorized_patient_ids)
        selected_patient = st.selectbox(
            "Contexto do paciente",
            patient_options,
            format_func=lambda value: value or "Sem prontuário",
            key="patient_selector",
        )
        patient_id = selected_patient or None
        st.button(
            "Limpar contexto do paciente",
            on_click=_clear_patient_context,
        )
        if patient_id is not None:
            st.caption(f"Somente o alias selecionado está no contexto: {patient_id}")
        st.metric("Protocolos no RAG", runtime.protocol_count)
        st.metric("Prontuários disponíveis", runtime.patient_record_count)
        st.metric("Pares MedQuAD no RAG", runtime.medquad_record_count)
        st.caption(f"LLM: {runtime.model_version}")
        st.caption(f"Busca: {runtime.retrieval_version}")
        if st.button("Limpar conversa"):
            st.session_state["chat_history"] = []
            st.rerun()

    chat_tab, intake_tab, exams_tab = st.tabs(
        ["Perguntas", "Atendimento sintético", "Exames"]
    )
    with chat_tab:
        _chat_tab(runtime, patient_id)
    with intake_tab:
        _synthetic_intake_tab(runtime)
    with exams_tab:
        _exam_tab(runtime, patient_id)


if __name__ == "__main__":
    main()
