import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=(
        r"^'asyncio\.iscoroutinefunction' is deprecated and slated for removal in "
        r"Python 3\.16; use inspect\.iscoroutinefunction\(\) instead$"
    ),
    category=DeprecationWarning,
    module=r"langsmith\._internal\._beta_decorator",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"^Support for Transformers < 5\.6 is deprecated in PEFT and will be "
        r"removed in the next release, v1\.0\. You have Transformers .*\. Please "
        r"upgrade to Transformers >= 5\.6\.$"
    ),
    category=FutureWarning,
    module=r"peft\.import_utils",
)

from medical_assistant.application import build_assistant_runtime
from medical_assistant.chatbot import run_chat_session


def _write_chat_line(line: str) -> None:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        print(line)
        return
    color_by_prefix = {
        "Resposta:": "\033[32m",
        "Fontes:": "\033[36m",
        "Revisão humana:": "\033[33m",
        "Alerta simulado:": "\033[33m",
        "Erro:": "\033[31m",
    }
    color = next(
        (value for prefix, value in color_by_prefix.items() if line.startswith(prefix)),
        None,
    )
    print(f"{color}{line}\033[0m" if color else line)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Chat local acadêmico com QLoRA, protocolos e prontuários pseudonimizados"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/qwen2_5_0_5b_pubmedqa_qlora.json"),
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--approved-protocols",
        type=Path,
        default=Path("data/processed/protocolos_institucionais_autorizados.jsonl"),
        help=(
            "Local JSONL corpus produced by scripts/import_authorized_protocols.py "
            "(default: data/processed/protocolos_institucionais_autorizados.jsonl)"
        ),
    )
    parser.add_argument(
        "--approved-patient-records",
        type=Path,
        default=Path("data/processed/prontuarios_autorizados.jsonl"),
        help=(
            "Locally pseudonymized patient JSONL for the single-operator demo "
            "(default: data/processed/prontuarios_autorizados.jsonl)"
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path("artifacts/chatbot-assistente-audit.jsonl"),
    )
    parser.add_argument(
        "--alerts-path",
        type=Path,
        default=Path("artifacts/chatbot-assistente-alerts.jsonl"),
    )
    args = parser.parse_args()
    runtime = build_assistant_runtime(
        project_root,
        config_path=args.config,
        adapter_dir=args.adapter_dir,
        approved_protocols_path=args.approved_protocols,
        approved_patient_records_path=args.approved_patient_records,
        audit_path=args.audit_path,
        alerts_path=args.alerts_path,
    )
    print(f"Modelo-base e adaptador carregados: {runtime.model_version}")
    print(
        "Índice local: fixtures sintéticos e "
        f"{runtime.pubmedqa_abstract_count} abstracts PQA-L."
    )
    print(f"Pares MedQuAD carregados no RAG: {runtime.medquad_record_count}.")
    print(
        "Protocolos institucionais aprovados carregados: "
        f"{runtime.protocol_count}."
    )
    print(
        "Prontuários pseudonimizados disponíveis em modo local de operador único: "
        f"{runtime.patient_record_count}."
    )
    print(f"Busca híbrida local: {runtime.retrieval_version}.")
    print("O índice inclui abstracts de todos os splits; não é avaliação independente.")
    print("Auditoria minimizada:", runtime.audit_path)
    print("Saída de alertas apenas simulados:", runtime.alerts_path)
    run_chat_session(
        runtime.workflow,
        runtime.actor,
        runtime.staff_member,
        runtime.patient_store,
        read_line=lambda: input("\nVocê> "),
        write_line=_write_chat_line,
    )


if __name__ == "__main__":
    main()
