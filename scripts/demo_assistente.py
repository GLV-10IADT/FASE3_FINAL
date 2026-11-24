from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medical_assistant.audit import AuditActor, JsonlAuditLogger
from medical_assistant.patient_data import (
    AccessContext,
    PatientRole,
    SyntheticPatientStore,
)
from medical_assistant.retrieval import DocumentIndex, load_synthetic_documents
from medical_assistant.workflows import AssistantWorkflow, WorkflowResult


def _show_result(label: str, result: WorkflowResult) -> None:
    response = result.response
    print(f"\n[{label}]")
    print("Status:", result.status.value)
    print("Intent:", result.intent.value)
    if response is None:
        print("Resposta: indisponivel")
        return
    print("Resposta:", response.answer)
    print("Fontes:", [source.source_id for source in response.sources])
    print("Revisao humana:", response.review_status.value)


def run_demo(project_root: Path, audit_path: Path) -> None:
    synthetic_dir = project_root / "data" / "synthetic"
    audit_sink = JsonlAuditLogger(audit_path)
    actor = AuditActor(user_id="demo-clinician", role="clinician")
    workflow = AssistantWorkflow(
        DocumentIndex(load_synthetic_documents(synthetic_dir)),
        SyntheticPatientStore(synthetic_dir / "patients.json"),
        audit_sink,
    )

    print("Demonstracao deterministica; sem LLM e sem dados reais.")
    print("Auditoria:", audit_path)
    _show_result(
        "Consulta informativa com fonte sintetica",
        workflow.run(
            "Como identificar origem metadados de um trecho?",
            actor=actor,
        ),
    )

    patient_id = "PATIENT-SYNTH-001"
    _show_result(
        "Leitura autorizada de fixture de paciente sintetico",
        workflow.run(
            "consultar registros",
            actor=actor,
            patient_id=patient_id,
            access_context=AccessContext(
                user_id=actor.user_id,
                role=PatientRole.CLINICIAN,
                authorized_patient_ids=frozenset({patient_id}),
            ),
        ),
    )

    _show_result(
        "Pedido de conduta encaminhado para revisao",
        workflow.run(
            "Qual tratamento devo prescrever?",
            actor=actor,
        ),
    )
    print("\nEventos de auditoria gravados: 3")
    print("Nenhum alerta foi enviado; nenhuma ordem ou prescricao foi executada.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Demonstrate the synthetic medical assistant workflow"
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path("artifacts/demo-assistente-audit.jsonl"),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audit_path = args.audit_path
    if not audit_path.is_absolute():
        audit_path = project_root / audit_path
    run_demo(project_root, audit_path)


if __name__ == "__main__":
    main()
