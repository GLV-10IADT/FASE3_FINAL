import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from medical_assistant.datasets.internal import (
    INTERNAL_SOURCE_DATASETS,
    validate_internal_training_record,
)
from medical_assistant.datasets.loaders import load_medquad, load_pubmedqa
from medical_assistant.training.evaluation import (
    classify_response_behavior,
    evaluate_system,
)
from medical_assistant.retrieval import (
    SourceDocument,
    load_medquad_documents,
    load_pubmedqa_documents,
    load_synthetic_documents,
)


_ALLOWED_DATASETS = {"synthetic", "pubmedqa", "medquad"} | INTERNAL_SOURCE_DATASETS
_ALLOWED_SPLITS = ("train", "validation", "test")


class _MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def write(self, event: Any) -> None:
        self.events.append(event)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"Dataset split is empty: {path}")
    return records


def _record_key(record: dict[str, Any]) -> set[tuple[str, ...]]:
    dataset = record["source_dataset"]
    source_id = record["source_id"]
    source_group_id = record.get("source_group_id") or source_id
    content = " ".join(
        " ".join(record[field].casefold().split()) for field in ("question", "answer")
    )
    return {
        ("source", dataset, source_id),
        ("source_group", dataset, source_group_id),
        ("content", content),
    }


def validate_splits(splits: dict[str, list[dict[str, Any]]], *, synthetic_only: bool) -> None:
    if set(splits) != set(_ALLOWED_SPLITS):
        raise ValueError(f"Expected exactly these splits: {', '.join(_ALLOWED_SPLITS)}")

    assigned_keys: dict[tuple[str, ...], str] = {}
    for split_name in _ALLOWED_SPLITS:
        records = splits[split_name]
        if not records:
            raise ValueError(f"Dataset split {split_name!r} must not be empty")
        seen_source_ids: set[tuple[str, str]] = set()
        for record in records:
            for field in ("question", "answer", "source_dataset", "source_id", "split"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"Record in {split_name!r} has invalid {field!r}")
            if record["split"] != split_name:
                raise ValueError(
                    f"Record {record['source_id']!r} is in {split_name!r} but declares "
                    f"{record['split']!r}"
                )
            if record["source_dataset"] not in _ALLOWED_DATASETS:
                raise ValueError(
                    f"Unsupported or internal source dataset {record['source_dataset']!r}"
                )
            source_group_id = record.get("source_group_id")
            if source_group_id is not None and (
                not isinstance(source_group_id, str) or not source_group_id.strip()
            ):
                raise ValueError(f"Record {record['source_id']!r} has invalid source_group_id")
            source_key = (record["source_dataset"], record["source_id"])
            if source_key in seen_source_ids:
                raise ValueError(f"Duplicate source ID in {split_name!r}: {source_key!r}")
            seen_source_ids.add(source_key)
            if synthetic_only and record.get("synthetic") is not True:
                raise ValueError("Smoke-test data must be explicitly marked synthetic")
            if record["source_dataset"] == "synthetic" and record.get("synthetic") is not True:
                raise ValueError("Synthetic records must carry synthetic=true")
            if record["source_dataset"] in INTERNAL_SOURCE_DATASETS:
                validate_internal_training_record(record)
            if record["source_dataset"] == "pubmedqa":
                if record.get("source_subset") == "PQA-U":
                    raise ValueError("Unlabeled PubMedQA PQA-U records are not supported")
                if record.get("source_subset") == "PQA-A" and split_name != "train":
                    raise ValueError("Artificial PubMedQA PQA-A records are train-only")

            for key in _record_key(record):
                previous_split = assigned_keys.get(key)
                if previous_split is not None and previous_split != split_name:
                    raise ValueError(
                        f"Source/content leakage between {previous_split!r} and "
                        f"{split_name!r} at record {record['source_id']!r}"
                    )
                assigned_keys[key] = split_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SYSTEM_MESSAGE = (
    "Você é um assistente informativo de demonstração. Use somente o contexto "
    "fornecido quando houver; não diagnostique, prescreva ou invente fatos e "
    "fontes. Declare quando não houver evidência suficiente."
)


def _format_messages(
    question: str,
    *,
    answer: str | None = None,
    context: str | None = None,
) -> list[dict[str, str]]:
    user_content = question
    if context is not None:
        if not isinstance(context, str):
            raise ValueError("Record context must be a string when provided")
        user_content = f"Pergunta: {question}\n\nContexto científico:\n{context}"
    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _render_messages(
    tokenizer: Any,
    question: str,
    answer: str | None = None,
    *,
    context: str | None = None,
) -> list[int]:
    return tokenizer.apply_chat_template(
        _format_messages(question, answer=answer, context=context),
        tokenize=True,
        add_generation_prompt=answer is None,
    )


def _fit_context(
    tokenizer: Any,
    *,
    question: str,
    answer: str | None,
    context: str | None,
    max_length: int,
    source_id: str,
) -> tuple[list[int], list[int], bool]:
    prompt_ids = _render_messages(tokenizer, question, context=context)
    full_ids = _render_messages(
        tokenizer,
        question,
        answer,
        context=context,
    )
    if len(full_ids) <= max_length or context is None:
        return prompt_ids, full_ids, False

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    lower_bound = 0
    upper_bound = len(context_ids)
    best_fit: tuple[list[int], list[int]] | None = None
    while lower_bound <= upper_bound:
        midpoint = (lower_bound + upper_bound) // 2
        shortened_context = tokenizer.decode(
            context_ids[:midpoint],
            skip_special_tokens=True,
        )
        shortened_prompt = _render_messages(
            tokenizer,
            question,
            context=shortened_context,
        )
        shortened_full = _render_messages(
            tokenizer,
            question,
            answer,
            context=shortened_context,
        )
        if len(shortened_full) <= max_length:
            best_fit = shortened_prompt, shortened_full
            lower_bound = midpoint + 1
        else:
            upper_bound = midpoint - 1

    if best_fit is None:
        raise ValueError(
            f"Question and answer exceed max sequence length for record {source_id!r}"
        )
    return best_fit[0], best_fit[1], True


def _tokenize_records(
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[dict[str, list[int]]], int]:
    tokenized: list[dict[str, list[int]]] = []
    truncated_contexts = 0
    for record in records:
        prompt_ids, full_ids, context_truncated = _fit_context(
            tokenizer,
            question=record["question"],
            answer=record["answer"],
            context=record.get("context"),
            max_length=max_length,
            source_id=record["source_id"],
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenizer chat template prompt prefix does not match full example")
        full_ids = full_ids[:max_length]
        truncated_contexts += int(context_truncated)
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        labels[:prompt_length] = [-100] * prompt_length
        if not any(label != -100 for label in labels):
            raise ValueError(f"Answer was truncated from record {record['source_id']!r}")
        tokenized.append(
            {
                "input_ids": list(full_ids),
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return tokenized, truncated_contexts


def _evaluation_limitations(
    *,
    test_count: int,
    smoke_test: bool,
    uses_pubmedqa: bool,
    uses_medquad: bool = False,
) -> list[str]:
    if smoke_test:
        limitations = [
            f"Evaluation uses {test_count} synthetic test cases; it is not clinical validation."
        ]
    else:
        limitations = [
            f"Evaluation uses {test_count} held-out records; it is not clinical validation."
        ]
    limitations.extend(
        [
            "Token F1 is lexical overlap, not semantic or clinical correctness.",
            "Behavior classification is a small rule-based proxy.",
        ]
    )
    if smoke_test:
        limitations.append(
            "The adapter was trained for one step on synthetic smoke-test data."
        )
    else:
        limitations.append(
            "The fine-tuning experiment is exploratory and not validated for clinical use."
        )
    if uses_pubmedqa:
        limitations.append(
            "The evaluation RAG index contains only PubMedQA training-split abstracts."
        )
    if uses_medquad:
        limitations.append(
            "The evaluation RAG index contains only MedQuAD training-split documents."
        )
    return limitations


def _load_training_retrieval_documents(
    project_root: Path,
    data_config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    *,
    uses_pubmedqa: bool,
    uses_medquad: bool,
) -> list[SourceDocument]:
    retrieval_documents = list(
        load_synthetic_documents(project_root / "data" / "synthetic")
    )
    if uses_pubmedqa:
        retrieval_source = data_config.get("retrieval_source")
        if not isinstance(retrieval_source, str) or not retrieval_source.strip():
            raise ValueError("PubMedQA experiments require data.retrieval_source")
        retrieval_path = Path(retrieval_source)
        if not retrieval_path.is_absolute():
            retrieval_path = project_root / retrieval_path
        pubmedqa_records = load_pubmedqa(retrieval_path, subset="PQA-L")
        training_pubmedqa_ids = {
            record["source_id"]
            for record in splits["train"]
            if record["source_dataset"] == "pubmedqa"
        }
        training_pubmedqa_records = tuple(
            record
            for record in pubmedqa_records.records
            if record.source_id in training_pubmedqa_ids
        )
        if len(training_pubmedqa_records) != len(training_pubmedqa_ids):
            raise ValueError("PubMedQA training split contains unknown source IDs")
        retrieval_documents.extend(
            load_pubmedqa_documents(training_pubmedqa_records)
        )
    if uses_medquad:
        medquad_source = data_config.get("medquad_source")
        if not isinstance(medquad_source, str) or not medquad_source.strip():
            raise ValueError("MedQuAD experiments require data.medquad_source")
        medquad_path = Path(medquad_source)
        if not medquad_path.is_absolute():
            medquad_path = project_root / medquad_path
        medquad_records = load_medquad(medquad_path).records
        training_medquad_ids = {
            record["id"]
            for record in splits["train"]
            if record["source_dataset"] == "medquad"
        }
        training_medquad_records = tuple(
            record for record in medquad_records if record.id in training_medquad_ids
        )
        if len(training_medquad_records) != len(training_medquad_ids):
            raise ValueError("MedQuAD training split contains unknown source IDs")
        retrieval_documents.extend(load_medquad_documents(training_medquad_records))
    return retrieval_documents


def run_experiment(
    config_path: Path,
    *,
    smoke_test: bool = False,
    resume_from_checkpoint: Path | None = None,
) -> Path:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing; install the project with "
            "`python -m pip install -e \".[training,test]\"`."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an available CUDA GPU for this experiment")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a JSON object")

    project_root = config_path.resolve().parents[2]
    data_config = config["data"]
    if smoke_test:
        data_paths = {
            split: project_root / "training" / "synthetic" / f"{split}.jsonl"
            for split in _ALLOWED_SPLITS
        }
    else:
        data_paths = {
            split: project_root / data_config[split] for split in _ALLOWED_SPLITS
        }
    splits = {split: load_jsonl(path) for split, path in data_paths.items()}
    validate_splits(splits, synthetic_only=smoke_test)
    uses_pubmedqa = any(
        record["source_dataset"] == "pubmedqa"
        for split_records in splits.values()
        for record in split_records
    )
    uses_medquad = any(
        record["source_dataset"] == "medquad"
        for split_records in splits.values()
        for record in split_records
    )
    if smoke_test:
        for record in splits["test"]:
            if record.get("expected_behavior") not in {"answer", "abstain", "review"}:
                raise ValueError(
                    f"Synthetic evaluation record {record['source_id']!r} "
                    "must declare an expected_behavior"
                )

    model_config = config["model"]
    training_config = config["training"].copy()
    lora_config = config["method"]["lora"]
    quantization_config = config["method"]["quantization"]
    if smoke_test:
        training_config.update(
            {
                "max_sequence_length": 256,
                "max_train_samples": min(4, len(splits["train"])),
                "max_steps": 1,
                "evaluation_steps": 1,
                "save_steps": 1,
                "logging_steps": 1,
            }
        )

    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke_test and resume_from_checkpoint is not None:
        raise ValueError("A smoke test cannot resume a full experiment checkpoint")
    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.is_absolute():
            resume_from_checkpoint = project_root / resume_from_checkpoint
        if not resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Training checkpoint does not exist: {resume_from_checkpoint}"
            )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repository"],
        revision=model_config["revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_quant_type=quantization_config["quant_type"],
        bnb_4bit_use_double_quant=quantization_config["double_quantization"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["repository"],
        revision=model_config["revision"],
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
    )
    model.config.use_cache = False

    from medical_assistant.audit import AuditActor
    from medical_assistant.patient_data import SyntheticPatientStore
    from medical_assistant.retrieval import (
        DocumentIndex,
    )
    from medical_assistant.training.langchain_model import TransformersChatModel
    from medical_assistant.workflows import AssistantWorkflow, WorkflowStatus

    synthetic_dir = project_root / "data" / "synthetic"
    retrieval_documents = _load_training_retrieval_documents(
        project_root,
        data_config,
        splits,
        uses_pubmedqa=uses_pubmedqa,
        uses_medquad=uses_medquad,
    )
    document_index = DocumentIndex(retrieval_documents)
    audit_sink = _MemoryAuditSink()
    workflow = AssistantWorkflow(
        document_index,
        SyntheticPatientStore(synthetic_dir / "patients.json"),
        audit_sink,
    )
    evaluation_actor = AuditActor(user_id="offline-evaluation", role="system")

    max_sequence_length = training_config["max_sequence_length"]
    baseline_predictions = _generate_predictions(
        model,
        tokenizer,
        splits["test"],
        max_new_tokens=64,
        max_input_length=max_sequence_length,
    )
    for prediction in baseline_predictions:
        prediction.update(
            {
                "observed_behavior": classify_response_behavior(
                    prediction["prediction"]
                ),
                "retrieved_source_ids": [],
                "citation_source_ids": [],
            }
        )
    safe_baseline_predictions = _apply_model_safety_gates(
        workflow=workflow,
        workflow_status=WorkflowStatus,
        actor=evaluation_actor,
        records=splits["test"],
        raw_predictions=baseline_predictions,
    )
    rag_predictions = _run_rag_predictions(
        workflow=workflow,
        workflow_status=WorkflowStatus,
        document_index=document_index,
        actor=evaluation_actor,
        records=splits["test"],
    )
    evaluation_records = [
        {
            **record,
            "expected_source_id": record["source_id"],
        }
        if record["source_dataset"] in {"pubmedqa", "medquad"}
        else record
        for record in splits["test"]
    ]
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_config["gradient_checkpointing"],
    )
    adapter_config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter_config)

    class TokenizedDataset(Dataset):
        def __init__(self, examples: list[dict[str, list[int]]]) -> None:
            self._examples = examples

        def __len__(self) -> int:
            return len(self._examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in self._examples[index].items()
            }

    class CausalLMCollator:
        def __call__(
            self,
            features: list[dict[str, torch.Tensor]],
        ) -> dict[str, torch.Tensor]:
            max_length = max(len(feature["input_ids"]) for feature in features)
            input_ids = torch.full(
                (len(features), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for row, feature in enumerate(features):
                length = len(feature["input_ids"])
                input_ids[row, :length] = feature["input_ids"]
                attention_mask[row, :length] = feature["attention_mask"]
                labels[row, :length] = feature["labels"]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    max_length = training_config["max_sequence_length"]
    training_records = splits["train"][: training_config["max_train_samples"]]
    tokenized_train, train_contexts_truncated = _tokenize_records(
        tokenizer,
        training_records,
        max_length,
    )
    tokenized_validation, validation_contexts_truncated = _tokenize_records(
        tokenizer,
        splits["validation"],
        max_length,
    )
    train_dataset = TokenizedDataset(tokenized_train)
    validation_dataset = TokenizedDataset(tokenized_validation)

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        max_steps=training_config["max_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        optim=training_config["optimizer"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        eval_strategy=training_config["evaluation_strategy"],
        eval_steps=training_config["evaluation_steps"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        logging_steps=training_config["logging_steps"],
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        metric_for_best_model=training_config.get(
            "metric_for_best_model",
            "eval_loss",
        ),
        greater_is_better=training_config.get("greater_is_better", False),
        seed=training_config["seed"],
        data_seed=training_config["seed"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(),
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(output_dir / "adapter")

    tuned_predictions = _generate_predictions(
        trainer.model,
        tokenizer,
        splits["test"],
        max_new_tokens=64,
        max_input_length=max_sequence_length,
    )
    finetuned_workflow = AssistantWorkflow(
        document_index,
        SyntheticPatientStore(synthetic_dir / "patients.json"),
        audit_sink,
        chat_model=TransformersChatModel(
            model=trainer.model,
            tokenizer=tokenizer,
            max_new_tokens=64,
        ),
        model_version=(
            f"{model_config['repository']}@{model_config['revision']}+qlora"
        ),
    )
    finetuned_rag_predictions = _run_rag_predictions(
        workflow=finetuned_workflow,
        workflow_status=WorkflowStatus,
        document_index=document_index,
        actor=evaluation_actor,
        records=splits["test"],
    )
    evaluation = {
        "test_split_sha256": _sha256(data_paths["test"]),
        "model_base_protocol": (
            "The raw model receives record context when provided. The common "
            "deterministic safety/evidence gate routes requests for review or "
            "abstains when retrieval is insufficient."
        ),
        "systems": {
            "raw_model_base_diagnostic": evaluate_system(
                evaluation_records, baseline_predictions
            ),
            "model_base_with_safety_gate": evaluate_system(
                evaluation_records, safe_baseline_predictions
            ),
            "rag": evaluate_system(evaluation_records, rag_predictions),
            "finetuning_plus_rag": evaluate_system(
                evaluation_records, finetuned_rag_predictions
            ),
        },
        "limitations": _evaluation_limitations(
            test_count=len(splits["test"]),
            smoke_test=smoke_test,
            uses_pubmedqa=uses_pubmedqa,
            uses_medquad=uses_medquad,
        ),
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "model": model_config,
        "method": config["method"],
        "training_config": training_config,
        "seed": training_config["seed"],
        "smoke_test": smoke_test,
        "resume_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        ),
        "train_samples_used": len(training_records),
        "validation_samples_used": len(splits["validation"]),
        "test_samples_used_for_evaluation_only": len(splits["test"]),
        "contexts_truncated_to_fit_max_sequence": {
            "train": train_contexts_truncated,
            "validation": validation_contexts_truncated,
            "test": sum(
                int(prediction["context_truncated"])
                for prediction in baseline_predictions
            ),
        },
        "split_sha256": {split: _sha256(path) for split, path in data_paths.items()},
        "baseline_predictions": baseline_predictions,
        "safe_baseline_predictions": safe_baseline_predictions,
        "rag_predictions": rag_predictions,
        "finetuned_predictions": tuned_predictions,
        "finetuned_rag_predictions": finetuned_rag_predictions,
        "evaluation": evaluation,
        "train_metrics": train_result.metrics,
    }
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _generate_predictions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    max_input_length: int,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for record in records:
        input_ids, _, context_truncated = _fit_context(
            tokenizer,
            question=record["question"],
            answer=None,
            context=record.get("context"),
            max_length=max_input_length,
            source_id=record["source_id"],
        )
        answer = _generate_answer(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        predictions.append(
            {
                "source_id": record["source_id"],
                "prediction": answer,
                "reference": record["answer"],
                "context_truncated": context_truncated,
            }
        )
    return predictions


def _run_rag_predictions(
    *,
    workflow: Any,
    workflow_status: Any,
    document_index: Any,
    actor: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        result = workflow.run(record["question"], actor=actor)
        if result.response is None:
            raise RuntimeError(
                f"RAG workflow returned no response for {record['source_id']!r}"
            )
        retrieved_source_ids = (
            [
                item.chunk.source_id
                for item in document_index.search(
                    record["question"], limit=5, min_score=0.5
                )
            ]
            if result.status is not workflow_status.REVIEW_REQUIRED
            else []
        )
        if result.status is workflow_status.REVIEW_REQUIRED:
            behavior = "review"
        elif result.status is workflow_status.INSUFFICIENT_EVIDENCE:
            behavior = "abstain"
        elif result.status is workflow_status.COMPLETE:
            behavior = "answer"
        else:
            raise RuntimeError(
                f"RAG workflow failed for {record['source_id']!r}: {result.status}"
            )
        predictions.append(
            {
                "source_id": record["source_id"],
                "prediction": result.response.answer,
                "reference": record["answer"],
                "observed_behavior": behavior,
                "retrieved_source_ids": retrieved_source_ids,
                "citation_source_ids": [
                    citation.source_id for citation in result.response.sources
                ],
            }
        )
    return predictions


def _apply_model_safety_gates(
    *,
    workflow: Any,
    workflow_status: Any,
    actor: Any,
    records: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(records) != len(raw_predictions):
        raise ValueError("Every evaluation record must have a base-model prediction")

    guarded_predictions: list[dict[str, Any]] = []
    for record, raw_prediction in zip(records, raw_predictions):
        if raw_prediction["source_id"] != record["source_id"]:
            raise ValueError("Base-model prediction order does not match evaluation data")
        result = workflow.run(record["question"], actor=actor)
        if result.status is workflow_status.REVIEW_REQUIRED:
            if result.response is None:
                raise RuntimeError("Safety gate returned no review response")
            answer = result.response.answer
            behavior = "review"
        elif result.status is workflow_status.INSUFFICIENT_EVIDENCE:
            if result.response is None:
                raise RuntimeError("Evidence gate returned no abstention response")
            answer = result.response.answer
            behavior = "abstain"
        elif result.status is workflow_status.COMPLETE:
            answer = (
                "Uma fonte foi recuperada, mas não foi fornecida ao modelo-base. "
                "Para evitar uma resposta sem fundamentação, consulte a fonte citada."
            )
            behavior = "abstain"
        else:
            raise RuntimeError(
                f"Base-model safety gate failed for {record['source_id']!r}: "
                f"{result.status}"
            )

        retrieved_source_ids = (
            [citation.source_id for citation in result.response.sources]
            if result.response is not None
            else []
        )
        answer, citation_source_ids = _append_retrieved_citations(
            answer,
            retrieved_source_ids,
        )
        guarded_predictions.append(
            {
                "source_id": record["source_id"],
                "prediction": answer,
                "reference": record["answer"],
                "observed_behavior": behavior,
                "retrieved_source_ids": retrieved_source_ids,
                "citation_source_ids": citation_source_ids,
            }
        )
    return guarded_predictions


def _append_retrieved_citations(
    answer: str,
    source_ids: list[str],
) -> tuple[str, list[str]]:
    if not source_ids:
        return answer, []
    missing_citations = [
        source_id for source_id in source_ids if f"[{source_id}]" not in answer
    ]
    if missing_citations:
        citation_text = ", ".join(f"[{source_id}]" for source_id in missing_citations)
        answer = f"{answer.rstrip()}\n\nFontes: {citation_text}"
    return answer, source_ids


def _generate_answer(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    *,
    max_new_tokens: int,
) -> str:
    import torch

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        generated[0, input_tensor.shape[1] :],
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned QLoRA baseline and experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/qwen2_5_0_5b_qlora.json"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use only checked-in synthetic data and one optimizer step.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume an experiment from a saved Trainer checkpoint.",
    )
    args = parser.parse_args()
    summary_path = run_experiment(
        args.config,
        smoke_test=args.smoke_test,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Experiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
