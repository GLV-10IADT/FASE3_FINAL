import hashlib
import json
from pathlib import Path

from medical_assistant.training.langchain_model import TransformersChatModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_finetuned_chat_model(
    config_path: Path,
    *,
    adapter_dir: Path | None = None,
    max_new_tokens: int = 192,
) -> tuple[TransformersChatModel, str]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Local model dependencies are missing; install the training extras."
        ) from exc

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than zero")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a JSON object")
    model_config = config.get("model")
    method_config = config.get("method")
    if not isinstance(model_config, dict) or not isinstance(method_config, dict):
        raise ValueError("Training configuration is missing model or method settings")
    repository = model_config.get("repository")
    revision = model_config.get("revision")
    experiment_id = config.get("experiment_id")
    quantization = method_config.get("quantization")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (repository, revision, experiment_id)
    ):
        raise ValueError("Training configuration has invalid model provenance")
    if not isinstance(quantization, dict) or quantization.get("load_in_4bit") is not True:
        raise ValueError("Local QLoRA runtime requires a 4-bit quantization config")

    project_root = config_path.resolve().parents[2]
    configured_output = config.get("output_dir")
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise ValueError("Training configuration has no output_dir")
    resolved_adapter_dir = adapter_dir or (project_root / configured_output / "adapter")
    if not resolved_adapter_dir.is_absolute():
        resolved_adapter_dir = project_root / resolved_adapter_dir
    adapter_config_path = resolved_adapter_dir / "adapter_config.json"
    adapter_weights_path = resolved_adapter_dir / "adapter_model.safetensors"
    tokenizer_config_path = resolved_adapter_dir / "tokenizer_config.json"
    for required_file in (
        adapter_config_path,
        adapter_weights_path,
        tokenizer_config_path,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Fine-tuned adapter artifact is missing: {required_file}"
            )

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, dict):
        raise ValueError("Adapter configuration must contain a JSON object")
    if adapter_config.get("base_model_name_or_path") != repository:
        raise ValueError("Adapter base model does not match the pinned configuration")
    if (
        adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("Adapter is not a compatible causal-language-model LoRA")
    if not torch.cuda.is_available():
        raise RuntimeError("The local QLoRA adapter requires a compatible CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_adapter_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_use_double_quant=quantization["double_quantization"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        repository,
        revision=revision,
        quantization_config=bitsandbytes_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        resolved_adapter_dir,
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()
    adapter_sha256 = _sha256(adapter_weights_path)
    model_version = (
        f"{repository}@{revision}+qlora:{experiment_id}:{adapter_sha256[:16]}"
    )
    return (
        TransformersChatModel(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        ),
        model_version,
    )
