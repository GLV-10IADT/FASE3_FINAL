import json
from pathlib import Path

import pytest

from medical_assistant.training.runtime import load_finetuned_chat_model


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "training" / "configs" / "qwen2_5_0_5b_internal_qlora.json"


def test_local_runtime_rejects_missing_adapter_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Fine-tuned adapter artifact is missing"):
        load_finetuned_chat_model(CONFIG_PATH, adapter_dir=tmp_path)


def test_local_runtime_rejects_adapter_for_a_different_base_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "different/model"}),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"synthetic test")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_finetuned_chat_model(CONFIG_PATH, adapter_dir=tmp_path)

