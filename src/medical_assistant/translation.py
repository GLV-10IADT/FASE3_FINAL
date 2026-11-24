from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class TextTranslator(Protocol):
    def translate(self, text: str) -> str: ...


class LocalOpusTranslator:
    def __init__(
        self,
        translator: Any,
        sentencepiece: Any,
        version: str = "injected",
    ) -> None:
        self._translator = translator
        self._sentencepiece = sentencepiece
        self.version = version

    @classmethod
    def from_model_directory(
        cls,
        directory: Path,
        *,
        from_code: str,
        to_code: str,
    ) -> LocalOpusTranslator:
        model_directory = directory / "model"
        sentencepiece_path = directory / "sentencepiece.model"
        metadata_path = directory / "metadata.json"
        attribution_path = directory / "OPUS-MT-ATTRIBUTION" / "README.md"
        for required_path in (
            model_directory / "model.bin",
            sentencepiece_path,
            metadata_path,
            attribution_path,
        ):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Local translation model is incomplete: {required_path}. "
                    "Run scripts/prepare_translation_models.py first."
                )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("from_code") != from_code
            or metadata.get("to_code") != to_code
            or metadata.get("package_version") != "1.9"
        ):
            raise ValueError(f"Unexpected OPUS-MT metadata in {directory}")

        try:
            import ctranslate2
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "Local translation dependencies are missing; install "
                "the translation extra with pip install -e \".[translation]\"."
            ) from exc

        translator = ctranslate2.Translator(
            str(model_directory),
            device="cpu",
            inter_threads=1,
        )
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(sentencepiece_path)
        )
        version = f"OPUS-MT {metadata['package_version']} {from_code}-{to_code}"
        return cls(translator, tokenizer, version)

    def translate(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        tokens = self._sentencepiece.encode(text.strip(), out_type=str)
        if not tokens:
            raise ValueError("text must contain translatable content")
        results = self._translator.translate_batch(
            [tokens],
            beam_size=4,
            max_decoding_length=256,
        )
        if not results or not results[0].hypotheses:
            raise RuntimeError("Local translation returned no result")

        translated = "".join(results[0].hypotheses[0])
        translated = translated.replace("▁", " ").strip()
        translated = " ".join(translated.split())
        if not translated:
            raise RuntimeError("Local translation returned an empty result")
        return translated
