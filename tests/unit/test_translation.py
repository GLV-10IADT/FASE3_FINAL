from types import SimpleNamespace

import pytest

from medical_assistant.translation import LocalOpusTranslator


class FakeSentencePiece:
    def __init__(self) -> None:
        self.encoded_text = ""

    def encode(self, text: str, *, out_type: type[str]) -> list[str]:
        assert out_type is str
        self.encoded_text = text
        return ["▁translated", "▁text", "."]


class FakeCTranslate2:
    def __init__(self, hypotheses: list[str]) -> None:
        self.hypotheses = hypotheses
        self.input_tokens: list[list[str]] = []
        self.options: dict[str, int] = {}

    def translate_batch(
        self,
        batch: list[list[str]],
        *,
        beam_size: int,
        max_decoding_length: int,
    ) -> list[SimpleNamespace]:
        self.input_tokens = batch
        self.options = {
            "beam_size": beam_size,
            "max_decoding_length": max_decoding_length,
        }
        return [SimpleNamespace(hypotheses=[self.hypotheses])]


def test_local_opus_translation_decodes_sentencepiece_boundaries() -> None:
    sentencepiece = FakeSentencePiece()
    engine = FakeCTranslate2(["▁Texto", "▁traduzido", "."])
    translator = LocalOpusTranslator(engine, sentencepiece)

    result = translator.translate("  A pergunta original  ")

    assert result == "Texto traduzido."
    assert sentencepiece.encoded_text == "A pergunta original"
    assert engine.input_tokens == [["▁translated", "▁text", "."]]
    assert engine.options == {"beam_size": 4, "max_decoding_length": 256}


@pytest.mark.parametrize("text", ["", "  "])
def test_local_opus_translation_rejects_empty_input(text: str) -> None:
    translator = LocalOpusTranslator(FakeCTranslate2([]), FakeSentencePiece())

    with pytest.raises(ValueError, match="non-empty string"):
        translator.translate(text)


def test_local_opus_translation_reports_empty_engine_result() -> None:
    class EmptyEngine:
        def translate_batch(
            self,
            batch: list[list[str]],
            *,
            beam_size: int,
            max_decoding_length: int,
        ) -> list[SimpleNamespace]:
            del batch, beam_size, max_decoding_length
            return []

    translator = LocalOpusTranslator(EmptyEngine(), FakeSentencePiece())

    with pytest.raises(RuntimeError, match="no result"):
        translator.translate("Texto para traduzir")


def test_local_opus_translator_reports_missing_model_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_translation_models.py"):
        LocalOpusTranslator.from_model_directory(
            tmp_path,
            from_code="pt",
            to_code="en",
        )
