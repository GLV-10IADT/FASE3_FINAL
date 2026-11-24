from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class TextEmbedder(Protocol):
    @property
    def version(self) -> str: ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class LocalE5Embedder:
    model_name = "intfloat/multilingual-e5-small"
    model_revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        if max_length < 1:
            raise ValueError("max_length must be greater than zero")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require the project training dependencies."
            ) from exc

        model_options: dict[str, str] = {"revision": self.model_revision}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_options["cache_dir"] = str(cache_dir)

        self._torch = torch
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            **model_options,
        )
        self._model = AutoModel.from_pretrained(self.model_name, **model_options)
        self._model.to(self._device)
        self._model.eval()

    @property
    def version(self) -> str:
        return f"{self.model_name}@{self.model_revision}"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> Sequence[float]:
        if not text.strip():
            raise ValueError("Embedding query must not be empty")
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        embeddings: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {
                name: tensor.to(self._device)
                for name, tensor in encoded.items()
            }
            with self._torch.inference_mode():
                output = self._model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                attention_mask = attention_mask.to(token_embeddings.dtype)
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled /= attention_mask.sum(dim=1).clamp(min=1e-9)
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(
                tuple(float(value) for value in row)
                for row in normalized.cpu().tolist()
            )
        return tuple(embeddings)
