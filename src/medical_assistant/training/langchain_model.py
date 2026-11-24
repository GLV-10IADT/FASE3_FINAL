from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class TransformersChatModel(BaseChatModel):
    model: Any
    tokenizer: Any
    max_new_tokens: int = 192

    @property
    def _llm_type(self) -> str:
        return "local-transformers-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        if stop:
            raise ValueError("Stop sequences are not supported by this local adapter")
        if kwargs:
            raise ValueError(f"Unsupported generation options: {', '.join(kwargs)}")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be greater than zero")

        chat_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message.content, str):
                raise TypeError("The local chat adapter accepts text-only messages")
            role = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
            }.get(message.type)
            if role is None:
                raise ValueError(f"Unsupported LangChain message type: {message.type!r}")
            chat_messages.append({"role": role, "content": message.content})

        input_ids = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        import torch

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.model.device,
        )
        self.model.eval()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        answer = self.tokenizer.decode(
            generated[0, input_tensor.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=answer)),
            ]
        )
