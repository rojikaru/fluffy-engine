import os
from dataclasses import dataclass
from typing import Protocol, Any, Generator, TypeVar

import dotenv
import openai
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

dotenv.load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str | None
    api_key: str
    model: str
    # Provider-specific request fields, e.g. switching off thinking for local Qwen
    extra_body: dict[str, Any] = field(default_factory=dict)


NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

PROVIDERS: dict[str, LLMConfig] = {
    "ollama": LLMConfig("http://localhost:11434/v1", "ollama", "qwen3.5:9b"),
    "docker": LLMConfig("http://localhost:12434/engines/llama.cpp/v1", "docker", "ai/qwen3.5:9b", NO_THINKING),
    "openai": LLMConfig(None, "", "gpt-5.6-terra"),
}
DEFAULT_PROVIDER = "ollama"


def config_from_env() -> LLMConfig:
    """
    Resolve the LLM config from `LLM_PROVIDER` (a key of `PROVIDERS`, default "ollama"),
    with `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` overriding the preset.
    Local providers ignore the key, but the OpenAI SDK requires a non-empty one.
    """
    preset = PROVIDERS[os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)]
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", preset.base_url),
        api_key=os.getenv("LLM_API_KEY") or preset.api_key or os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("LLM_MODEL", preset.model),
        extra_body=preset.extra_body,
    )


@dataclass
class ChatResult:
    id: str
    model: str
    text: str
    usage: dict
    message: dict  # Raw message object
    original_response: dict  # The original response from the LLM


class LLMClient(Protocol):
    """
    A protocol for a chat-based LLM client.
    This protocol defines methods for sending chat-style completion requests,
    including support for structured outputs and streaming responses.
    """

    # For structured outputs hinting
    T = TypeVar("T", bound=BaseModel)

    def chat(
            self,
            messages: str | list[dict[str, str]],
            **kwargs: Any
    ) -> ChatResult:
        """
        Send a chat-style completion request.
        :param messages: List of {"role": ..., "content": ...}
        :return: A lightweight object containing the response text, usage, raw message, id, and model.
        """
        ...

    def structured_output(
            self,
            messages: str | list[dict[str, Any]],
            schema: type[T],
            **kwargs: Any
    ) -> T:
        """
        Send a chat-style completion request expecting a structured output.
        
        :param messages: A string or a list of messages in the format [{"role": ..., "content": ...}]
        :param schema: A class that inherits from pydantic.BaseModel, defining the expected structure of the output.
        :param kwargs: Any additional native parameters to pass to the model.
        :return: A dictionary containing the structured output.
        """
        ...

    def stream(
            self,
            messages: str | list[dict[str, str]],
            **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Send a chat-style completion with streaming.
        :param messages: List of {"role": ..., "content": ...}
        :return: A generator yielding partial strings (tokens).
        """
        ...


class OpenAIClient:
    """
    A client for any OpenAI-compatible Chat Completions API: OpenAI, Ollama, Docker Model Runner.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or config_from_env()
        self.model = self.config.model
        self.client = openai.OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def chat(
            self,
            messages: str | list[dict[str, str]],
            **kwargs: Any
    ) -> ChatResult:
        response = self.client.responses.create(
            model=kwargs.get("model", self.model),
            input=messages,
            stream=False,
            **kwargs
        )
        return ChatResult(
            id=response.id,
            model=response.model,
            text=response.output_text,
            usage=response.usage.model_dump(),
            message=response.output[0].model_dump(),
            original_response=response.model_dump(),
        )

    def structured_output(
            self,
            messages: str | list[dict[str, Any]],
            schema: type[BaseModel],
            **kwargs: Any
    ) -> BaseModel:
        response = self.client.responses.parse(
            model=kwargs.get("model", self.model),
            input=messages,
            text_format=schema,
            stream=False,
            **kwargs
        )
        return response.output_parsed

    def stream(
            self,
            messages: str | list[dict],
            **kwargs: Any
    ) -> Generator[str, None, None]:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            stream=True,
            **kwargs
        )

        # yield responses from OpenAI's streaming API
        for event in response:
            if isinstance(event, ResponseTextDeltaEvent):
                yield event.delta
