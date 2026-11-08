import os
from dataclasses import dataclass, field
from typing import Protocol, Any, Generator, TypeVar

import dotenv
import openai
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


def env_provider() -> str:
    return os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)


def config_from_env(provider: str | None = None) -> LLMConfig:
    """
    Resolve the LLM config for `provider` (a key of `PROVIDERS`, default `LLM_PROVIDER` or "ollama").
    For the provider selected by `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` override the preset.
    Local providers ignore the key, but the OpenAI SDK requires a non-empty one.
    """
    provider = provider or env_provider()
    preset = PROVIDERS[provider]
    overrides = os.environ if provider == env_provider() else {}
    return LLMConfig(
        base_url=overrides.get("LLM_BASE_URL", preset.base_url),
        api_key=overrides.get("LLM_API_KEY") or preset.api_key or os.getenv("OPENAI_API_KEY", ""),
        model=overrides.get("LLM_MODEL", preset.model),
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


def as_messages(messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else messages


class OpenAIClient:
    """
    A client for any OpenAI-compatible Chat Completions API: OpenAI, Ollama, Docker Model Runner.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or config_from_env()
        self.model = self.config.model
        self.client = openai.OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def _request(self, messages: str | list[dict[str, Any]], kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": as_messages(messages),
            "extra_body": self.config.extra_body,
            **kwargs,
        }

    def chat(
            self,
            messages: str | list[dict[str, str]],
            **kwargs: Any
    ) -> ChatResult:
        response = self.client.chat.completions.create(**self._request(messages, kwargs))
        message = response.choices[0].message
        return ChatResult(
            id=response.id,
            model=response.model,
            text=message.content or "",
            usage=response.usage.model_dump() if response.usage else {},
            message=message.model_dump(),
            original_response=response.model_dump(),
        )

    def structured_output(
            self,
            messages: str | list[dict[str, Any]],
            schema: type[BaseModel],
            **kwargs: Any
    ) -> BaseModel:
        response = self.client.chat.completions.parse(response_format=schema, **self._request(messages, kwargs))
        return response.choices[0].message.parsed

    def stream(
            self,
            messages: str | list[dict],
            **kwargs: Any
    ) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(stream=True, **self._request(messages, kwargs))
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
