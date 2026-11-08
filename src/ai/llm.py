import os
from dataclasses import dataclass
from typing import Protocol, Any, Generator, TypeVar

import dotenv
import openai
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

dotenv.load_dotenv()


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
    A client for OpenAI's chat-based LLMs.
    """

    def __init__(self, **client_kwargs):
        # Give room for custom one-off key passing
        openai.api_key = client_kwargs.pop("api_key", os.getenv("OPENAI_API_KEY"))

        self.model = client_kwargs.pop("model", "gpt-4o")
        self.client = openai.OpenAI()

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
