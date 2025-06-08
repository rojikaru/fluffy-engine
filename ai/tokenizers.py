from typing import Protocol
import tiktoken
from transformers import AutoTokenizer


class Tokenizer(Protocol):
    """
    A protocol for tokenizers that defines methods for counting tokens,
    tokenizing text, and truncating text to a maximum number of tokens.
    This protocol can be implemented by any tokenizer class that adheres to
    the specified methods.
    """

    # Count the number of tokens in a given text.
    def count_tokens(self, text: str) -> int: ...

    # Tokenize the given text into a list of tokens.
    def tokenize(self, text: str) -> list[int]: ...

    # Truncate the given text to a maximum number of tokens.
    def truncate(self, text: str, max_tokens: int) -> str: ...


def tokenizer_factory(provider: str) -> Tokenizer:
    """
    Factory function to create a tokenizer based on the specified provider.
    :param provider: The name of the tokenizer provider. Supported values are "openai" and "huggingface".
    :return: An instance of a tokenizer that implements the Tokenizer protocol.
    """

    match provider:
        case "openai":
            return OpenAITokenizer("text-embedding-3-small")
        case "huggingface":
            return HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        case _:
            raise ValueError(f"Unknown tokenizer provider: {provider}")


class OpenAITokenizer:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)[:max_tokens]
        return self.encoding.decode(tokens)


class HuggingFaceTokenizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
