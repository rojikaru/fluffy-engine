import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.api import Embeddable
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


def embedding_factory(
        embedding_function: str | None = None
) -> EmbeddingFunction[Embeddable]:
    """
    Factory function to create an embedding function based on the provided name.
    :param embedding_function: Name of the embedding function to create.
    :return: An instance of the specified embedding function.
    """

    match embedding_function:
        case "noop":
            return NoOpEmbeddingFunction()
        case "openai":
            return OpenAIEmbeddingFunction()
        case "huggingface":
            return HuggingFaceEmbeddingFunction()
        case _:  # Default to OpenCLIPEmbeddingFunction
            return OpenCLIPEmbeddingFunction()


class NoOpEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """
    A no-op embedding function that returns a zero vector for each input document.
    This is useful for testing or when embeddings are not needed.
    """

    def __init__(self) -> None:
        """
        Initialize the NoOp embedding function.
        This function doesn't require any configuration parameters.
        """
        # No initialization needed for NoOp
        pass

    def __call__(self, _input: Documents) -> Embeddings:
        """
        Generate zero embeddings for the input documents.
        :param _input: List of documents to embed.
        :return: List of zero embeddings for each document.
        """
        return [np.array([0.0], dtype=np.float32) for _ in _input]


class OpenAIEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """
    A wrapper for OpenAI's embedding API.
    This class uses the OpenAI API to generate embeddings for a list of texts.
    It requires the OpenAI Python client library to be installed.
    Usage:
        embedding_function = OpenAIEmbeddingFunction(model="text-embedding-ada-002")
        embeddings = embedding_function(["text1", "text2"])
    """

    def __init__(self, model="text-embedding-ada-002"):
        import openai
        self.model = model
        self.client = openai

    def __call__(self, texts: list[str]) -> Embeddings:
        response = self.client.Embedding.create(input=texts, model=self.model)
        return [d["embedding"] for d in response["data"]]


class HuggingFaceEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """
    A wrapper for Hugging Face's Sentence Transformers.
    This class uses the Sentence Transformers library to generate embeddings for a list of texts.
    Usage:
        embedding_function = HuggingFaceEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_function(["text1", "text2"])
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> Embeddings:
        return self.model.encode(texts, show_progress_bar=False).tolist()
