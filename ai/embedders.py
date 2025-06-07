import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.api import Embeddable


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
