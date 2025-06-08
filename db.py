from datetime import datetime, UTC

import chromadb
from chromadb import EmbeddingFunction

from ai.embedders import embedding_factory


def get_collection(collection_name: str, **kwargs: any) -> chromadb.Collection:
    """
    Get a collection from the ChromaDB client.
    If the collection does not exist, it will be created.

    :param collection_name: Name of the collection to retrieve or create.
    :keyword client: Optional ChromaDB client instance. If not provided, a new PersistentClient will be created.
    :keyword embedding_function: Optional embedding function to use for the collection.
    :keyword data_loader: Optional data loader to use for the collection.

    :return: The collection instance.
    """
    client = kwargs.get('client', chromadb.PersistentClient())
    data_loader = kwargs.get('data_loader', None)

    ef = kwargs.get('embedding_function', embedding_factory())
    if not isinstance(ef, EmbeddingFunction):
        raise TypeError("embedding_function must be an instance of EmbeddingFunction")

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        data_loader=data_loader,
        metadata={
            "created_by": "ETL Pipeline",
            'created_at': str(datetime.now(UTC)),
        }
    )
