from datetime import datetime, UTC
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api import Embeddable, DataLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


def get_collection(
        collection_name: str,
        client: Optional[chromadb.Client] = None,
        embedding_function: Optional[EmbeddingFunction[Embeddable]] = None,
        data_loader: Optional[DataLoader] = None
):
    """
    Get a collection from the ChromaDB client.
    If the collection does not exist, it will be created.

    :param collection_name: Name of the collection to retrieve or create.
    :param client: Optional ChromaDB client instance. If not provided, a new PersistentClient will be created.
    :param embedding_function: Optional embedding function to use for the collection.
    :param data_loader: Optional data loader to use for the collection.

    :return: The collection instance.
    """
    client = client or chromadb.PersistentClient()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function or OpenCLIPEmbeddingFunction(),
        data_loader=data_loader,
        metadata={
            "created_by": "ETL Pipeline",
            'created_at': str(datetime.now(UTC)),
        }
    )
