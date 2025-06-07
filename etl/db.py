from datetime import datetime, UTC
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api import Embeddable

from ai.embedders import local_openclip_ef


def get_collection(
        collection_name: str,
        client: Optional[chromadb.Client] = None,
        embedding_function: Optional[
            EmbeddingFunction[Embeddable]
        ] = None,
):
    """
    Get a collection from the ChromaDB client.
    If the collection does not exist, it will be created.
    """
    client = client or chromadb.PersistentClient()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function or local_openclip_ef(),
        metadata={
            "created_by": "ETL Pipeline",
            'created_at': str(datetime.now(UTC)),
        }
    )
