# these three lines swap the stdlib sqlite3 lib with the pysqlite3@latest package
# this should mitigate old version issues with chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from datetime import datetime, timezone
from functools import cache

import dotenv
import chromadb
from chromadb import EmbeddingFunction

from src.ai.embedders import EmbedderName, embedding_factory

dotenv.load_dotenv()


COLLECTION_EMBEDDINGS: dict[str, EmbedderName] = {
    "texts": "huggingface",
    "articles": "noop",
    "images": "openclip",
}


@cache
def db_client_factory() -> chromadb.Client:
    """
    Create a ChromaDB client with the default settings.
    If the environment variables `CHROMA_SERVER_HOST` and `CHROMA_SERVER_AUTHN_CREDENTIALS`,
    it will connect to a remote ChromaDB server.
    Otherwise, it will create a persistent client that uses the local file system.

    :return: chromadb.Client
    """
    host = os.getenv("CHROMA_SERVER_HOST")
    header = os.getenv("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER")
    provider = os.getenv("CHROMA_CLIENT_AUTH_PROVIDER")
    creds = os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")

    if all([host, header, provider, creds]):
        return chromadb.HttpClient(
            host=host,
            port=8000,
            ssl=True,
        )

    return chromadb.PersistentClient()


def get_collection(collection_name: str, **kwargs: any) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection with its embedding function from `COLLECTION_EMBEDDINGS`.
    """

    client = kwargs.get("client") or db_client_factory()
    data_loader = kwargs.get("data_loader", None)

    ef = kwargs.get("embedding_function") or embedding_factory(COLLECTION_EMBEDDINGS.get(collection_name, "openclip"))
    if not isinstance(ef, EmbeddingFunction):
        raise TypeError("`embedding_function` must be a chromadb.EmbeddingFunction")

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        data_loader=data_loader,
        metadata={
            "created_by": "ETL Pipeline",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
