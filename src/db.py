# these three lines swap the stdlib sqlite3 lib with the pysqlite3@latest package
# this should mitigate old version issues with chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from datetime import datetime, timezone
from functools import cache

import dotenv
import numpy as np
import chromadb
from chromadb import EmbeddingFunction

import torch
from transformers import CLIPProcessor, CLIPModel

from src.ai.embedders import DEVICE, embedding_factory

dotenv.load_dotenv()


CLIP_MODEL = "openai/clip-vit-base-patch32"


@cache
def _clip() -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, use_safetensors=True)
    model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True, dtype=torch.float32)
    return processor, model.to(DEVICE).eval()


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Run CLIP text encoder and return a NumPy array of shape (len(texts), embed_dim)."""
    processor, model = _clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs).pooler_output
    return feats.cpu().numpy()


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
    Get or create a ChromaDB collection, wiring up your standard embedding function.
    """

    client = kwargs.get("client", db_client_factory())
    data_loader = kwargs.get("data_loader", None)

    ef = kwargs.get("embedding_function", embedding_factory())
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


def query_images_by_text(
    collection: chromadb.Collection,
    text_query: str,
    n_results: int = 3,
):
    """
    Find the top-n images in `collection` whose embeddings match `text_query`.
    """
    # 1) embed the query text
    q_emb = _embed_texts([text_query]).tolist()

    # 2) query chromadb
    result = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        include=["metadatas", "distances", "documents"],
    )

    return result
