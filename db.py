from datetime import datetime, timezone
import numpy as np
import chromadb
from chromadb import EmbeddingFunction

import torch
from transformers import CLIPProcessor, CLIPModel

from ai.embedders import embedding_factory


# Torch processor setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)
_model = (CLIPModel
          .from_pretrained(
              "openai/clip-vit-base-patch32",
              use_safetensors=True,
              torch_dtype=torch.float32,
          )
          .to(DEVICE)
          .eval()
)


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Run CLIP text encoder and return a NumPy array of shape (len(texts), embed_dim)."""
    inputs = _processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        feats = _model.get_text_features(**inputs)
    return feats.cpu().numpy()


def get_collection(collection_name: str, **kwargs: any) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection, wiring up your standard embedding function.
    """
    client = kwargs.get("client", chromadb.PersistentClient())
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
