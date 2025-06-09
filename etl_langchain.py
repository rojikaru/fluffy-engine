import io
import os
import uuid
from uuid import uuid5
from urllib.parse import unquote

import dotenv
import nltk
from bs4 import BeautifulSoup
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from PIL import Image
import requests
import numpy as np

from etl_pipeline import get_articles_links

dotenv.load_dotenv()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class ParagraphSplitter(TextSplitter):
    """A TextSplitter that preserves each <p> as one chunk."""

    def __init__(self):
        super().__init__(chunk_size=1, chunk_overlap=0)  # we override split_text

    def split_text(self, text: str) -> list[str]:
        soup = BeautifulSoup(text, "html.parser")
        paras = [p.get_text(strip=True) for p in soup.find_all("p") + soup.find_all('ul') if p.get_text(strip=True)]
        return paras


def run_etl_with_langchain(
        persist_dir: str = "./chroma",
        embedding_provider: str = "huggingface",
):
    urls = get_articles_links()

    # Choose embeddings for stores
    if embedding_provider == "huggingface":
        text_emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    elif embedding_provider == "google":
        text_emb = VertexAIEmbeddings(
            model_name="textembedding-gecko@001",
        )
    else:
        text_emb = OpenAIEmbeddings(
            model="text-embedding-ada-002",
        )

    # embed images using CLIP
    image_emb = OpenCLIPEmbeddings(
        model_name="ViT-g-14",
        checkpoint="laion2b_s34b_b88k"
    )

    # Init db collections
    text_store = Chroma(
        persist_directory=os.path.join(persist_dir, "texts"),
        collection_name="texts",
        embedding_function=text_emb,
    )
    article_store = Chroma(
        persist_directory=os.path.join(persist_dir, "articles"),
        collection_name="articles",
        embedding_function=text_emb,  # or a noop shim
    )
    image_store = Chroma(
        persist_directory=os.path.join(persist_dir, "images"),
        collection_name="images",
        embedding_function=image_emb,
    )

    # 2a) Loader + splitter for text paragraphs
    loader = UnstructuredURLLoader(urls=urls, headers={
        "User-Agent": os.getenv("USER_AGENT", "langchain-etl/1.0")
    })
    raw_docs = loader.load()  # List[Document(page_content=full HTML, metadata)]

    splitter = ParagraphSplitter()

    for doc in raw_docs:
        url = doc.metadata.get("source", "")
        # Generate a stable article_id
        article_id = uuid5(uuid.NAMESPACE_URL, url).hex

        # 2b) Article-level metadata doc: description + metadata
        title = BeautifulSoup(doc.page_content, "html.parser").find("h1")
        description = title.get_text(strip=True) if title else ""
        article_store.add_documents(
            ids=[article_id],
            documents=[Document(
                page_content=description,
                metadata={
                    "id": article_id,
                    "canonical_url": url,
                    "title": description[:100],
                    **{k: v for k, v in doc.metadata.items() if k != "source"}
                }
            )],
        )

        # 2c) Paragraph-level docs
        paras = splitter.split_text(doc.page_content)
        for idx, p in enumerate(paras):
            if not p.strip():
                continue

            text_store.add_documents(
                ids=[f"{article_id}_{idx}"],
                documents=[Document(
                    page_content=p,
                    metadata={
                        "article_id": article_id,
                        "paragraph_idx": idx,
                        "total_paragraphs": len(paras),
                        "canonical_url": url
                    }
                )],
            )

    # 2g) Image extraction & storage (fallback to manual, but still LangChain+Chroma)
    for url in urls:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser").find("article")

        img = (soup.find("noscript") or soup).find("img")
        if not img or not img.get("src"):
            continue

        # normalize src
        if img['src'].startswith('/_next/image/?url='):
            # If its Next.js lazy loaded image, extract the encoded URL
            # And remove parameters if present (because they clog the extension)
            img = (
                img['src'].replace('/_next/image/?url=', '')
                .split('&', 1)[0]
            )
        else:
            img = img['src']
        # Decode URL-encoded image path (if any)
        img = unquote(img)

        img_data = requests.get(img).content
        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        arr = np.array(pil)

        # one image per article_id (for now)
        article_id = uuid.uuid5(uuid.NAMESPACE_URL, url).hex
        image_store.add_documents(
            ids=[article_id],
            documents=[Document(
                img_data=arr.tobytes(),  # bytes payload
                page_content=img,  # store the URL as text
                metadata={"article_id": article_id, "url": img}
            )],
        )

    print("ETL complete. Collections persisted to:", persist_dir)


if __name__ == "__main__":
    run_etl_with_langchain(persist_dir='./chroma_db1')
