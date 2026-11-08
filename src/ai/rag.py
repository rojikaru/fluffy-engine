from typing import Literal, Any, Generator

from pydantic import BaseModel

from src.ai.llm import LLMClient, ChatResult
from src.db import get_collection


class IsRAGableResponseSchema(BaseModel):
    label: Literal["RAGABLE", "NOT_RAGABLE"] = "NOT_RAGABLE"


class RagResponseSchema(BaseModel):
    response: ChatResult | Generator[str, None, None]
    rag_used: bool = False
    articles: list[dict] | None = None
    images: list[dict] | None = None


# A zero-shot classifier to determine if a question
# is suitable for RAG (Retrieval-Augmented Generation).
# It is a prototype that filters out junk,
# like "What is the weather today?" or "How to cook pasta?",
# but it is too strict for now.
# TODO: make it more flexible and allow more questions.
def is_ragable(
        question: str,
        llm: LLMClient,
        system_prompt: str = (
                "You are a classification assistant. "
                "Your task is to determine if a question is suitable "
                "for RAG (Retrieval-Augmented Generation). "
                "Label as NOT_RAGABLE only for clearly easy questions "
                "like cooking, sports, greetings, code help etc. "
                "This means that you should label as RAGABLE any question "
                "that you couldn't answer without additional context. "
                "Always respond in JSON with exactly one key “label” "
                "whose value is either “RAGABLE” or “NOT_RAGABLE”. "
        ),
        **model_kwargs: Any,
) -> bool:
    """
    A zero-shot classifier to determine if a question is suitable for RAG (Retrieval-Augmented Generation).
    :param question: The user question to classify.
    :param llm: The LLM client to use for classification.
    :param system_prompt: A system prompt to guide the LLM's classification.
    :param model_kwargs: Additional parameters to pass to the LLM model.
    :return: bool: True if the question is RAGable, False otherwise.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # call out to the provided LLM client
    resp = llm.structured_output(
        messages,
        IsRAGableResponseSchema,
        **model_kwargs,
    )
    return resp.label == "RAGABLE"


def relevant_excerpts(
        collection,
        question: str,
        threshold: float,
        margin: float,
        n_results: int = 10,
) -> list[tuple[str, dict]]:
    """
    Return (text, metadata) of chunks within `threshold` and within `margin` of the best hit,
    most relevant first. Distances are squared L2 over unit vectors: d = 2 - 2 * cosine similarity.
    The margin drops loosely related chunks (same company, different story) that sit just
    under the absolute threshold when one article clearly answers the question.
    """
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    distances = result['distances'][0]
    cutoff = min(threshold, distances[0] + margin) if distances else threshold
    hits = zip(result['documents'][0], result['metadatas'][0], distances)
    return [(text, meta) for text, meta, distance in hits if distance <= cutoff]


def source_articles(collection, excerpts: list[tuple[str, dict]]) -> dict[str, dict]:
    """Metadata of the articles the excerpts come from, keyed by id, in relevance order."""
    ids = list(dict.fromkeys(meta['article_id'] for _, meta in excerpts))
    if not ids:
        return {}
    found = collection.get(ids=ids, include=["metadatas"])
    by_id = dict(zip(found['ids'], found['metadatas']))
    return {article_id: by_id[article_id] for article_id in ids if article_id in by_id}


def article_images(collection, question: str, articles: dict[str, dict], n_results: int = 3) -> list[dict]:
    """
    Hero images of the source articles, ranked by text-to-image similarity with the question.
    Restricting to source articles avoids a global threshold: CLIP text-to-image distances
    cluster too tightly (about 0.70 for a match vs 0.80+ otherwise) to cut reliably.
    """
    if not articles:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=min(n_results, len(articles)),
        where={"article_id": {"$in": list(articles)}},
        include=["metadatas"],
    )
    return [
        {
            "url": meta["url"],
            "title": articles[meta["article_id"]].get("title", ""),
            "canonical_url": articles[meta["article_id"]].get("canonical_url", ""),
        }
        for meta in result["metadatas"][0]
    ]


def format_context(excerpts: list[tuple[str, dict]], articles: dict[str, dict]) -> str:
    def block(text: str, meta: dict) -> str:
        article = articles.get(meta['article_id'], {})
        heading = f"[{article.get('title', 'Untitled')}, {article.get('date', 'n.d.')}]"
        subtitle = f"{meta['subtitle']}: " if meta.get('subtitle') else ""
        return f"{heading}\n{subtitle}{text}"

    return "\n\n".join(block(text, meta) for text, meta in excerpts)


def _internal_llm_call(
        llm: LLMClient,
        system_prompt: str,
        question: str,
        context: str | None = None,
        stream: bool = True,
        **model_kwargs: Any,
) -> ChatResult | Generator:
    # One system message: some chat templates (e.g. Qwen) reject a system message that isn't first
    if context:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"Answer using the following excerpts from the AI magazine ‘The Batch’:\n\n{context}"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if stream:
        return llm.stream(messages, **model_kwargs)
    return llm.chat(messages, **model_kwargs)


def rag_call(
        question: str,
        llm: LLMClient,
        system_prompt: str = (
                "You are an expert assistant on AI news. "
        ),
        stream: bool = True,
        rag_threshold: float = 0.75,
        rag_margin: float = 0.25,
        **model_kwargs: Any,
) -> RagResponseSchema:
    """
    A function to perform a RAG (Retrieval-Augmented Generation) call.
    It uses the provided LLM client to answer a question based on context.

    :param question: The user question to answer.
    :param llm: The LLM client to use for answering the question.
    :param system_prompt: A system prompt to guide the LLM's response.
    :param stream: Whether to stream the response or not.
    :param model_kwargs: Additional parameters to pass to the LLM model.

    :return: A data object with a response from the LLM,
             a flag if it used RAG or not,
             a list of articles used as context (if any),
             and a list of images related to the articles (if any).
    """

    texts_collection = get_collection("texts")
    articles_collection = get_collection("articles")
    images_collection = get_collection("images")

    excerpts = relevant_excerpts(texts_collection, question, rag_threshold, rag_margin)
    if not excerpts:
        response_stream = _internal_llm_call(llm, system_prompt, question, stream=stream, **model_kwargs)
        return RagResponseSchema(response=response_stream, rag_used=False)

    articles = source_articles(articles_collection, excerpts)

    response_stream = _internal_llm_call(
        llm,
        system_prompt,
        question,
        context=format_context(excerpts, articles),
        stream=stream,
        **model_kwargs
    )
    return RagResponseSchema(
        response=response_stream,
        rag_used=True,
        articles=list(articles.values()),
        images=article_images(images_collection, question, articles),
    )
