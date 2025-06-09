from typing import Literal, Any, Generator

from pydantic import BaseModel

from ai.llm import LLMClient, ChatResult
from db import get_collection


class IsRAGableResponseSchema(BaseModel):
    label: Literal["RAGABLE", "NOT_RAGABLE"] = "NOT_RAGABLE"


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


def _internal_llm_call(
        llm: LLMClient,
        system_prompt: str,
        question: str,
        articles: list[str] | None = None,
        stream: bool = True,
        **model_kwargs: Any,
) -> ChatResult | Generator:
    # Internal function to perform a call to the LLM with context from articles.

    # Prepare the messages for the LLM
    messages = [{"role": "system", "content": system_prompt}]
    if articles:
        for article in articles:
            messages.append({"role": "system", "content": article})
    messages.append({"role": "user", "content": question})

    if stream:
        return llm.stream(messages, **model_kwargs)
    else:
        return llm.chat(messages, **model_kwargs)


def rag_call(
        question: str,
        llm: LLMClient,
        system_prompt: str = (
                "You are an expert assistant. "
                "Use the following context from AI Magazine ‘The Batch’ to answer the question. "
        ),
        stream: bool = True,
        **model_kwargs: Any,
) -> tuple[ChatResult | Generator, bool, list[dict] | None]:
    """
    A function to perform a RAG (Retrieval-Augmented Generation) call.
    It uses the provided LLM client to answer a question based on context.

    :param question: The user question to answer.
    :param llm: The LLM client to use for answering the question.
    :param system_prompt: A system prompt to guide the LLM's response.
    :param stream: Whether to stream the response or not.
    :param model_kwargs: Additional parameters to pass to the LLM model.

    :return: Tuple with a response from the LLM,
             a flag if it used RAG or not,
             and a list of articles used as context (if any).
    """

    texts_collection = get_collection("texts")
    articles_collection = get_collection("articles")

    # Query the texts collection for relevant excerpts
    context_query = texts_collection.query(
        query_texts=[question],
        n_results=10,
    )
    # Metadatas are sliced by relevance to queries, we had one - we take one
    excerpts = context_query['documents'][0]
    excerpts_metas = context_query['metadatas'][0]
    distances = context_query['distances'][0]

    # Cosine similarity threshold to determine if the question is relevant to text
    ragable = distances[0] <= 0.75
    if not ragable:
        # just call the LLM without any articles
        return _internal_llm_call(
            llm,
            system_prompt,
            question,
            articles=None,
            stream=stream,
            **model_kwargs
        ), ragable, None

    # Query articles that contain the found excerpts
    orig_articles_ids = list(set(map(lambda res: res['article_id'], excerpts_metas)))
    articles_query = articles_collection.get(
        ids=orig_articles_ids,
        include=["metadatas", "documents", "embeddings"],
    )
    articles_titles, articles_meta = articles_query['documents'], articles_query['metadatas']

    for i, title in enumerate(articles_titles):
        # Convert to dict to ensure mutability
        d = dict(articles_meta[i])
        d['title'] = title or 'Reference'
        articles_meta[i] = d

    # call out to the provided LLM client with the articles as context
    return _internal_llm_call(
        llm,
        system_prompt,
        question,
        excerpts,
        stream,
        **model_kwargs
    ), ragable, articles_meta
