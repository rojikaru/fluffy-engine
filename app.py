import streamlit as st

from ai.llm import OpenAIClient
from ai.rag import rag_call
from db import get_collection


def init_session_state():
    if 'texts' not in st.session_state:
        st.session_state['texts'] = get_collection('texts')

    if 'articles' not in st.session_state:
        st.session_state['articles'] = get_collection('articles')

    if 'llm' not in st.session_state:
        st.session_state['llm'] = OpenAIClient()

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []


def display_history():
    for entry in st.session_state['chat_history']:
        # User’s bubble
        st.chat_message("user").markdown(entry['query'])

        # RAG notice
        if not entry['rag_used']:
            st.warning("RAG was not used.")

        # Create an assistant bubble and stream into it
        assistant_msg = st.chat_message("assistant")
        with assistant_msg:
            st.markdown(entry['response'])



def display_chat(
        query: str,
        stream,
        rag_used: bool,
        articles
):
    # User’s bubble
    st.chat_message("user").markdown(query)

    # RAG notice
    if not rag_used:
        st.warning(
            "RAG was not used. "
            "Please, check your prompt and try again "
            "if you expected RAG to be used."
        )

    # Create an assistant bubble and stream into it
    assistant_msg = st.chat_message("assistant")
    buffer = ""
    with assistant_msg:
        placeholder = st.empty()
        for chunk in stream:
            buffer += chunk
            placeholder.markdown(buffer + "▌")

        # Display the articles used
        if articles:
            buffer += "\n\n### Articles used:\n"
            for article in articles:
                title = article.get('title', 'Reference')
                url = article.get('canonical_url', '#')
                buffer += f"- [{title}]({url})\n"

        # final, without cursor
        placeholder.markdown(buffer)



    return buffer


def main():
    st.title("Fluffy Engine - RAG Helper")
    st.write(
        'Hello! I am Fluffy, your RAG assistant. '
        'I can help you answer questions using the latest AI Magazine articles.'
    )

    init_session_state()

    query = st.text_input(
        label="Ask me anything about AI Magazine articles:",
        value="",
    )

    display_history()
    if query:
        # Call the RAG function
        response, rag_used, articles = rag_call(
            question=query,
            llm=st.session_state['llm'],
            stream=True,
        )

        # Display the chat
        response_text = display_chat(query, response, rag_used, articles)

        # Store the chat history
        st.session_state['chat_history'].append({
            'query': query,
            'response': response_text,
            'rag_used': rag_used
        })


if __name__ == "__main__":
    main()
