import sys
import types

import streamlit as st

from ai.llm import OpenAIClient
from ai.rag import rag_call
from db import get_collection


# Patch torch.classes to prevent Streamlit from scanning its __path__
class DummyModule(types.ModuleType):
    __path__ = []  # Fake path so Streamlit doesn't try to scan it
sys.modules['torch.classes'] = DummyModule('torch.classes')


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
    for idx, entry in enumerate(st.session_state['chat_history']):
        # User’s bubble
        st.chat_message("user").markdown(entry['query'])

        # RAG notice
        if not entry['rag_used']:
            st.warning("RAG was not used.")

        # Create an assistant bubble and stream into it
        assistant_msg = st.chat_message("assistant")
        with assistant_msg:
            st.markdown(entry['response'])

            # Display images if available
            if entry.get('images', None):
                st.markdown("\n\n### Images:\n")
                for image in entry['images']:
                    image_url = image.get('image', None)
                    caption = image.get('title', 'Image')
                    if image_url:
                        st.image(image_url, caption=caption, use_column_width=True)

        # Add a visual separator between messages
        if idx < len(st.session_state['chat_history']) - 1:
            st.markdown("---")


def display_chat(
        query: str,
        stream,
        rag_used: bool,
        articles,
        images,
):
    with st.container():
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

            # display images if available
            if images:
                st.markdown("\n\n### Images:\n")
                for image in images:
                    image_url = image.get('image', None)
                    caption = image.get('title', 'Image')
                    if image_url:
                        st.image(image_url, caption=caption, use_column_width=True)

        return buffer


def main():
    st.title("Fluffy Engine - RAG Helper")

    with st.spinner("Loading Fluffy's database..."):
        init_session_state()

    # Create a container for chat history
    chat_container = st.container()

    # Create a container for input (this will be rendered at the bottom)
    input_container = st.empty()

    # Display chat history in the chat container
    with chat_container:
        if st.session_state['chat_history']:
            with st.spinner("Loading your chat history..."):
                display_history()
        else:
            st.write(
                'Hello! I am Fluffy, your RAG assistant. '
                'I can help you answer questions using the latest AI Magazine articles.'
            )

        st.markdown("---")  # Visual separator

    # Input form at the bottom
    with input_container:
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                query = st.text_input(
                    label="Ask me something =)",
                    placeholder="Type your question here...",
                    key="chat_input"
                )

            with col2:
                # Spacer to align the button
                st.markdown(
                    "<div style='min-height: 28px;'></div>",
                    unsafe_allow_html=True
                )
                submitted = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                )

        if submitted and query:
            st.markdown("---")  # Visual separator

            # Spinner while searching DB and preparing response
            with st.spinner("Searching the database..."):
                # Call the RAG function (may take time)
                response = rag_call(
                    question=query,
                    llm=st.session_state['llm'],
                    stream=True,
                )

            # Display the chat with streaming response
            response_text = display_chat(
                query,
                response.response,
                response.rag_used,
                response.articles,
                response.images,
            )

            # Store in chat history
            st.session_state['chat_history'].append({
                'query': query,
                'response': response_text,
                'rag_used': response.rag_used,
                'articles': response.articles,
                'images': response.images,
            })

            # Rerun to update the display
            st.rerun()


if __name__ == "__main__":
    main()
