import sys
import types

import openai
import streamlit as st

from src.ai.llm import LLMConfig, OpenAIClient, PROVIDERS, config_from_env, env_provider
from src.ai.rag import rag_call
from src.db import get_collection


# Patch torch.classes to prevent Streamlit from scanning its __path__
class DummyModule(types.ModuleType):
    __path__ = []  # Fake path so Streamlit doesn't try to scan it
sys.modules['torch.classes'] = DummyModule('torch.classes')


def init_session_state():
    if 'texts' not in st.session_state:
        st.session_state['texts'] = get_collection('texts')

    if 'articles' not in st.session_state:
        st.session_state['articles'] = get_collection('articles')

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []


PROVIDER_LABELS = {"ollama": "Ollama", "docker": "Docker Model Runner", "openai": "OpenAI"}


def sidebar_llm_config() -> LLMConfig:
    providers = list(PROVIDERS)
    with st.sidebar:
        st.header("Model")
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(env_provider()),
            format_func=lambda name: PROVIDER_LABELS.get(name, name),
        )
        defaults = config_from_env(provider)
        # Keys are per provider, so switching providers restores that provider's defaults
        base_url = st.text_input("Base URL", value=defaults.base_url or "", key=f"base_url_{provider}",
                                 help="Leave empty for the provider's official endpoint.")
        model = st.text_input("Model", value=defaults.model, key=f"model_{provider}")
        api_key = st.text_input("API key", type="password", key=f"api_key_{provider}",
                                help="Leave empty to use the key from `.env`.")

    return LLMConfig(
        base_url=base_url or None,
        api_key=api_key or defaults.api_key,
        model=model or defaults.model,
        extra_body=defaults.extra_body,
    )


def get_llm(config: LLMConfig) -> OpenAIClient:
    llm = st.session_state.get('llm')
    if llm is None or llm.config != config:
        llm = st.session_state['llm'] = OpenAIClient(config)
    return llm


def show_images(images: list[dict] | None) -> None:
    if not images:
        return
    st.markdown("### Images")
    for image in images:
        st.image(image['url'], caption=image.get('title') or None, width="stretch")


def articles_markdown(articles: list[dict] | None) -> str:
    if not articles:
        return ""
    links = "".join(f"- [{a.get('title', 'Reference')}]({a.get('canonical_url', '#')})\n" for a in articles)
    return f"\n\n### Articles used:\n{links}"


def stream_into(placeholder, stream) -> str:
    buffer = ""
    for chunk in stream:
        buffer += chunk
        placeholder.markdown(buffer + "▌")
    return buffer


def display_history():
    history = st.session_state['chat_history']
    for idx, entry in enumerate(history):
        st.chat_message("user").markdown(entry['query'])
        if not entry['rag_used']:
            st.warning("RAG was not used.")
        with st.chat_message("assistant"):
            st.markdown(entry['response'])
            show_images(entry.get('images'))
        if idx < len(history) - 1:
            st.markdown("---")


def display_chat(query: str, stream, rag_used: bool, articles, images) -> str:
    with st.container():
        st.chat_message("user").markdown(query)
        if not rag_used:
            st.warning(
                "RAG was not used. "
                "Please, check your prompt and try again "
                "if you expected RAG to be used."
            )
        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = stream_into(placeholder, stream) + articles_markdown(articles)
            placeholder.markdown(answer)
            show_images(images)
        return answer


def main():
    st.title("Fluffy Engine - RAG Helper")
    config = sidebar_llm_config()
    if not config.api_key:
        st.info("Enter an API key in the sidebar to use this provider.")
        st.stop()
    llm = get_llm(config)

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

        if error := st.session_state.pop('error', None):
            st.error(error)

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
                submitted = st.form_submit_button("Send", width="stretch")

        if submitted and query:
            st.markdown("---")  # Visual separator

            # Spinner while searching DB and preparing response
            with st.spinner("Searching the database..."):
                # Call the RAG function (may take time)
                response = rag_call(
                    question=query,
                    llm=llm,
                    stream=True,
                )

            try:
                response_text = display_chat(
                    query,
                    response.response,
                    response.rag_used,
                    response.articles,
                    response.images,
                )
            except openai.OpenAIError as error:
                # The chat output replaced the form; rerun to bring it back and show the error
                st.session_state['error'] = f"The model request failed: {error}"
                st.rerun()

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
