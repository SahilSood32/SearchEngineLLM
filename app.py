import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)

from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)

from langchain.agents import initialize_agent, AgentType

# IMPORTANT: Updated import
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()

# -------------------- Tools --------------------

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# -------------------- Streamlit --------------------

st.set_page_config(page_title="LangChain Search Chatbot")

st.title("🔎 LangChain Search Chatbot")

st.write(
    "Ask anything. The agent can use DuckDuckGo, Wikipedia and Arxiv."
)

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile",
        streaming=True,
    )

    tools = [
        search,
        wiki,
        arxiv,
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):

        st_callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False,
        )

        response = agent.run(
            prompt,
            callbacks=[st_callback],
        )

        st.markdown(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

