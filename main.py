from typing import Set
import streamlit as st
from core import run_llm
from streamlit_chat import message

st.header("LangChain Documentation Helper Bot")

password_secret = st.secrets.get("PASSWORD")

if password_secret is None:
    st.error("Required secrets are missing. Please check your secrets configuration.")
    st.stop()

password = st.text_input("Enter password:", type="password")
if password != password_secret.get("password"):
    st.error("Reach out to cingul@usc.edu for a password")
    st.stop()

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...", key='prompt_key')

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

        # clear the chat input
        st.text_input("Prompt", value='', key='prompt_key')

if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(
        zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        )
    ):
        message(user_query, is_user=True, key=f"user_msg_{i}")
        message(generated_response, key=f"bot_msg_{i}")
