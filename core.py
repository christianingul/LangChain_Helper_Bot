import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pinecone
import streamlit as st
from langchain.vectorstores import Pinecone
from typing import Any, List




pinecone_secret = st.secrets.get("pinecone2")
pinecone_env = st.secrets.get("environment2")


pinecone_api_key = pinecone_secret.get("key")
pinecone_environment = pinecone_env.get("key")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment,
)


def run_llm(query: str, chat_history: List[tuple[str, Any]] = []) -> Any:
    openai_secret = st.secrets.get("openai")
    openai_api_key = openai_secret.get("key")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Pinecone.from_existing_index(
        index_name="langchain-information", embedding=embeddings
    )
    chat = ChatOpenAI(model_name="gpt-4",verbose=True, temperature=0, openai_api_key=openai_api_key)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    result = run_llm(query="What is RetrievalQA?")
    print(result)
