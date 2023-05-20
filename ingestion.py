from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit as st
import pinecone


pinecone_secret = st.secrets.get("pinecone2")
pinecone_env = st.secrets.get("environment2")


pinecone_api_key = pinecone_secret.get("key")
pinecone_environment = pinecone_env.get("key")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment,
)


def ingest_docs() -> None:

    openai_secret = st.secrets.get("openai")
    openai_api_key = openai_secret.get("key")

    loader = ReadTheDocsLoader(
        path="langchain-docs-new/langchain.readthedocs.io/en/latest"
    )
    raw_documents = loader.load()
    # print(raw_documents)
    print(f"loaded {len(raw_documents)}documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs-new", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name="langchain-information"
    )
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
