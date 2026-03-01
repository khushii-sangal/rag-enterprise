from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()
def get_retriever():
    return retriever