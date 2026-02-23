from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load existing Chroma DB
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

# Load LLM
llm = Ollama(model="llama3")

# Define prompt template
template = """
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask question
query = input("Ask a question: ")
response = rag_chain.invoke(query)

print("\nAnswer:")
print(response)