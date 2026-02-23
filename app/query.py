from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load vector DB
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

# Strict system prompt
prompt = ChatPromptTemplate.from_template("""
You are an enterprise assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I cannot answer this question based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

# Load LLM
llm = OllamaLLM(model="llama3")

# Format retrieved docs into single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build LCEL retrieval chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:")
    print(response)