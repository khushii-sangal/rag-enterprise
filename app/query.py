from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load vector DB
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

# Strict system prompt
template = """
You are an enterprise assistant.

You must answer ONLY using the provided context.
If the answer is not in the context, say:
"I cannot answer this question based on the provided documents."

Context:
{context}

Question:
{input}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# Load LLM
llm = OllamaLLM(model="llama3")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break

    response = retrieval_chain.invoke({"input": query})
    print("\nAnswer:")
    print(response["answer"])