from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load embeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Load vector database
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

# Strong system prompt (NO hallucination)
template = """
You are an enterprise assistant.

You must answer ONLY using the provided context.
If the answer is not in the context, say:
"I cannot answer this question based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Load LLM
llm = OllamaLLM(model="llama3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break

    response = qa_chain.run(query)
    print("\nAnswer:")
    print(response)