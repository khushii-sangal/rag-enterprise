from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vector_store import get_retriever

llm = OllamaLLM(model="llama3")

prompt = ChatPromptTemplate.from_template("""
You are an enterprise assistant.
Answer ONLY using the provided context.
{context}
Question: {question}
""")

def get_rag_chain():
    retriever = get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain