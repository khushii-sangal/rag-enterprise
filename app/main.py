from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

# ---- Load Models Once ----

embedding = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

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

llm = OllamaLLM(model="llama3")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):

    async def generate():
        # Invoke normally
        answer = rag_chain.invoke(request.question)

        # Simulate token streaming
        for word in answer.split():
            yield word + " "
            await asyncio.sleep(0.02)  # small delay for streaming effect

    return StreamingResponse(generate(), media_type="text/plain")