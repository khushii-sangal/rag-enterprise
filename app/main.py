from fastapi import FastAPI
from app.services.rag_chain import get_rag_chain
from app.services.vector_store import get_retriever
from app.models.schemas import QueryRequest, QueryResponse, Source
import os

app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation API with source attribution"
)

# Initialize once at startup
rag_chain = get_rag_chain()
retriever = get_retriever()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    # Step 1: Retrieve documents
    docs = retriever.invoke(request.question)

    # Step 2: Generate answer
    answer = rag_chain.invoke(request.question)

    # Step 3: Extract unique sources
    unique_sources = set()
    sources = []

    for doc in docs:
        raw_source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        filename = os.path.basename(raw_source)
        key = (filename, page)

        if key not in unique_sources:
            unique_sources.add(key)
            sources.append(
                Source(
                    source=filename,
                    page=page
                )
            )

    return QueryResponse(
        answer=answer,
        sources=sources
    )