# app/main.py

import time
from fastapi import FastAPI, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.services.vector_store import get_vectorstore
from app.services.rag_chain import get_rag_chain
from app.core.logging_config import logger

app = FastAPI()

vectordb = get_vectorstore()
rag_chain = get_rag_chain()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    start_time = time.time()

    try:
        logger.info(f"Query received: {request.question}")

        # 1️⃣ Similarity search with scores
        retrieval_start = time.time()

        docs_with_scores = vectordb.similarity_search_with_score(
            request.question,
            k=4
        )

        retrieval_time = time.time() - retrieval_start

        # 2️⃣ Smart filtering with fallback
        score_threshold = 1.5  # safe default

        filtered_docs = [
            doc for doc, score in docs_with_scores
            if score < score_threshold
        ]

        if not filtered_docs:
            logger.warning("No docs passed threshold. Using fallback top-k.")
            filtered_docs = [doc for doc, _ in docs_with_scores]

        docs = filtered_docs

        # 3️⃣ Generate answer using filtered docs
        llm_start = time.time()

        answer = rag_chain.invoke(request.question)

        llm_time = time.time() - llm_start
        total_time = time.time() - start_time

        logger.info(f"Retrieval time: {retrieval_time:.2f}s")
        logger.info(f"LLM time: {llm_time:.2f}s")
        logger.info(f"Total request time: {total_time:.2f}s")

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    # 4️⃣ Extract unique sources
    unique_sources = set()
    sources = []

    for doc in docs:
        raw_source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        key = (raw_source, page)

        if key not in unique_sources:
            unique_sources.add(key)
            sources.append(
                Source(
                    source=raw_source,
                    page=page
                )
            )

    return QueryResponse(
        answer=answer,
        sources=sources
    )