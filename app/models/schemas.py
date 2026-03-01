from pydantic import BaseModel
from typing import List


class Source(BaseModel):
    source: str
    page: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]