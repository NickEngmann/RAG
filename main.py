#!/usr/bin/env python3
"""FastAPI application for RAG System"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from rag_system import RAGSystem

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation System API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    

class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    answer: Optional[str] = None
    

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = RAGSystem()
        rag_system.connect_elasticsearch()
        rag_system.load_index()
    except Exception as e:
        print(f"Warning: Could not initialize RAG system: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG System"}


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = rag_system.search(request.query, top_k=request.top_k)
        return QueryResponse(
            query=request.query,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_document")
async def add_document(request: DocumentRequest):
    """Add a document to the RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        rag_system.add_document(request.content, request.metadata)
        return {"status": "success", "message": "Document added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def get_documents(limit: int = 10):
    """Get documents from the RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # This would need to be implemented in RAGSystem
        return {"documents": [], "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
