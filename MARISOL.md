# MARISOL.md — Pipeline Context

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings (FAISS) with natural language processing using OpenAI's GPT-3.5-turbo model. The system ingests logs from Elasticsearch, creates vector embeddings using SentenceTransformers, and exposes a FastAPI endpoint for natural language queries.

## Build & Run
- **Language**: Python 3.12
- **Framework**: FastAPI
- **Docker image**: python:3.12-slim
- **Install deps**: `cd /workspace/repo && pip install -r requirements.txt 2>&1 | tail -5 || true; pip install pytest 2>&1 | tail -3`
- **Run**: `python rag_system.py` (starts log processing and API server)

## Testing
- **Test framework**: pytest
- **Test command**: `python -m pytest tests/ -v`
- **Hardware mocks needed**: No
- **Known test issues**: None documented

## Pipeline History
- Initial pipeline run: Created MARISOL.md with project context

## Known Issues
- Requires Elasticsearch and OpenAI API credentials via environment variables
- Vector index stored at `/mnt/vectordb/vector_index.faiss` (requires persistent storage)
- Metadata stored at `/mnt/vectordb/metadata.json` (requires persistent storage)

## Notes
- The system uses a hybrid approach: text embeddings from SentenceTransformer('all-MiniLM-L6-v2') combined with normalized timestamps for time-aware retrieval
- Logs are processed in batches of 1000 with threading for producer/consumer pattern
- The API endpoint `/rag_query` accepts text queries with optional time range and hostname pattern filters
- Scheduled to process new logs every hour via the schedule library