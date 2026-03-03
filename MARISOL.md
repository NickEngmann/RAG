# MARISOL.md — Pipeline Context

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system for log analysis that combines Elasticsearch for log storage, FAISS for vector similarity search, sentence-transformers for embeddings, and OpenAI GPT-3.5 for generating responses. The system processes logs from Elasticsearch, creates vector embeddings with timestamps, and provides a FastAPI interface for querying logs and generating AI-powered responses.

## Build & Run
- **Language**: Python 3.x
- **Framework**: FastAPI with uvicorn
- **Docker image**: python:3.12-slim
- **Install deps**: pip install -r requirements.txt
- **Run**: python rag_system.py (runs initial log processing, starts API server on port 8000, and schedules hourly log processing)

## Testing
- **Test framework**: Custom test scripts in test/ directory (no pytest setup found)
- **Test command**: python test/requirements-test.py (runs all test scripts)
- **Hardware mocks needed**: no
- **Known test issues**: 
  - requirements-test.py has a bug: it runs individual test scripts from wrong directory (looks for scripts in root instead of test/)
  - To run individual tests: python test/tqdm-test.py, python test/pytorch-test.py, etc.
  - Tests require external dependencies (Elasticsearch, OpenAI API key) to be configured via environment variables

## Pipeline History
No pipeline history available - this is initial documentation.

## Known Issues
- Requires Elasticsearch instance running at configured URL
- Requires OpenAI API key configured in environment
- Uses /mnt/vectordb/ for persistent storage (FAISS index and metadata)
- Test scripts are custom subprocess runners, not pytest-based

## Notes
- Main entry point: rag_system.py
- API endpoint: POST /rag_query
- Environment variables needed: ELASTICSEARCH_URL, OPENAI_API_KEY
- Uses threading for concurrent log processing and API serving
- Schedule library for hourly log processing
- FAISS index stored at /mnt/vectordb/vector_index.faiss
- Metadata stored at /mnt/vectordb/metadata.json
