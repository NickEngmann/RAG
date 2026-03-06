# MARISOL.md — Pipeline Context for RAG

## Project Overview
This is a Python-based RAG (Retrieval-Augmented Generation) system for log analysis that combines Elasticsearch for log storage, Sentence Transformers for embeddings, and FAISS for vector search. The system processes logs, creates embeddings, and enables natural language queries over log data. It uses FastAPI as a dependency but the main entry point is rag_system.py, not a server.


## Build & Run
- **Language**: python
- **Framework**: fastapi
- **Docker image**: python:3.12-slim
- **Install deps**: `cd /workspace/repo && pip install  -r requirements.txt 2>&1 | tail -5 || true; pip install  pytest 2>&1 | tail -3`
- **Run**: (see source code)

## Testing
- **Test framework**: pytest
- **Test command**: `python -m pytest tests/ -v`
- **Hardware mocks needed**: no
- **Last result**: 8/8 passed

## Pipeline History
- Initial setup: Created rag_system.py with log processing pipeline
- Added test scripts for individual component verification
- Dependencies updated to include fastapi, uvicorn, pydantic
- FAISS index persistence implemented with progress saving


## Known Issues
- Test scripts require live Elasticsearch and OpenAI services - cannot run in isolated environment
- No pytest test suite exists in tests/ directory - only integration test scripts in test/
- rag_system.py requires environment variables: ELASTICSEARCH_URL, OPENAI_API_KEY
- Progress saving uses metadata.json which may have file permission issues


## Notes
- Main entry point is rag_system.py (log processing script), not a FastAPI server
- Test scripts in test/ directory are integration tests for individual components
- FAISS index is saved to metadata/index.faiss for persistence
- Batch processing scheduled hourly via schedule library
- Uses dotenv for environment variable management

