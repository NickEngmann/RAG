# MARISOL.md — Pipeline Context

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system for log analysis that combines Elasticsearch for log storage, sentence-transformers for embedding generation, FAISS for vector similarity search, and OpenAI for natural language processing. The system provides a FastAPI interface for querying logs and includes a main script (rag_system.py) for indexing logs and managing the pipeline.

## Build & Run
- **Language**: Python 3.x
- **Framework**: FastAPI (embedded in rag_system.py)
- **Docker image**: python:3.12-slim
- **Install deps**: pip install -r requirements.txt
- **Run**: python rag_system.py (runs indexing and starts API server on port 8000)

## Testing
- **Test framework**: pytest (installed separately)
- **Test command**: python -m pytest tests/ -v (no formal test suite; standalone scripts in test/ directory)
- **Hardware mocks needed**: no
- **Known test issues**: No formal pytest test suite found; only individual test scripts in test/ directory (elastic-test.py, sentence-test.py, tqdm-test.py, pytorch-test.py, gc-test.py, requirements-test.py)

## Pipeline History
No pipeline history available - this is initial documentation.

## Known Issues
- No formal test suite in tests/ directory
- Individual test scripts exist in test/ but are not integrated into pytest
- rag_system.py requires environment variables (OPENAI_API_KEY, ELASTICSEARCH_URL, etc.) to be set via .env file

## Notes
- Main entry point: rag_system.py (standalone script for indexing and querying)
- Dependencies include: elasticsearch, sentence-transformers, faiss-cpu, fastapi, uvicorn, openai
- Uses Elasticsearch for log storage and FAISS for vector similarity search
- Requires .env file with OPENAI_API_KEY, ELASTICSEARCH_URL, and other configuration
- Test scripts in test/ directory are standalone and not part of a formal test suite
