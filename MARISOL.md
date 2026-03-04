# MARISOL.md — Pipeline Context

## Project Overview
This is a Python-based RAG (Retrieval-Augmented Generation) system for log analysis that combines Elasticsearch for log storage, Sentence Transformers for embeddings, and FAISS for vector search. The system processes logs, creates embeddings, and enables natural language queries over log data. It uses FastAPI as a dependency but the main entry point is rag_system.py, not a server.

## Build & Run
- **Language**: Python 3.10+
- **Framework**: None (library/script) - FastAPI is a dependency but not used as the main framework
- **Docker image**: python:3.12-slim
- **Install deps**: pip install -r requirements.txt
- **Run**: python rag_system.py (main script) or python test/requirements-test.py (test runner)

## Testing
- **Test framework**: pytest (listed in requirements.txt)
- **Test command**: python -m pytest tests/ -v OR python test/requirements-test.py (runs integration test scripts)
- **Hardware mocks needed**: yes - tests require live Elasticsearch and OpenAI services. Mocks needed for: elasticsearch connection, openai API calls, faiss index operations
- **Known test issues**: Integration tests in test/ directory (elastic-test.py, pytorch-test.py, sentence-test.py, tqdm-test.py, gc-test.py) will fail without external services (Elasticsearch, OpenAI API). These are not unit tests and require live infrastructure.

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
