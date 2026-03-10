# MARISOL.md — Pipeline Context

## Project Overview
This is a Python 3.12-based RAG (Retrieval-Augmented Generation) system for log analysis. It uses Elasticsearch for log storage and retrieval, Sentence Transformers for embeddings, FAISS for vector search, and FastAPI for the web interface. The system processes log entries, creates vector embeddings with timestamp normalization, and provides semantic search capabilities with LLM-powered responses.

## Build & Run
- **Language**: Python 3.12
- **Framework**: FastAPI (web interface), Elasticsearch (log storage), Sentence Transformers (embeddings), FAISS (vector search)
- **Docker image**: python:3.12-slim
- **Install deps**: cd /workspace/repo && pip install -r requirements.txt
- **Run**: python rag_system.py (main script), uvicorn rag_system:app --host 0.0.0.0 --port 8000 (FastAPI server only)

## Testing
- **Test framework**: Custom test scripts (no pytest setup found)
- **Test command**: python test/test-runner.py (runs all individual test scripts)
- **Hardware mocks needed**: no
- **Known test issues**: Tests require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY). Some tests may fail if dependencies are not properly installed or if Elasticsearch is not running.

## Pipeline History
- 2024-01-15: Initial project setup with RAG system components
- 2024-01-16: Added test scripts for individual components (tqdm, pytorch, sentence-transformers, elasticsearch, gc)
- 2024-01-17: Created test-runner.py to consolidate test execution
- 2024-01-18: Fixed test runner naming convention (requirements-test.py -> test-runner.py)

## Known Issues
- Test scripts require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY)
- Some tests may fail if dependencies are not properly installed
- No pytest configuration found - using custom test scripts instead
- Elasticsearch connection requires running instance or mock
- File paths use /mnt/vectordb/ which may not exist in all environments

## Notes
- Main application entry point: rag_system.py
- Test scripts: test/tqdm-test.py, test/pytorch-test.py, test/sentence-test.py, test/elastic-test.py, test/gc-test.py
- Test runner: test/test-runner.py (consolidated test execution)
- Dependencies: elasticsearch, sentence-transformers, faiss-cpu, fastapi, uvicorn, openai, tqdm, schedule
- Environment variables required: ELASTICSEARCH_URL, OPENAI_API_KEY
- Vector index stored at: /mnt/vectordb/vector_index.faiss
- Metadata stored at: /mnt/vectordb/metadata.json
- FastAPI endpoint: POST /rag_query for semantic search with LLM response generation
