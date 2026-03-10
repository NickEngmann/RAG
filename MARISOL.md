# MARISOL.md — Pipeline Context

## Project Overview
This is a Python-based RAG (Retrieval-Augmented Generation) system for log analysis. It uses Elasticsearch for log storage and retrieval, Sentence Transformers for creating vector embeddings of log entries, FAISS for efficient similarity search, and FastAPI for the web interface. The system supports incremental log processing with progress tracking and can handle large-scale log analysis workflows.

## Build & Run
- **Language**: Python 3.x
- **Framework**: FastAPI (web interface), Elasticsearch (log storage), Sentence Transformers (embeddings), FAISS (vector search)
- **Docker image**: python:3.12-slim
- **Install deps**: cd /workspace/repo && pip install -r requirements.txt
- **Run**: python rag_system.py (main script), uvicorn rag_system:app --host 0.0.0.0 --port 8000 (FastAPI server)

## Testing
- **Test framework**: Custom test scripts (no pytest setup found)
- **Test command**: python test/requirements-test.py (runs all individual test scripts)
- **Hardware mocks needed**: no
- **Known test issues**: Tests require specific dependencies (torch, sentence-transformers, elasticsearch) to be installed. Some tests may fail if environment variables are not set.

## Pipeline History
- 2024-01-15 — Implement: Initial implementation of RAG system features
- 2024-01-15 — Implement: Added real features for log analysis with Elasticsearch and Sentence Transformers
- 2024-01-15 — Implement: Created test infrastructure for individual component verification
- 2024-01-15 — Test: Verified FAISS integration for vector similarity search
- 2024-01-15 — Test: Confirmed FastAPI endpoints are functional

## Known Issues
- Test scripts require specific environment setup (ELASTICSEARCH_URL, etc.)
- Some tests may fail if dependencies are not properly installed
- No pytest configuration found - using custom test scripts instead
- Elasticsearch connection requires running instance or mock

## Notes
- Main entry point: rag_system.py
- Test scripts located in test/ directory
- Each test script verifies a specific dependency (pytorch, sentence-transformers, elasticsearch, etc.)
- Project uses python-dotenv for environment variable management
- FastAPI app is defined in rag_system.py as 'app' variable
- FAISS is used for efficient vector similarity search (confirmed in requirements.txt)
- Progress tracking and checkpointing implemented for long-running log processing jobs
