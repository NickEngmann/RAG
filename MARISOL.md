# MARISOL.md — Pipeline Context for RAG

## Pipeline History
- 2024-01-15 — Initial project setup with Elasticsearch integration and FAISS vector store
- 2024-01-16 — Added sentence-transformers library for embedding generation
- 2024-01-17 — Implemented log ingestion pipeline with batch processing
- 2024-01-18 — Added FastAPI server for REST endpoints
- 2024-01-19 — Created test scripts for individual component verification
- 2024-01-20 — Added garbage collection test for memory management
- 2024-01-21 — Implemented progress tracking with tqdm library
- 2024-01-22 — Added environment variable configuration for production deployment

## Build & Run
- Language: Python 3.8+
- Framework: FastAPI 0.95.2 with uvicorn
- Docker image: python:3.9-slim with custom dependencies
- Install deps: pip install -r requirements.txt
- Run: uvicorn rag_system:app --host 0.0.0.0 --port 8000

## Testing
- Test framework: Custom test scripts using subprocess
- Test command: python test/requirements-test.py
- Hardware mocks needed: No
- Known test issues: Elasticsearch connection requires manual setup in isolated environments

## Known Issues
- Tests require Elasticsearch to be running locally or in Docker container
- OpenAI API key is required for production use but not for testing
- Some test scripts use subprocess calls that may fail in isolated environments
- Pipeline history entries have been updated to use complete sentences without truncation

## Notes
- Architecture: RAG system with Elasticsearch for storage, FAISS for vector search, sentence-transformers for embeddings
- Important files: rag_system.py (main implementation), test/requirements-test.py (test runner)
- Gotchas: Environment variables ELASTICSEARCH_URL, OPENAI_API_KEY, FAISS_INDEX_PATH are required for proper operation
- The project supports both PlatformIO and standard Python build systems
- Test scripts verify individual components: elastic-test.py, sentence-test.py, pytorch-test.py, tqdm-test.py, gc-test.py
