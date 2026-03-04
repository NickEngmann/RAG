# MARISOL.md — Pipeline Context

## Project Overview
This is a Python-based RAG (Retrieval-Augmented Generation) system for log analysis. It uses Elasticsearch for log storage, sentence-transformers for embeddings, and OpenAI for LLM-based question answering. The system processes logs hourly, creates vector embeddings, and stores them in Elasticsearch for semantic search.

## Build & Run
- **Language**: Python 3.x
- **Framework**: None (standalone scripts)
- **Docker image**: python:3.12-slim
- **Install deps**: pip install -r requirements.txt
- **Run**: python rag_system.py (main entry point)

## Testing
- **Test framework**: None (standalone test scripts in test/ directory)
- **Test command**: python test/requirements-test.py (runs all test scripts)
- **Individual tests**: python test/elastic-test.py, python test/pytorch-test.py, python test/sentence-test.py, python test/tqdm-test.py, python test/gc-test.py
- **Hardware mocks needed**: yes — requires mocking for Elasticsearch, OpenAI API, and sentence-transformers model downloads
- **Known test issues**: Tests require environment variables (ELASTICSEARCH_URL, OPENAI_API_KEY) to be set; external service dependencies prevent running tests without mocks or actual services

## Pipeline History
- Initial setup: Created standalone test scripts for dependency verification
- Test scripts verify: Elasticsearch connectivity, PyTorch installation, sentence-transformers model loading, tqdm progress bars, garbage collection
- No pytest-style tests found in repository

## Known Issues
- Tests require external services (Elasticsearch, OpenAI) — cannot run in isolation without mocks
- No unit test framework (pytest) configured
- Environment variables required for all tests
- Test scripts in test/ directory are standalone, not organized as pytest tests

## Notes
- Main entry point: rag_system.py
- Test scripts: test/elastic-test.py, test/pytorch-test.py, test/sentence-test.py, test/tqdm-test.py, test/gc-test.py
- Test runner: test/requirements-test.py (runs all individual tests)
- Dependencies: elasticsearch, sentence-transformers, openai, python-dotenv, dateutil, faiss-cpu, tqdm
- Uses .env file for environment variables
- No hardware mocks needed for basic dependency tests, but Elasticsearch/OpenAI tests require service mocks
