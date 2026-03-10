# MARISOL.md — Pipeline Context for RAG

## Project Overview
This is a Python-based RAG (Retrieval-Augmented Generation) system for log analysis that combines Elasticsearch for log storage, sentence-transformers for embeddings, and OpenAI for natural language processing. The system provides a FastAPI server for querying logs and batch processing capabilities.

## Build & Run
- **Language**: Python 3.x
- **Framework**: FastAPI
- **Docker image**: python:3.12-slim
- **Install deps**: `pip install -r requirements.txt`
- **Run**: `uvicorn rag_system:app --host 0.0.0.0 --port 8000`

## Testing
- **Test framework**: Custom integration test scripts (not pytest)
- **Test command**: `python test/requirements-test.py`
- **Hardware mocks needed**: no
- **Known test issues**: Tests require live Elasticsearch and OpenAI services - cannot run in isolated environment

## Pipeline History
- 2024-01-15: Initial project setup with FastAPI integration
- 2024-01-20: Added batch processing with schedule library
- 2024-01-25: Implemented progress tracking via metadata.json
- 2024-02-01: Added comprehensive test scripts for individual components

## Known Issues
- Test scripts require live Elasticsearch and OpenAI services - cannot run in isolated environment
- Progress saving uses metadata.json which may have file permission issues
- No pytest test suite exists - only integration test scripts in test/ directory

## Notes
- rag_system.py requires environment variables: ELASTICSEARCH_URL, OPENAI_API_KEY
- Entry point is FastAPI app defined in rag_system.py, run via uvicorn
- Individual component tests: pytorch-test.py, sentence-test.py, elastic-test.py, tqdm-test.py, gc-test.py
- Test runner: test/requirements-test.py executes all component tests
- Progress state tracked via metadata.json
- Batch processing scheduled hourly via schedule library
