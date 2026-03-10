# MARISOL.md — Pipeline Context

## Project Overview
This is a Python 3.12-based RAG (Retrieval-Augmented Generation) system for log analysis. It uses Elasticsearch for log storage and retrieval, Sentence Transformers for embedding generation, FAISS for vector similarity search, and OpenAI's GPT-3.5 for generating responses. The system processes logs from Elasticsearch, creates vector embeddings, and provides a FastAPI endpoint for semantic search queries with LLM-generated answers.

## Build & Run
- **Language**: Python 3.12
- **Framework**: FastAPI (web interface), Elasticsearch (log storage), Sentence Transformers (embeddings), FAISS (vector search)
- **Docker image**: python:3.12-slim
- **Install deps**: cd /workspace/repo && pip install -r requirements.txt
- **Run**: python rag_system.py (main script), uvicorn rag_system:app --host 0.0.0.0 --port 8000 (FastAPI server only)

## Testing
- **Test framework**: Custom test scripts (no pytest setup found)
- **Test command**: Run individual test files in test/ directory (e.g., python test/elastic-test.py, python test/pytorch-test.py)
- **Hardware mocks needed**: no
- **Known test issues**: Tests require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY). Some tests may fail if dependencies are not properly installed or if Elasticsearch is not running. The test-runner.py file mentioned in documentation does not exist - use individual test scripts instead.

## Pipeline History
2024-01-15: Initial project setup with Python 3.12, FastAPI, Elasticsearch, and FAISS integration
2024-01-16: Added sentence-transformers for embedding generation and OpenAI integration
2024-01-17: Implemented scheduled log processing with schedule library
2024-01-18: Added FastAPI endpoint for RAG queries
2024-01-19: Created individual test scripts for component verification

## Known Issues
- Test scripts require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY)
- No pytest configuration found - using custom test scripts instead
- Elasticsearch connection requires running instance or mock
- File paths use /mnt/vectordb/ which may not exist in all environments
- The test-runner.py file does not exist - use individual test scripts in test/ directory
- OpenAI API key must be set in environment or .env file
- FAISS index and metadata are stored at /mnt/vectordb/ - ensure this directory exists

## Notes
- Vector index stored at: /mnt/vectordb/vector_index.faiss
- Metadata stored at: /mnt/vectordb/metadata.json
- FastAPI endpoint: POST /rag_query for semantic search with LLM response generation
- Main entry point: rag_system.py (runs both API server and scheduled processing)
- Dependencies: elasticsearch, sentence-transformers, faiss-cpu, fastapi, uvicorn, openai, schedule, tqdm, pydantic, scikit-learn
- Environment variables required: ELASTICSEARCH_URL, OPENAI_API_KEY
- Uses all-MiniLM-L6-v2 model for sentence embeddings (768 dimensions + 1 for timestamp)
