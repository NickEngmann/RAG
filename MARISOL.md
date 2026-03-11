# MARISOL.md — Pipeline Context

## Project Overview
This is a Python 3.12-based RAG (Retrieval-Augmented Generation) system for log analysis. It uses Elasticsearch for log storage and retrieval, Sentence Transformers for generating embeddings, FAISS for vector search, and FastAPI for the web interface. The system processes log files, creates vector embeddings, and provides natural language query capabilities for log analysis.

## Build & Run
- **Language**: Python 3.12
- **Framework**: FastAPI (web interface), Elasticsearch (log storage), Sentence Transformers (embeddings), FAISS (vector search)
- **Docker image**: python:3.12-slim
- **Install deps**: `pip install -r requirements.txt`
- **Run**: `python rag_system.py` (runs both API server and scheduled processing)
- **API server only**: `uvicorn rag_system:app --host 0.0.0.0 --port 8000`

## Testing
- **Test framework**: Custom test scripts (no pytest setup found)
- **Test command**: `python test/test_runner.py` or run individual test files
- **Hardware mocks needed**: No
- **Known test issues**: Tests require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY). Some tests may fail if dependencies are not properly installed or if Elasticsearch is not running. File paths use `/mnt/vectordb/` which may not exist in all environments.

## Pipeline History
- **2024-01-15**: Initial project setup with FastAPI, Elasticsearch, and Sentence Transformers integration
- **2024-01-16**: Added FAISS vector search for efficient similarity matching
- **2024-01-17**: Implemented scheduled log processing with Python schedule library
- **2024-01-18**: Added OpenAI API integration for natural language query processing
- **2024-01-19**: Created comprehensive test suite for individual components
- **2024-01-20**: Added test_runner.py for unified test execution
- **2024-01-21**: Fixed environment variable handling with python-dotenv integration

## Known Issues
- Elasticsearch connection requires a running instance or mock
- OpenAI API key must be set in environment or .env file
- FAISS index and metadata are stored at `/mnt/vectordb/` - ensure this directory exists
- Some tests may fail in environments without proper dependencies installed

## Notes
- Main application entry point: `rag_system.py`
- Test files located in `test/` directory
- Environment variables can be set via `.env` file or system environment
- Dependencies are listed in `requirements.txt`
- Test runner is located at `test/test_runner.py`
- The system uses async/await patterns for API endpoints
- Log processing is scheduled to run at configurable intervals
- Vector embeddings are generated using sentence-transformers/all-MiniLM-L6-v2 model
- Metadata and FAISS index are persisted to disk for efficient reloading
