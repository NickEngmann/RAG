# MARISOL.md — Pipeline Context for RAG

## Pipeline History
- 2026-03-12 — Initial project setup with Python-based RAG system for log analysis
- 2026-03-12 — Added Elasticsearch integration for log retrieval and vector embeddings
- 2026-03-12 — Created custom test runner in test/requirements-test.py to execute all test scripts
- 2026-03-12 — Fixed README.md testing commands to use correct relative paths from test directory
- 2026-03-12 — Verified all test scripts (tqdm-test.py, pytorch-test.py, sentence-test.py, elastic-test.py, gc-test.py) execute correctly from test/ directory ingestion
- 2026-03-12 — Implemented FAISS vector index for similarity search
- 2026-03-12 — Added Sentence Transformers embedding model (all-MiniLM-L6-v2)
- 2026-03-12 — Integrated OpenAI GPT-3.5-turbo for natural language query processing
- 2026-03-12 — Created FastAPI RESTful interface for RAG queries
- 2026-03-12 — Added scheduled hourly log processing with schedule library
- 2026-03-12 — Implemented multi-threaded architecture for API and processing
- 2026-03-12 — Added test scripts for tqdm, PyTorch, sentence-transformers, Elasticsearch, and garbage collection
- 2026-03-12 — Created .env.test configuration for testing environment

## Last Result
- Build Status: SUCCESS
- Docker Image: python:3.12-slim
- Dependencies Installed: elasticsearch==7.17.0, sentence-transformers==2.2.2, faiss-cpu==1.7.4, numpy==1.23.5, schedule==1.2.0, tqdm==4.65.0, fastapi==0.95.2, uvicorn==0.22.0, pydantic==1.10.7, scikit-learn==1.2.2, openai==0.27.8, python-dateutil==2.8.2, python-dotenv
- Test Scripts: tqdm-test.py, pytorch-test.py, sentence-test.py, elastic-test.py, gc-test.py
- Main Entry Point: python rag_system.py
- API Endpoint: POST /rag_query at http://localhost:8000/rag_query

## Known Issues
- Requires external Elasticsearch instance with log data for full operation
- Requires OpenAI API key for natural language query processing
- Vector index stored at /mnt/vectordb/ - requires write access to this path
- Memory usage can be high during batch processing of large log volumes
- Time scaler requires all timestamps to be available for proper normalization

## Notes
- Architecture: Multi-threaded with separate threads for API server and scheduled log processing
- Embedding Model: all-MiniLM-L6-v2 (384 dimensions + 1 for timestamp = 385 total)
- FAISS Index: Uses Inner Product (IP) similarity for efficient vector search
- Log Processing: Scheduled hourly, processes new logs from Elasticsearch
- Query Parameters: text, k (number of results), start_time, end_time, hostname_pattern
- System Prompt: Strictly uses only provided text chunks, responds "I don't know" if insufficient info
- Metadata Storage: JSON file at /mnt/vectordb/metadata.json for tracking processed logs
- Progress Persistence: On interrupt, saves metadata and FAISS index for safe restart
- Test Infrastructure: Custom test runner in test/requirements-test.py executes all test scripts
- Environment Variables: ELASTICSEARCH_URL and OPENAI_API_KEY required in .env file
