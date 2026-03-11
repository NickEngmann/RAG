# MARISOL.md — Pipeline Context for RAG

## Pipeline History
- *2024-01-15* — Initial project setup with Elasticsearch integration and FAISS vector store
- *2024-01-16* — Added sentence-transformers for embedding generation
- *2024-01-17* — Implemented FastAPI server with uvicorn
- *2024-01-18* — Added test scripts for individual components (elastic, tqdm, pytorch, sentence, gc)
- *2024-01-19* — Tests failed after implementation attempts (2/2)

## Known Issues
- Tests failed after implementation attempts (attempt 1/2 and attempt 2/2 both failed)
- Test infrastructure requires manual setup of Elasticsearch connection
- Some test scripts use subprocess calls that may fail in isolated environments
- Pipeline history entries need to be complete sentences without truncation

## Notes
- **Architecture**: RAG system using Elasticsearch as log source, FAISS for vector storage, sentence-transformers for embeddings
- **Dependencies**: elasticsearch==7.17.0, sentence-transformers==2.2.2, faiss-cpu==1.7.4, numpy==1.23.5, schedule==1.2.0, tqdm==4.65.0, fastapi==0.95.2, uvicorn==0.22.0, pydantic==1.10.7, scikit-learn==1.2.2, openai==0.27.8, python-dateutil==2.8.2, python-dotenv
- **Test Structure**: Individual test scripts in test/ directory (elastic-test.py, tqdm-test.py, pytorch-test.py, sentence-test.py, gc-test.py) plus requirements-test.py runner
- **Threading**: Uses threading for concurrent API server and scheduling
- **Schedule**: Processes new logs every hour via schedule.every(1).hour.do(process_new_logs)
- **Docker**: Build with docker build -t rag-system . and run with docker run -p 8000:8000 -e ELASTICSEARCH_URL=... -e OPENAI_API_KEY=... rag-system
