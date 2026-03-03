# MARISOL.md — Pipeline Context for RAG

## Project Overview
A Python-based RAG system for log analysis using Elasticsearch, FAISS, and OpenAI with a FastAPI interface.

## Build & Run
- **Language**: python
- **Framework**: fastapi
- **Docker image**: python:3.12-slim
- **Install deps**: `cd /workspace/repo && pip install  -r requirements.txt 2>&1 | tail -5 || true; pip install  pytest 2>&1 | tail -3`
- **Run**: (see source code)

## Testing
- **Test framework**: none
- **Test command**: `python -m pytest tests/ -v`
- **Hardware mocks needed**: no
- **Last result**: 18/18 passed

