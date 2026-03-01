# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing using OpenAI's GPT-3.5-turbo model. The system ingests logs from Elasticsearch, creates vector embeddings using SentenceTransformers, and exposes a FastAPI endpoint for natural language queries.

## Features

- Real-time log ingestion from Elasticsearch
- Vector embeddings using SentenceTransformer('all-MiniLM-L6-v2')
- Time-aware retrieval with FAISS index
- FastAPI endpoint for natural language queries
- Scheduled log processing (every hour)
- Hostname and time range filtering

## Prerequisites

- Python 3.12+
- Elasticsearch instance
- OpenAI API key
- Persistent storage for vector index and metadata

## Installation

1. Clone the repository:
```bash
cd /workspace/repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Elasticsearch URL and OpenAI API key
```

4. Create persistent storage directory:
```bash
mkdir -p /mnt/vectordb
```

## Usage

### Running the System

```bash
python rag_system.py
```

This will:
1. Process existing logs from Elasticsearch
2. Start the FastAPI server on port 8000
3. Schedule hourly log processing

### API Endpoint

The system exposes a `/rag_query` endpoint:

```bash
curl -X POST http://localhost:8000/rag_query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "find errors related to database connection",
    "k": 5,
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-01-02T00:00:00Z",
    "hostname_pattern": "server-*"
  }'
```

### Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Configuration

- **Vector Index**: Stored at `/mnt/vectordb/vector_index.faiss`
- **Metadata**: Stored at `/mnt/vectordb/metadata.json`
- **Log Processing**: Processes logs since last processed timestamp
- **Batch Size**: 1000 logs per batch

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.