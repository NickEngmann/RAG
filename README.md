# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing capabilities. The system is designed to help developers and operations teams analyze large volumes of log data more effectively.

## Features

- **Vector Embeddings**: Uses Sentence Transformers to create semantic embeddings of log entries
- **Similarity Search**: FAISS-based vector search for finding similar log patterns
- **Elasticsearch Integration**: Stores and indexes log entries for fast retrieval
- **FastAPI Web Interface**: RESTful API for programmatic access
- **Incremental Processing**: Supports batch and streaming log ingestion
- **Progress Tracking**: Monitors processing progress with tqdm
- **Checkpointing**: Saves state for long-running jobs

## Architecture

The system consists of several key components:

1. **Log Ingestion Module**: Handles incoming log data and stores in Elasticsearch
2. **Embedding Generator**: Creates vector embeddings using Sentence Transformers
3. **Similarity Search**: Uses FAISS for efficient vector similarity queries
4. **API Layer**: FastAPI endpoints for interaction with the system
5. **Progress Tracker**: Monitors and reports processing status

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch instance (local or remote)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NickEngmann/RAG.git
cd RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with the following variables:
```
ELASTICSEARCH_URL=http://localhost:9200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index
LOG_DATA_PATH=./data/logs
```

4. Initialize the system:
```bash
python rag_system.py --init
```

## Usage

### Running the Main Script

Start the log analysis system:
```bash
python rag_system.py
```

### Using the FastAPI Server

Start the web interface:
```bash
uvicorn rag_system:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /embed

Create vector embeddings for log text.

**Request Body**:
```json
{
  "text": "Your log text here"
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, ...],
  "text": "Your log text here"
}
```

#### POST /search

Search for similar log entries using vector similarity.

**Request Body**:
```json
{
  "query_text": "error message",
  "top_k": 5
}
```

**Response**:
```json
{
  "results": [
    {
      "log_id": "abc123",
      "text": "Error: Connection timeout",
      "similarity": 0.95
    },
    {
      "log_id": "def456",
      "text": "Timeout connecting to database",
      "similarity": 0.87
    }
  ]
}
```

#### POST /ingest

Ingest new log entries into the system.

**Request Body**:
```json
{
  "logs": [
    {
      "id": "log001",
      "text": "Application started successfully",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "id": "log002",
      "text": "Database connection established",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ]
}
```

**Response**:
```json
{
  "ingested": 2,
  "status": "success"
}
```

## Testing

### Running Tests

The project uses custom test scripts located in the `test/` directory:

```bash
python test/requirements-test.py
```

This will run all individual test scripts:
- `tqdm-test.py` - Progress bar functionality
- `pytorch-test.py` - PyTorch integration
- `sentence-test.py` - Sentence Transformers
- `elastic-test.py` - Elasticsearch connection
- `gc-test.py` - Garbage collection handling

### Test Requirements

Each test script verifies specific dependencies and functionality. Make sure all required packages are installed before running tests.

## Configuration

### Environment Variables

The system uses python-dotenv for environment variable management. Key variables:

- `ELASTICSEARCH_URL` - Elasticsearch connection string
- `EMBEDDING_MODEL` - Sentence Transformers model to use
- `FAISS_INDEX_PATH` - Path to FAISS index file
- `LOG_DATA_PATH` - Directory for log data storage

### Model Selection

Supported embedding models:
- `sentence-transformers/all-MiniLM-L6-v2` (default)
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-MiniLM-L6-v2`

## Project Structure

```
RAG/
├── rag_system.py          # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── MARISOL.md            # Pipeline context documentation
├── test/                 # Test scripts
│   ├── requirements-test.py
│   ├── tqdm-test.py
│   ├── pytorch-test.py
│   ├── sentence-test.py
│   ├── elastic-test.py
│   └── gc-test.py
├── .env.test             # Test environment configuration
└── .gitignore            # Git ignore rules
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Known Issues

- Elasticsearch connection requires a running instance or mock
- Some tests may fail if environment variables are not properly set
- Large log datasets may require additional memory for embedding generation

## Future Enhancements

- Support for multiple Elasticsearch clusters
- Async log processing pipeline
- Enhanced error handling and logging
- Docker containerization support
- Integration with additional vector databases
