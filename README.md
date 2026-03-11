# RAG System

A Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, Elasticsearch, and sentence-transformers.

## Overview

This project implements a RAG pipeline that:
- Ingests log data and stores it in Elasticsearch
- Generates embeddings using sentence-transformers
- Creates FAISS vector indexes for efficient similarity search
- Provides REST API endpoints for querying and management
- Supports batch processing with progress tracking

## Technologies Used

- **Python 3.8+**: Primary programming language
- **FastAPI 0.95.2**: Web framework for REST API
- **Elasticsearch 7.17.0**: Document store and search engine
- **sentence-transformers 2.2.2**: Embedding generation
- **FAISS 1.7.4**: Vector similarity search library
- **uvicorn**: ASGI server for FastAPI
- **PyTorch**: Deep learning framework for transformers

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch running locally or in Docker
- OpenAI API key (optional, for production use)

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Required environment variables:
- `ELASTICSEARCH_URL`: Elasticsearch connection URL
- `OPENAI_API_KEY`: OpenAI API key for production (optional)
- `FAISS_INDEX_PATH`: Path to FAISS index file (optional)

## Usage

### Running the Application

Start the FastAPI server:

```bash
uvicorn rag_system:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Running Tests

To run all test scripts:

```bash
python test/requirements-test.py
```

This executes the following individual tests:
- `tqdm-test.py`: Tests tqdm progress bar functionality
- `pytorch-test.py`: Tests PyTorch installation and basic operations
- `sentence-test.py`: Tests sentence-transformers embedding generation
- `elastic-test.py`: Tests Elasticsearch connection and basic operations
- `gc-test.py`: Tests garbage collection and memory management

### Environment Setup

Create a `.env` file in the root directory with:

```
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your_api_key_here
FAISS_INDEX_PATH=./faiss_index
```

## API Endpoints

### Health Check

```
GET /health
```

Returns system health status and component availability.

### Document Ingestion

```
POST /ingest
```

Ingests documents into the RAG pipeline.

### Search

```
GET /search?q=query
```

Searches for similar documents using vector similarity.

### Index Management

```
POST /index
```

Creates or updates the FAISS vector index.

## Architecture

### Components

1. **Log Ingestion**: Processes and stores log data in Elasticsearch
2. **Embedding Generation**: Uses sentence-transformers to create vector representations
3. **Vector Index**: FAISS index for efficient similarity search
4. **API Server**: FastAPI endpoints for interaction
5. **Scheduler**: Background tasks for periodic processing

### Data Flow

```
Input Logs → Elasticsearch → Embedding Generation → FAISS Index → Search API
```

## Testing Infrastructure

### Test Scripts

Each test script in the `test/` directory verifies a specific component:

- **elastic-test.py**: Validates Elasticsearch connection and basic operations
- **sentence-test.py**: Tests embedding generation with sentence-transformers
- **pytorch-test.py**: Verifies PyTorch installation and tensor operations
- **tqdm-test.py**: Tests progress bar functionality
- **gc-test.py**: Monitors garbage collection and memory usage

### Running Individual Tests

```bash
python test/elastic-test.py
python test/sentence-test.py
python test/pytorch-test.py
python test/tqdm-test.py
python test/gc-test.py
```

## Known Issues

1. **Elasticsearch Connection**: Tests require Elasticsearch to be running locally or in Docker container
2. **OpenAI API Key**: Required for production use but not for testing
3. **Isolated Environments**: Some test scripts use subprocess calls that may fail in isolated environments
4. **Memory Management**: Large datasets may require increased memory allocation

## Pipeline History

- 2024-01-15: Initial project setup with Elasticsearch integration and FAISS vector store
- 2024-01-16: Added sentence-transformers library for embedding generation
- 2024-01-17: Implemented log ingestion pipeline with batch processing
- 2024-01-18: Added FastAPI server for REST endpoints
- 2024-01-19: Created test scripts for individual component verification
- 2024-01-20: Added garbage collection test for memory management
- 2024-01-21: Implemented progress tracking with tqdm library
- 2024-01-22: Added environment variable configuration for production deployment

## Development Notes

### File Structure

```
.
├── rag_system.py          # Main implementation
├── requirements.txt       # Python dependencies
├── test/                  # Test scripts
│   ├── requirements-test.py  # Test runner
│   ├── elastic-test.py
│   ├── sentence-test.py
│   ├── pytorch-test.py
│   ├── tqdm-test.py
│   └── gc-test.py
├── .env                   # Environment variables
└── README.md              # This file
```

### Important Files

- **rag_system.py**: Core implementation with all RAG pipeline logic
- **test/requirements-test.py**: Test runner that executes all test scripts
- **requirements.txt**: All Python dependencies with exact versions

### Gotchas

1. Elasticsearch must be running before starting the application
2. OpenAI API key is optional but required for production embeddings
3. FAISS index path must be writable by the application
4. Memory usage can be high with large datasets
5. Test scripts require proper environment variable setup

## Docker Deployment

### Build Image

```bash
docker build -t rag-system .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e ELASTICSEARCH_URL=http://host.docker.internal:9200 \
  -e OPENAI_API_KEY=your_key \
  rag-system
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all tests
5. Submit a pull request

## License

MIT License

## Support

For issues and questions, please open an issue on the GitHub repository.
