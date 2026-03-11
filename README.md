# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing capabilities. The system uses Elasticsearch as the primary log source, FAISS for vector similarity search, and sentence-transformers for embedding generation.

## Architecture

The system follows a RAG pattern where:
- **Elasticsearch**: Source of log data and text retrieval
- **FAISS**: Vector store for similarity search
- **sentence-transformers**: Generates embeddings for log content
- **FastAPI**: Provides REST API for the system
- **OpenAI**: Optional integration for generation tasks

## Features

- Log ingestion from Elasticsearch
- Vector embedding generation using sentence-transformers
- Similarity search using FAISS
- REST API for log retrieval and analysis
- Scheduled processing of new logs
- Progress tracking with tqdm

## Installation

### Prerequisites

- Python 3.8+
- Elasticsearch instance running
- Environment variables configured

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

3. Create a `.env` file with required variables:
```bash
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Application

Start the FastAPI server:
```bash
uvicorn rag_system:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

Build and run the Docker container:
```bash
docker build -t rag-system .
docker run -p 8000:8000 -e ELASTICSEARCH_URL=... -e OPENAI_API_KEY=... rag-system
```

### Scheduled Processing

The system includes scheduled processing that runs every hour:
```python
import schedule
import time

def job():
    process_new_logs()

schedule.every(1).hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

## Testing

### Running Tests

To run the test suite, use the following commands:

```bash
# Run individual component tests
python test/elastic-test.py
python test/tqdm-test.py
python test/pytorch-test.py
python test/sentence-test.py
python test/gc-test.py

# Run the test runner script
python test/requirements-test.py
```

### Test Components

- **elastic-test.py**: Tests Elasticsearch connectivity and version
- **tqdm-test.py**: Tests progress bar functionality
- **pytorch-test.py**: Tests PyTorch integration
- **sentence-test.py**: Tests sentence-transformers functionality
- **gc-test.py**: Tests garbage collection behavior

## API Endpoints

### Health Check
```
GET /health
```

### Log Retrieval
```
POST /retrieve
```

### System Status
```
GET /status
```

## Configuration

### Environment Variables

- `ELASTICSEARCH_URL`: Elasticsearch connection string
- `OPENAI_API_KEY`: OpenAI API key for generation tasks
- `FAISS_INDEX_PATH`: Path to FAISS index file
- `LOG_PROCESSING_INTERVAL`: Interval for scheduled processing (default: 3600 seconds)

## Known Issues

- Tests may fail if Elasticsearch is not running
- Memory usage can be high during batch processing
- Scheduled tasks may not execute if the process is interrupted

## Performance Considerations

- Use GPU acceleration for sentence-transformers if available
- Configure Elasticsearch index size appropriately
- Monitor memory usage during batch operations
- Use connection pooling for Elasticsearch

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.
