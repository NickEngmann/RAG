# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing to provide intelligent log search and analysis capabilities.

## Architecture

The system consists of the following components:

- **Elasticsearch**: Log storage and retrieval
- **Sentence Transformers**: Generates embeddings for log entries
- **FAISS**: Vector similarity search for efficient retrieval
- **FastAPI**: Web interface for querying the RAG system
- **OpenAI GPT-3.5**: Generates contextual responses based on retrieved logs

## Features

- **Log Processing**: Automatically processes logs from Elasticsearch and creates vector embeddings
- **Semantic Search**: Search logs using natural language queries
- **Scheduled Processing**: Background jobs to continuously process new logs
- **API Endpoints**: RESTful API for programmatic access to the RAG system
- **Progress Tracking**: Visual progress indicators during processing using tqdm
- **Memory Management**: Efficient garbage collection for large datasets

## Quick Start

### Prerequisites

- Python 3.12+
- Elasticsearch instance running
- OpenAI API key

### Installation

```bash
cd /workspace/repo
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with the following variables:

```
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your_api_key_here
```

### Running the System

**Main Script (includes API server and scheduled processing):**
```bash
python rag_system.py
```

**API Server Only:**
```bash
uvicorn rag_system:app --host 0.0.0.0 --port 8000
```

### Testing Components

Run individual test scripts to verify each component:

```bash
# Test tqdm progress bars
python test/tqdm-test.py

# Test PyTorch installation
python test/pytorch-test.py

# Test sentence-transformers
python test/sentence-test.py

# Test Elasticsearch connection
python test/elastic-test.py

# Test garbage collection
python test/gc-test.py
```

Or run all tests using the test runner:
```bash
python test/test_runner.py
```

## API Endpoints

### POST /rag_query

Perform semantic search on logs with LLM-generated responses.

**Request Body:**
```json
{
    "query": "error messages from last hour",
    "top_k": 5
}
```

**Response:**
```json
{
    "results": [...],
    "response": "Generated response based on logs"
}
```

## File Structure

```
/workspace/repo/
├── rag_system.py          # Main application and API server
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── MARISOL.md            # Pipeline context and documentation
├── .env                  # Environment variables
├── .env.test             # Test environment variables
├── test/                 # Test scripts
│   ├── test_runner.py    # Test runner script
│   ├── tqdm-test.py      # tqdm progress bar test
│   ├── pytorch-test.py   # PyTorch test
│   ├── sentence-test.py  # Sentence transformers test
│   ├── elastic-test.py   # Elasticsearch test
│   └── gc-test.py        # Garbage collection test
└── /mnt/vectordb/        # Vector database storage
    ├── vector_index.faiss
    └── metadata.json
```

## Dependencies

- **elasticsearch**: Log storage and retrieval
- **sentence-transformers**: Embedding generation
- **faiss-cpu**: Vector similarity search
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **openai**: LLM integration
- **schedule**: Background job scheduling
- **tqdm**: Progress bars
- **pydantic**: Data validation
- **scikit-learn**: Data processing utilities
- **python-dateutil**: Date parsing

## Known Issues

- Test scripts require specific environment setup (ELASTICSEARCH_URL, OPENAI_API_KEY)
- Elasticsearch connection requires a running instance or mock
- File paths use `/mnt/vectordb/` which may not exist in all environments
- OpenAI API key must be set in environment or `.env` file
- FAISS index and metadata are stored at `/mnt/vectordb/` - ensure this directory exists

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
