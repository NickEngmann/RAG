# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing to provide semantic search capabilities.

## Python Version

This project requires **Python 3.12** or later. Ensure you have the correct version installed before proceeding with setup. It combines efficient log retrieval using vector embeddings with natural language processing capabilities to provide insightful answers to queries about log data.

## Features

- Efficient log ingestion from Elasticsearch
- Vector embedding creation using Sentence Transformers
- Fast similarity search using FAISS
- Time-aware and hostname-aware querying
- Natural language query processing using OpenAI's GPT model
- RESTful API for easy integration

## Prerequisites

- Python 3.8+
- Elasticsearch instance with log data
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/log-analysis-rag.git
   cd log-analysis-rag
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables in an .env file:
   ```
   ELASTICSEARCH_URL=http://your_elasticsearch_ip:9200
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the RAG system:
   ```
   python rag_system.py
   ```

2. The system will begin processing logs from Elasticsearch and start the API server.

3. To query the system, send a POST request to `http://localhost:8000/rag_query` with a JSON body:
   ```json
   {
     "text": "What are the most common errors?",
     "k": 5,
     "start_time": "2024-08-01T00:00:00Z",
     "end_time": "2024-08-09T00:00:00Z",
     "hostname_pattern": "web-server-*"
   }
   ```

4. The system will return a JSON response with the generated answer and relevant log entries.

## Configuration

- Adjust the `batch_size` in `process_new_logs()` to control memory usage during log processing.
- Modify the `schedule.every(1).hour.do(process_new_logs)` line to change how often new logs are processed.
- Update the `dimension` variable if you change the embedding model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing

The project includes individual test scripts to verify component functionality:

### Running All Tests
```bash
cd /workspace/repo && python test/requirements-test.py
```

### Running Individual Tests
- **tqdm-test.py**: Tests tqdm progress bar functionality
- **pytorch-test.py**: Tests PyTorch installation and basic operations
- **sentence-test.py**: Tests sentence-transformers embedding generation
- **elastic-test.py**: Tests Elasticsearch connection (requires ELASTICSEARCH_URL env var)
- **gc-test.py**: Tests garbage collector functionality

### Test Requirements
- PyTorch must be installed
- sentence-transformers must be installed (note: may have compatibility issues with newer huggingface_hub versions)
- Elasticsearch connection requires ELASTICSEARCH_URL environment variable
- tqdm and gc tests have no external dependencies

## Architecture

### Components

1. **Elasticsearch Integration**: Retrieves log data from Elasticsearch cluster
2. **Sentence Transformers**: Generates vector embeddings for log entries using models like `all-MiniLM-L6-v2`
3. **FAISS**: Efficient similarity search over vector embeddings
4. **FastAPI**: RESTful API for querying the RAG system
5. **OpenAI Integration**: Generates natural language responses based on retrieved logs

### Data Flow

1. Logs are ingested from Elasticsearch
2. Each log entry is converted to a vector embedding using Sentence Transformers
3. Embeddings are stored in FAISS index for efficient similarity search
4. User queries are processed to find relevant log entries
5. Relevant logs are sent to OpenAI for natural language response generation
6. API returns JSON response with answer and relevant log entries

### Key Files

- `rag_system.py`: Main application script with FastAPI server and log processing logic
- `test/requirements-test.py`: Test runner for all individual test scripts
- `test/*.py`: Individual component test scripts
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (ELASTICSEARCH_URL, OPENAI_API_KEY)

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**: Ensure ELASTICSEARCH_URL is set correctly and Elasticsearch is running
2. **Sentence Transformers Import Error**: May need to downgrade huggingface_hub to version < 0.20.0
3. **OpenAI API Errors**: Verify OPENAI_API_KEY is set and has valid credentials
4. **Memory Issues**: Adjust batch_size in process_new_logs() method

### Environment Setup

Create a `.env` file in the project root with:
```
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your_api_key_here
```

## Testing

The project uses custom test scripts to verify individual component functionality. Run all tests using:

```bash
python test/test-runner.py
```

Individual test scripts:
- `test/tqdm-test.py` - Verifies tqdm progress bar functionality
- `test/pytorch-test.py` - Verifies PyTorch installation and GPU availability
- `test/sentence-test.py` - Verifies sentence-transformers model loading
- `test/elastic-test.py` - Verifies Elasticsearch connection
- `test/gc-test.py` - Verifies garbage collection behavior

Environment variables required for tests:
- `ELASTICSEARCH_URL` - Elasticsearch connection string (default: http://localhost:9200)
- `OPENAI_API_KEY` - OpenAI API key for LLM integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
