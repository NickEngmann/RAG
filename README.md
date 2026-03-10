# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing capabilities to provide insightful answers to queries about log data.

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
   uvicorn rag_system:app --host 0.0.0.0 --port 8000
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

The project includes integration test scripts for individual components. Note that these tests require live Elasticsearch and OpenAI services.

### Test Scripts

- `test/pytorch-test.py` - Tests PyTorch installation and GPU availability
- `test/sentence-test.py` - Tests Sentence Transformers model loading
- `test/elastic-test.py` - Tests Elasticsearch connection and queries
- `test/tqdm-test.py` - Tests progress bar functionality
- `test/gc-test.py` - Tests garbage collection behavior

### Running Tests

To run all integration tests:
```bash
python test/requirements-test.py

**Note**: These tests require live Elasticsearch and OpenAI services. They cannot run in an isolated environment without these services running.
```

To run individual tests:
```bash
python test/pytorch-test.py
python test/sentence-test.py
python test/elastic-test.py
python test/tqdm-test.py
python test/gc-test.py
```

### Test Requirements

- Live Elasticsearch instance running
- Valid OpenAI API key configured
- Sufficient system memory for embedding models

## Architecture

### Components

1. **Elasticsearch Integration**: Retrieves log data from Elasticsearch cluster
2. **Sentence Transformers**: Generates vector embeddings for log entries
3. **FAISS**: Provides fast similarity search over embeddings
4. **OpenAI API**: Generates natural language responses based on retrieved logs
5. **FastAPI**: RESTful API interface for client applications

### Data Flow

1. Logs are ingested from Elasticsearch
2. Each log entry is converted to a vector embedding
3. Embeddings are stored in FAISS index for efficient search
4. User queries are embedded and matched against FAISS index
5. Top-k relevant logs are retrieved
6. OpenAI generates natural language response based on retrieved logs

### Persistence

- FAISS index saved to `metadata/index.faiss`
- Progress state tracked via `metadata.json`
- Batch processing scheduled hourly via schedule library

## License

This project is licensed under the MIT License - see the LICENSE file for details.
