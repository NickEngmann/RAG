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

### Full System (Indexing + API Server)

Run the complete system which processes logs and starts the API server:
   ```
   python rag_system.py
   ```

This script will:
- Process logs from Elasticsearch and create vector embeddings (indexing phase)
- Start the FastAPI server on port 8000
- Run scheduled log processing every hour

Note: The indexing phase runs on startup. For production use, you may want to run indexing separately from the API server.

### Standalone API Server Only

If you only need the API server without log processing:
   ```python
   from rag_system import app
   import uvicorn

   uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

### Querying the API

Once the API server is running, send POST requests to `http://localhost:8000/rag_query`:
   ```bash
   curl -X POST http://localhost:8000/rag_query \
     -H "Content-Type: application/json" \
     -d '{
       "text": "What are the most common errors?",
       "k": 5,
       "start_time": "2024-08-01T00:00:00Z",
       "end_time": "2024-08-09T00:00:00Z",
       "hostname_pattern": "web-server-*"
     }'
   ```

The system will return a JSON response with the generated answer and relevant log entries.

3. To query the system, send a POST request to `http://localhost:8000/rag_query`:
   ```bash
   curl -X POST http://localhost:8000/rag_query \
     -H "Content-Type: application/json" \
     -d '{
       "text": "What are the most common errors?",
       "k": 5,
       "start_time": "2024-08-01T00:00:00Z",
       "end_time": "2024-08-09T00:00:00Z",
       "hostname_pattern": "web-server-*"
     }'
   ```

4. The system will return a JSON response with the generated answer and relevant log entries.

### Standalone API Usage

If you only need the API server without log processing:
```python
from rag_system import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Configuration

- Adjust the `batch_size` in `process_new_logs()` to control memory usage during log processing.
- Modify the `schedule.every(1).hour.do(process_new_logs)` line to change how often new logs are processed.
- Update the `dimension` variable if you change the embedding model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing

The project includes standalone test scripts in the `test/` directory:
- `elastic-test.py` - Elasticsearch connectivity tests
- `sentence-test.py` - Sentence transformer tests
- `pytorch-test.py` - PyTorch environment tests
- `gc-test.py` - Garbage collection tests
- `tqdm-test.py` - Progress bar tests
- `requirements-test.py` - Dependency tests

Note: There is no formal pytest test suite. These scripts are meant for manual testing and verification of individual components.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
