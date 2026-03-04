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
   git clone https://github.com/yourusername/log-analysis-rag-system/log-analysis-rag.git
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

The project includes integration test scripts in the `test/` directory. To run all tests:

```bash
python test/requirements-test.py
```

Individual test scripts:
- `test/elastic-test.py` - Elasticsearch connectivity test
- `test/sentence-test.py` - Sentence Transformers embedding test
- `test/pytorch-test.py` - PyTorch installation test
- `test/tqdm-test.py` - Progress bar test
- `test/gc-test.py` - Garbage collection test

Note: Tests require external dependencies (Elasticsearch instance, OpenAI API key) to run successfully.

## API Reference

### POST /rag_query

Process a natural language query against log data.

**Request Body:**
```json
{
  "text": "What are the most common errors?",
  "k": 5,
  "start_time": "2024-08-01T00:00:00Z",
  "end_time": "2024-08-09T00:00:00Z",
  "hostname_pattern": "web-server-*"
}
```

**Parameters:**
- `text` (string, required): Natural language query to process
- `k` (integer, optional): Number of similar logs to retrieve (default: 5)
- `start_time` (string, optional): ISO 8601 start time filter
- `end_time` (string, optional): ISO 8601 end time filter
- `hostname_pattern` (string, optional): Regex pattern for hostname filtering

**Response:**
```json
{
  "answer": "Generated response from OpenAI...",
  "relevant_logs": [
    {"timestamp": "...", "hostname": "...", "message": "..."}
  ]
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
