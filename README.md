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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Testing

The project includes test scripts in the `test/` directory:

```bash
python test/requirements-test.py
```

This runs all individual test scripts (tqdm-test.py, pytorch-test.py, sentence-test.py, elastic-test.py, gc-test.py) to verify dependencies and basic functionality.

Note: Tests require Elasticsearch and OpenAI API to be configured.

## Environment Variables

Required environment variables (can be stored in `.env` file):

- `ELASTICSEARCH_URL` - URL of your Elasticsearch instance (e.g., http://localhost:9200)
- `OPENAI_API_KEY` - Your OpenAI API key for GPT model access

## Data Storage

- FAISS index: `/mnt/vectordb/vector_index.faiss`
- Metadata: `/mnt/vectordb/metadata.json`

The system supports checkpointing - if interrupted, it saves progress and can resume from the last checkpoint.

## Testing

Run individual test scripts:

```bash
python test/tqdm-test.py
python test/pytorch-test.py
python test/sentence-test.py
python test/elastic-test.py
python test/gc-test.py
```

Run all tests with the test runner:

```bash
python test/requirements-test.py
```

**Note**: The `requirements-test.py` runner has a known bug where it looks for test scripts in the wrong directory. Use individual test commands above for reliable testing.