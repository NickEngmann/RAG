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

## Testing

The project includes several test scripts to verify dependencies and functionality:

### Test Scripts

- **tqdm-test.py**: Tests the tqdm progress bar library
- **pytorch-test.py**: Tests PyTorch installation and tensor operations
- **sentence-test.py**: Tests sentence-transformers model loading and embedding generation
- **elastic-test.py**: Tests Elasticsearch connection and basic operations
- **gc-test.py**: Tests garbage collection functionality

### Running Tests

To run all tests:
```bash
cd /workspace/repo/test && python requirements-test.py
```

To run individual tests:
```bash
cd /workspace/repo/test && python tqdm-test.py
cd /workspace/repo/test && python pytorch-test.py
cd /workspace/repo/test && python sentence-test.py
cd /workspace/repo/test && python elastic-test.py
cd /workspace/repo/test && python gc-test.py
```

### Test Requirements

- Python 3.8+
- All dependencies from requirements.txt
- Elasticsearch instance (for elastic-test.py)
- OpenAI API key (for some tests)

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Error**: Ensure ELASTICSEARCH_URL is correctly set in .env file and Elasticsearch is running
2. **OpenAI API Error**: Verify OPENAI_API_KEY is set and has valid credentials
3. **Memory Issues**: Reduce batch_size in process_new_logs() function
4. **FAISS Index Errors**: Ensure /mnt/vectordb/ directory exists and is writable

### Environment Setup

Create a .env file with the following variables:
```
ELASTICSEARCH_URL=http://your_elasticsearch_ip:9200
OPENAI_API_KEY=your_openai_api_key
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

