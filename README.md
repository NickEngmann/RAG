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

## API Documentation

The FastAPI server provides the following endpoints:

### Health Check
- **GET** `/health` - Returns system health status

### Log Processing
- **POST** `/process_logs` - Trigger manual log processing
- **GET** `/logs` - List available log entries
- **GET** `/logs/{log_id}` - Get specific log entry

### RAG Query
- **POST** `/rag_query` - Perform RAG-based query on log data
  - `text`: Query text for log analysis (required)
  - `k`: Number of relevant logs to retrieve (default: 5)
  - `start_time`: Optional start time filter (ISO 8601 format)
  - `end_time`: Optional end time filter (ISO 8601 format)
  - `hostname_pattern`: Optional hostname pattern filter

### System Info
- **GET** `/info` - Return system information and version

## Testing

### Running Tests

Run all tests using the test runner:
```bash
python test/test_runner.py
```

Or run individual test files:
```bash
python test/tqdm-test.py
python test/pytorch-test.py
python test/sentence-test.py
python test/elastic-test.py
python test/gc-test.py
```

### Test Dependencies

Install test dependencies:
```bash
pip install -r test/requirements-test.py
```

### Environment Setup for Tests

Ensure the following environment variables are set:
- `ELASTICSEARCH_URL`: URL of the Elasticsearch instance (default: http://localhost:9200)
- `OPENAI_API_KEY`: OpenAI API key for LLM integration

Create a `.env` file in the project root with:
```
ELASTICSEARCH_URL=http://localhost:9200
OPENAI_API_KEY=your_api_key_here
```

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Ensure Elasticsearch is running at the specified URL
   - Check network connectivity and firewall settings
   - Verify the ELASTICSEARCH_URL environment variable

2. **FAISS Index Not Found**
   - Run the initial log processing to create the index
   - Ensure the `/mnt/vectordb/` directory exists and is writable
   - Check disk space availability

3. **OpenAI API Errors**
   - Verify the OPENAI_API_KEY is correctly set
   - Check API quota and rate limits
   - Ensure internet connectivity for API calls

4. **Memory Issues**
   - Reduce batch size in log processing
   - Increase system memory or use smaller embedding models
   - Clear and rebuild FAISS index periodically

## Architecture

### Component Overview

1. **Log Ingestion**: Processes raw log files and stores them in Elasticsearch
2. **Embedding Generation**: Creates vector embeddings using Sentence Transformers
3. **Vector Search**: Uses FAISS for efficient similarity search
4. **API Layer**: FastAPI endpoints for log retrieval and search
5. **Scheduler**: Background job for periodic log processing

### Data Flow

```
Raw Logs → Elasticsearch → Embedding Generation → FAISS Index → API Queries
```

### File Structure

```
.
├── rag_system.py          # Main application entry point
├── requirements.txt       # Python dependencies
├── test/                  # Test scripts directory
│   ├── test_runner.py    # Unified test execution
│   ├── elastic-test.py   # Elasticsearch connectivity test
│   ├── sentence-test.py  # Sentence transformer test
│   ├── pytorch-test.py   # PyTorch backend test
│   ├── tqdm-test.py      # Progress bar test
│   └── gc-test.py        # Garbage collection test
├── .env                   # Environment variables (not in git)
└── README.md              # This file
```

## Security Considerations

- Never commit `.env` files containing API keys to version control
- Use environment variables for sensitive configuration
- Implement rate limiting on API endpoints in production
- Validate all user inputs before processing
- Use HTTPS for all API communications in production

## Performance Optimization

- Use batch processing for large log files
- Implement caching for frequently accessed embeddings
- Optimize FAISS index parameters for your use case
- Use asynchronous processing for I/O-bound operations
- Monitor and tune Elasticsearch cluster settings

## Future Enhancements

- [ ] Add support for multiple embedding models
- [ ] Implement log categorization and tagging
- [ ] Add real-time log streaming support
- [ ] Implement distributed processing for large datasets
- [ ] Add visualization dashboards for log analysis
- [ ] Support for additional log formats (JSON, XML, etc.)
- [ ] Implement log anomaly detection
- [ ] Add multi-language support for embeddings

