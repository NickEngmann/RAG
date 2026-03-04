# Log Analysis RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for log analysis. It combines efficient log retrieval using vector embeddings with natural language processing capabilities. The system processes logs, creates embeddings, and stores them in Elasticsearch for semantic search.

## Features

- **Log Processing**: Hourly scheduled processing of log files
- **Vector Embeddings**: Uses sentence-transformers for semantic log representation
- **Elasticsearch Integration**: Stores embeddings and metadata for efficient retrieval
- **OpenAI Integration**: LLM-based question answering on log data
- **Progress Tracking**: tqdm-based progress bars for long-running operations
- **Checkpoint Support**: Can resume from interruptions

## Installation

### Prerequisites

- Python 3.8+
- Elasticsearch running locally or on a server
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with required environment variables:
   ```
   ELASTICSEARCH_URL=http://localhost:9200
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the main script:
   ```bash
   python rag_system.py
   ```

## Testing

### Running Tests

The project uses standalone test scripts in the `test/` directory (not pytest):

```bash
# Run all tests via test runner
python test/requirements-test.py

# Run individual tests
python test/elastic-test.py      # Tests Elasticsearch connectivity
python test/pytorch-test.py      # Tests PyTorch installation
python test/sentence-test.py     # Tests sentence-transformers model loading
python test/tqdm-test.py         # Tests tqdm progress bars
python test/gc-test.py           # Tests garbage collection
```

### Test Requirements

- **External services**: Elasticsearch must be running (or mocked)
- **API access**: OpenAI API key required for LLM-based tests
- **Dependencies**: All packages from `requirements.txt` must be installed
- **Python version**: 3.8+

**Note**: This project uses standalone test scripts rather than pytest. Tests verify dependency installation and service connectivity. For CI/CD or isolated testing, mock Elasticsearch and OpenAI services.

## Architecture

### Components

1. **rag_system.py**: Main entry point for log processing and RAG operations
2. **test/**: Directory containing standalone test scripts
3. **requirements.txt**: Python dependencies
4. **.env**: Environment variables configuration

### Data Flow

1. Logs are processed hourly via schedule library
2. Text embeddings created using sentence-transformers 'all-MiniLM-L6-v2'
3. Embeddings stored in Elasticsearch with metadata
4. Queries processed through OpenAI for natural language responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
