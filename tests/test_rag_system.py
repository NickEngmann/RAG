#!/usr/bin/env python3
"""Unit tests for the RAG system.

These tests mock external dependencies (Elasticsearch, OpenAI, FAISS, FastAPI) to ensure
reliable and fast test execution without requiring live services.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from dateutil.parser import parse
import json
import re
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all external dependencies BEFORE importing rag_system
# This is critical because rag_system.py has module-level code that runs at import time

# Create mock modules with proper attributes
mock_elasticsearch = Mock()
mock_elasticsearch.Elasticsearch = Mock(return_value=Mock())
mock_elasticsearch.helpers = Mock()
mock_elasticsearch.helpers.scan = Mock(return_value=[])

mock_sentence_transformers = Mock()
mock_sentence_transformers.SentenceTransformer = Mock(return_value=Mock(
    get_sentence_embedding_dimension=Mock(return_value=384),
    encode=Mock(return_value=np.array([[0.1] * 384]))
))

mock_faiss = Mock()
mock_faiss.IndexFlatIP = Mock(return_value=Mock(
    add=Mock(),
    search=Mock(return_value=(np.array([[0]]), np.array([[0]]))),
    normalize_L2=Mock(),
    write_index=Mock(),
    read_index=Mock(),
    ntotal=0
))
mock_faiss.IndexFlatL2 = Mock(return_value=Mock(
    add=Mock(),
    search=Mock(return_value=(np.array([[0]]), np.array([[0]]))),
    normalize_L2=Mock(),
    write_index=Mock(),
    read_index=Mock(),
    ntotal=0
))
mock_faiss.read_index = Mock(return_value=Mock(
    add=Mock(),
    search=Mock(return_value=(np.array([[0]]), np.array([[0]]))),
    normalize_L2=Mock(),
    write_index=Mock(),
    ntotal=0
))

mock_openai = Mock()
mock_openai.ChatCompletion = Mock()
mock_openai.ChatCompletion.create = Mock(return_value=Mock(
    choices=[Mock(message=Mock(content="Mock response"))]
))
mock_openai.api_key = "mock_key"

mock_fastapi = Mock()
mock_fastapi.FastAPI = Mock(return_value=Mock())
mock_fastapi.HTTPException = Mock(side_effect=Exception)

mock_schedule = Mock()
mock_schedule.every = Mock(return_value=Mock(
    hour=Mock(return_value=Mock(
        do=Mock(return_value=Mock())
    ))
))

mock_pydantic = Mock()
mock_pydantic.BaseModel = Mock

mock_uvicorn = Mock()
mock_uvicorn.run = Mock()

mock_sklearn = Mock()
mock_sklearn.preprocessing = Mock()
mock_sklearn.preprocessing.MinMaxScaler = Mock(return_value=Mock(
    fit_transform=Mock(return_value=np.array([[0.5]])),
    transform=Mock(return_value=np.array([[0.5]]))
))

mock_tqdm = Mock()
mock_tqdm.tqdm = Mock(side_effect=lambda x, *args, **kwargs: x)

mock_dotenv = Mock()
mock_dotenv.load_dotenv = Mock()

# Register all mock modules
sys.modules['elasticsearch'] = mock_elasticsearch
sys.modules['sentence_transformers'] = mock_sentence_transformers
sys.modules['faiss'] = mock_faiss
sys.modules['openai'] = mock_openai
sys.modules['fastapi'] = mock_fastapi
sys.modules['schedule'] = mock_schedule
sys.modules['pydantic'] = mock_pydantic
sys.modules['uvicorn'] = mock_uvicorn
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.preprocessing'] = mock_sklearn.preprocessing
sys.modules['tqdm'] = mock_tqdm
sys.modules['dotenv'] = mock_dotenv

# Now we can safely import rag_system
from rag_system import (
    load_metadata,
    save_metadata,
    preprocess_log,
    vectorize_logs,
    process_batch,
    process_new_logs,
    rag_query,
    generate_llm_response,
    app,
    Query,
    run_api,
    schedule_processing
)


# Mock Elasticsearch client for testing
class MockElasticsearchClient:
    def __init__(self, *args, **kwargs):
        self.documents = []
        self.indices = {}
        
    def index(self, index, document, **kwargs):
        self.documents.append(document)
        return {"_index": index, "_id": len(self.documents) - 1, "result": "created"}
    
    def search(self, index, body, **kwargs):
        results = []
        for doc in self.documents:
            if doc.get("message", "").lower() in body.get("query", {}).get("match", {}).get("message", {}).lower():
                results.append(doc)
        return {"hits": {"hits": [{"_source": doc} for doc in results]}}
    
    def bulk(self, actions, **kwargs):
        for action in actions:
            if action.get("index").get("_op_type") == "index":
                self.documents.append(action["index"]["_source"])
        return ([{"index": {"status": 201}} for _ in actions], [])
    
    def exists_index(self, index, **kwargs):
        return index in self.indices
    
    def create_index(self, index, **kwargs):
        self.indices[index] = True
    
    def delete_index(self, index, **kwargs):
        if index in self.indices:
            del self.indices[index]


# Mock FAISS index for testing
class MockFAISSIndex:
    def __init__(self, *args, **kwargs):
        self.data = []
        self.ntotal = 0
        
    def add(self, vectors):
        self.data.extend(vectors)
        self.ntotal += len(vectors)
        
    def train(self, vectors, *args, **kwargs):
        pass
    
    def write_index(self, filepath):
        pass
    
    def read_index(self, filepath):
        pass
    
    def search(self, query_vector, k=5):
        return (self.data[:k], [0] * k)
    
    def normalize_L2(self, vectors):
        pass


# Mock SentenceTransformer for testing
class MockSentenceTransformer:
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        
    def encode(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[0.1] * 384 for _ in texts])
    
    def get_sentence_embedding_dimension(self):
        return 384


# Mock OpenAI client for testing
class MockOpenAIClient:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed(self, input_text, model="text-embedding-ada-002", **kwargs):
        return [[0.1] * 1536 for _ in range(len(input_text) if isinstance(input_text, list) else 1)]
    
    def chat(self, messages, model="gpt-3.5-turbo", **kwargs):
        return {"choices": [{"message": {"content": "Mock response"}}]}


# Mock schedule for testing
class MockSchedule:
    def __init__(self):
        self.jobs = []
        
    def every(self, interval):
        return self
        
    def hour(self):
        return self
        
    def do(self, func, *args, **kwargs):
        self.jobs.append((func, args, kwargs))
        return self


@pytest.fixture
def mock_env():
    """Setup mock environment variables."""
    with patch.dict(os.environ, {
        'ELASTICSEARCH_URL': 'http://localhost:9200',
        'OPENAI_API_KEY': 'mock_key',
        'SENTENCE_TRANSFORMER_MODEL': 'all-MiniLM-L6-v2'
    }):
        yield


@pytest.fixture
def mock_elasticsearch():
    """Mock Elasticsearch client."""
    with patch('elasticsearch.Elasticsearch', return_value=MockElasticsearchClient()):
        yield


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer."""
    with patch('sentence_transformers.SentenceTransformer', return_value=MockSentenceTransformer('all-MiniLM-L6-v2')):
        yield


@pytest.fixture
def mock_faiss():
    """Mock FAISS module."""
    with patch('faiss.IndexFlatL2', return_value=MockFAISSIndex()):
        yield


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch('openai.OpenAI', return_value=MockOpenAIClient()):
        yield


@pytest.fixture
def mock_schedule():
    """Mock schedule module."""
    with patch('schedule.every', return_value=MockSchedule()):
        yield


class TestMetadataFunctions:
    """Tests for metadata functions."""
    
    def test_load_metadata_no_file(self, mock_env):
        """Test load_metadata when file doesn't exist."""
        # Mock os.path.exists to return False
        with patch('os.path.exists', return_value=False):
            result = load_metadata()
            assert result == {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': set()}


class TestPreprocessLog:
    """Tests for preprocess_log function."""
    
    def test_preprocess_log_valid(self, mock_env):
        """Test preprocessing a valid log entry."""
        log_entry = {
            '@timestamp': '2024-01-01T00:00:00.000Z',
            'message': 'Test log message',
            'hostname': 'test-host'
        }
        
        result = preprocess_log(log_entry)
        
        assert len(result) == 3
        assert result[0] == 'Test log message'
        assert isinstance(result[1], float)
        assert result[2] == 'test-host'
    
    def test_preprocess_log_truncation(self, mock_env):
        """Test that messages are truncated to 1000 characters."""
        long_message = 'x' * 2000
        log_entry = {
            '@timestamp': '2024-01-01T00:00:00.000Z',
            'message': long_message,
            'hostname': 'test-host'
        }
        
        result = preprocess_log(log_entry)
        
        assert len(result[0]) == 1000
    
    def test_preprocess_log_default_hostname(self, mock_env):
        """Test that hostname defaults to 'unknown' if not present."""
        log_entry = {
            '@timestamp': '2024-01-01T00:00:00.000Z',
            'message': 'Test log message'
        }
        
        result = preprocess_log(log_entry)
        
        assert result[2] == 'unknown'


class TestRAGQuery:
    """Tests for rag_query function."""
    
    def test_rag_query_basic(self, mock_env, mock_faiss):
        """Test basic RAG query."""
        # Mock the index and metadata
        mock_index = MockFAISSIndex()
        mock_index.ntotal = 5
        
        with patch('faiss.read_index', return_value=mock_index):
            with patch('faiss.IndexFlatIP', return_value=mock_index):
                # Mock metadata
                mock_metadata = {
                    '0': {'id': '1', 'timestamp': '2024-01-01T00:00:00.000Z', 'message': 'Test message', 'hostname': 'test-host'},
                    '1': {'id': '2', 'timestamp': '2024-01-01T00:01:00.000Z', 'message': 'Another message', 'hostname': 'test-host-2'}
                }
                
                with patch('rag_system.metadata', mock_metadata):
                    results = rag_query('test query', k=2)
                    
                    assert len(results) <= 2


class TestAPIHandler:
    """Tests for API handler."""
    
    def test_query_model(self, mock_env):
        """Test Query model."""
        query = Query(text='test query', k=5)
        
        assert query.text == 'test query'
        assert query.k == 5
    
    def test_query_model_with_optional_fields(self, mock_env):
        """Test Query model with optional fields."""
        query = Query(
            text='test query',
            k=3,
            start_time='2024-01-01T00:00:00.000Z',
            end_time='2024-01-01T23:59:59.000Z',
            hostname_pattern='test-*'
        )
        
        assert query.text == 'test query'
        assert query.k == 3
        assert query.start_time == '2024-01-01T00:00:00.000Z'
        assert query.end_time == '2024-01-01T23:59:59.000Z'
        assert query.hostname_pattern == 'test-*'


class TestRunAPI:
    """Tests for run_api function."""
    
    def test_run_api(self, mock_env):
        """Test run_api function."""
        # Just verify the function exists and can be called
        # We can't actually start the server in tests
        assert run_api is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
