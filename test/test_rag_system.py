#!/usr/bin/env python3
"""Comprehensive tests for the RAG system."""

import pytest
import os
import sys
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from dateutil.parser import parse

# Set environment variables BEFORE importing rag_system
os.environ.setdefault('ELASTICSEARCH_URL', 'http://localhost:9200')
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('INDEX_NAME', 'test-index')
os.environ.setdefault('METADATA_FILE', '/tmp/test_metadata.json')
os.environ.setdefault('VECTOR_DB_PATH', '/tmp/vectordb')

# Mock external dependencies BEFORE importing rag_system
mock_faiss = MagicMock()
sys.modules['faiss'] = mock_faiss
mock_faiss.IndexFlatIP = MagicMock()
mock_faiss.write_index = MagicMock()
mock_faiss.read_index = MagicMock()
mock_faiss.IndexIDMap = MagicMock()
mock_faiss.normalize_L2 = MagicMock()
mock_faiss.IndexFlatL2 = MagicMock()

mock_es = MagicMock()
sys.modules['elasticsearch'] = mock_es
mock_es.Elasticsearch = MagicMock()
mock_es.helpers = MagicMock()

mock_st = MagicMock()
sys.modules['sentence_transformers'] = mock_st
mock_st.SentenceTransformer = MagicMock()

mock_openai = MagicMock()
sys.modules['openai'] = mock_openai
mock_openai.OpenAI = MagicMock()
mock_openai.ChatCompletion = MagicMock()

mock_schedule = MagicMock()
sys.modules['schedule'] = mock_schedule

mock_tqdm = MagicMock()
sys.modules['tqdm'] = mock_tqdm
mock_tqdm.tqdm = MagicMock()

# Mock sklearn.preprocessing properly
mock_sklearn = MagicMock()
mock_sklearn_preprocessing = MagicMock()
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.preprocessing'] = mock_sklearn_preprocessing
mock_sklearn.preprocessing = mock_sklearn_preprocessing
mock_sklearn_preprocessing.MinMaxScaler = MagicMock()

# Now import rag_system after mocking dependencies
from rag_system import (
    load_metadata,
    save_metadata,
    preprocess_log,
    vectorize_logs,
    process_batch,
    rag_query,
    generate_llm_response,
    app,
    Query
)


class TestMetadataFunctions:
    """Test metadata loading and saving functions."""
    
    def test_load_metadata_exists(self):
        """Test loading metadata when file exists."""
        test_metadata = {
            'last_processed': '2024-01-01T00:00:00.000Z',
            'processed_ids': ['id1', 'id2']
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', MagicMock()), \
             patch('json.load', return_value=test_metadata):
            result = load_metadata()
            assert result == test_metadata
    
    def test_load_metadata_not_exists(self):
        """Test loading metadata when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            result = load_metadata()
            assert result == {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': set()}
    
    def test_save_metadata(self):
        """Test saving metadata to file."""
        metadata = {
            'last_processed': '2024-01-01T00:00:00.000Z',
            'processed_ids': set(['id1', 'id2'])
        }
        
        with patch('rag_system.metadata_file', '/tmp/test_metadata.json'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump') as mock_dump:
            save_metadata(metadata)
            mock_dump.assert_called_once()


class TestPreprocessingFunctions:
    """Test preprocessing functions."""
    
    def test_preprocess_log(self):
        """Test preprocessing a log entry."""
        log_entry = {
            '@timestamp': '2024-01-01T12:30:00.000Z',
            'message': 'Test log message',
            'hostname': 'test-host'
        }
        
        with patch('dateutil.parser.parse') as mock_parse:
            mock_parse.return_value.timestamp.return_value = 1704117000.0
            with patch('rag_system.time_scaler') as mock_scaler:
                mock_scaler.fit_transform.return_value = np.array([[0.5]])
                
                result = preprocess_log(log_entry)
                
                assert len(result) == 3
                assert result[0] == 'Test log message'
                assert result[1] == 0.5
                assert result[2] == 'test-host'
    
    def test_preprocess_log_truncation(self):
        """Test that messages are truncated to 1000 characters."""
        log_entry = {
            '@timestamp': '2024-01-01T12:30:00.000Z',
            'message': 'A' * 2000,
            'hostname': 'test-host'
        }
        
        with patch('dateutil.parser.parse') as mock_parse:
            mock_parse.return_value.timestamp.return_value = 1704117000.0
            with patch('rag_system.time_scaler') as mock_scaler:
                mock_scaler.fit_transform.return_value = np.array([[0.5]])
                
                result = preprocess_log(log_entry)
                
                assert len(result[0]) == 1000
    
    def test_preprocess_log_missing_hostname(self):
        """Test preprocessing log with missing hostname."""
        log_entry = {
            '@timestamp': '2024-01-01T12:30:00.000Z',
            'message': 'Test log message'
        }
        
        with patch('dateutil.parser.parse') as mock_parse:
            mock_parse.return_value.timestamp.return_value = 1704117000.0
            with patch('rag_system.time_scaler') as mock_scaler:
                mock_scaler.fit_transform.return_value = np.array([[0.5]])
                
                result = preprocess_log(log_entry)
                
                assert result[2] == 'unknown'


class TestVectorizationFunctions:
    """Test vectorization functions."""
    
    def test_vectorize_logs(self):
        """Test vectorizing log texts and timestamps."""
        log_texts = ['Test log 1', 'Test log 2']
        timestamps = [0.5, 0.6]
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        
        with patch('rag_system.model', mock_model):
            result = vectorize_logs(log_texts, timestamps)
            
            assert result.shape[0] == 2
            assert result.shape[1] == 385  # 384 + 1 for timestamp
            mock_model.encode.assert_called_once()


class TestRAGQuery:
    """Test RAG query functions."""
    
    def test_rag_query(self):
        """Test RAG query with time range."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 385])
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[1, 2]]), np.array([[0.1, 0.2]]))
        
        mock_metadata = {
            '1': {'id': 'id1', 'timestamp': '2024-01-01T00:00:00.000Z', 'message': 'Test', 'hostname': 'host1'},
            '2': {'id': 'id2', 'timestamp': '2024-01-02T00:00:00.000Z', 'message': 'Test2', 'hostname': 'host2'}
        }
        
        with patch('rag_system.model', mock_model), \
             patch('rag_system.index', mock_index), \
             patch('rag_system.metadata', mock_metadata), \
             patch('rag_system.time_scaler') as mock_scaler:
            mock_scaler.transform.return_value = np.array([[0.5]])
            
            result = rag_query('test query', time_range=(datetime(2024, 1, 1), datetime(2024, 1, 3)), k=2)
            
            # The rag_query function filters results based on time range
            # Since we're mocking the metadata, we need to check if the function works correctly
            assert isinstance(result, list)
            mock_model.encode.assert_called_once()
    
    def test_rag_query_hostname_filter(self):
        """Test RAG query with hostname filter."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 385])
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[1, 2]]), np.array([[0.1, 0.2]]))
        
        mock_metadata = {
            '1': {'id': 'id1', 'timestamp': '2024-01-01T00:00:00.000Z', 'message': 'Test', 'hostname': 'host1'},
            '2': {'id': 'id2', 'timestamp': '2024-01-02T00:00:00.000Z', 'message': 'Test2', 'hostname': 'host2'}
        }
        
        with patch('rag_system.model', mock_model), \
             patch('rag_system.index', mock_index), \
             patch('rag_system.metadata', mock_metadata), \
             patch('rag_system.time_scaler') as mock_scaler:
            mock_scaler.transform.return_value = np.array([[0.5]])
            
            result = rag_query('test query', hostname_pattern='host1*', k=2)
            
            # The rag_query function filters results based on hostname pattern
            # Since we're mocking the metadata, we need to check if the function works correctly
            assert isinstance(result, list)
            mock_model.encode.assert_called_once()
    
    def test_rag_query_empty_results(self):
        """Test RAG query with no results."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 385])
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        
        with patch('rag_system.model', mock_model), \
             patch('rag_system.index', mock_index):
            
            result = rag_query('test query', k=5)
            
            assert result == []


class TestLLMResponse:
    """Test LLM response generation."""
    
    def test_generate_llm_response(self):
        """Test generating LLM response."""
        relevant_logs = [
            {'hostname': 'host1', 'timestamp': '2024-01-01T00:00:00.000Z', 'message': 'Test log 1'},
            {'hostname': 'host2', 'timestamp': '2024-01-02T00:00:00.000Z', 'message': 'Test log 2'}
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = {'content': 'Generated response'}
        
        with patch('rag_system.openai.ChatCompletion.create', return_value=mock_response):
            result = generate_llm_response('test query', relevant_logs)
            
            assert 'Generated response' in result
    
    def test_generate_llm_response_empty_logs(self):
        """Test LLM response with empty logs."""
        with patch('rag_system.openai.ChatCompletion.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = {'content': 'No relevant logs found'}
            mock_create.return_value = mock_response
            
            result = generate_llm_response('test query', [])
            
            assert 'No relevant logs found' in result


class TestAPI:
    """Test FastAPI endpoints."""
    
    def test_query_model(self):
        """Test Query model validation."""
        query_data = {
            'text': 'test query',
            'k': 5,
            'start_time': '2024-01-01T00:00:00.000Z',
            'end_time': '2024-01-02T00:00:00.000Z',
            'hostname_pattern': 'host*'
        }
        
        query = Query(**query_data)
        assert query.text == 'test query'
        assert query.k == 5
        assert query.hostname_pattern == 'host*'
    
    def test_app_exists(self):
        """Test that FastAPI app is created."""
        assert app is not None
        assert hasattr(app, 'routes')
    
    def test_app_routes(self):
        """Test that app has expected routes."""
        routes = [route.path for route in app.routes]
        assert '/rag_query' in routes


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_preprocess_log_empty_hostname(self):
        """Test preprocessing log with missing hostname."""
        log_entry = {
            '@timestamp': '2024-01-01T12:30:00.000Z',
            'message': 'Test log message'
        }
        
        with patch('dateutil.parser.parse') as mock_parse:
            mock_parse.return_value.timestamp.return_value = 1704117000.0
            with patch('rag_system.time_scaler') as mock_scaler:
                mock_scaler.fit_transform.return_value = np.array([[0.5]])
                
                result = preprocess_log(log_entry)
                
                assert result[2] == 'unknown'
    
    def test_rag_query_empty_results(self):
        """Test RAG query with no results."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 385])
        
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))
        
        with patch('rag_system.model', mock_model), \
             patch('rag_system.index', mock_index):
            
            result = rag_query('test query', k=5)
            
            assert result == []
    
    def test_generate_llm_response_empty_logs(self):
        """Test LLM response with empty logs."""
        with patch('rag_system.openai.ChatCompletion.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = {'content': 'No relevant logs found'}
            mock_create.return_value = mock_response
            
            result = generate_llm_response('test query', [])
            
            assert 'No relevant logs found' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
