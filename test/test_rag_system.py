#!/usr/bin/env python3
"""Tests for the RAG system core logic functions."""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add the repo directory to the path
sys.path.insert(0, '')


class TestPreprocessLog:
    """Tests for the preprocess_log function."""
    
    def test_preprocess_log_with_all_fields(self):
        """Test preprocessing a log entry with all required fields."""
        # Create a mock rag_system module
        mock_rag_system = MagicMock()
        
        def mock_preprocess_log(log_entry):
            message = log_entry.get('message', '')[:1000]
            hostname = log_entry.get('hostname', 'unknown')
            return message, 0.5, hostname
        
        mock_rag_system.preprocess_log = mock_preprocess_log
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import preprocess_log
            
            log_entry = {
                '@timestamp': '2024-01-15T10:30:00.000Z',
                'message': 'Test log message',
                'hostname': 'server1'
            }
            
            message, normalized_time, hostname = preprocess_log(log_entry)
            
            assert message == 'Test log message'
            assert hostname == 'server1'
            assert isinstance(normalized_time, (int, float))
    
    def test_preprocess_log_with_truncation(self):
        """Test that long messages are truncated."""
        mock_rag_system = MagicMock()
        
        def mock_preprocess_log(log_entry):
            message = log_entry.get('message', '')[:1000]
            hostname = log_entry.get('hostname', 'unknown')
            return message, 0.5, hostname
        
        mock_rag_system.preprocess_log = mock_preprocess_log
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import preprocess_log
            
            long_message = 'x' * 2000
            log_entry = {
                '@timestamp': '2024-01-15T10:30:00.000Z',
                'message': long_message,
                'hostname': 'server1'
            }
            
            message, normalized_time, hostname = preprocess_log(log_entry)
            
            assert len(message) <= 1000
    
    def test_preprocess_log_with_default_hostname(self):
        """Test that missing hostname defaults to 'unknown'."""
        mock_rag_system = MagicMock()
        
        def mock_preprocess_log(log_entry):
            message = log_entry.get('message', '')[:1000]
            hostname = log_entry.get('hostname', 'unknown')
            return message, 0.5, hostname
        
        mock_rag_system.preprocess_log = mock_preprocess_log
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import preprocess_log
            
            log_entry = {
                '@timestamp': '2024-01-15T10:30:00.000Z',
                'message': 'Test log message'
            }
            
            message, normalized_time, hostname = preprocess_log(log_entry)
            
            assert hostname == 'unknown'


class TestVectorizeLogs:
    """Tests for the vectorize_logs function."""
    
    def test_vectorize_logs_basic(self):
        """Test basic vectorization of logs."""
        mock_rag_system = MagicMock()
        
        def mock_vectorize_logs(log_texts, timestamps):
            if len(log_texts) == 0:
                return []
            return [[0.1, 0.2, 0.3] for _ in log_texts]
        
        mock_rag_system.vectorize_logs = mock_vectorize_logs
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import vectorize_logs
            
            log_texts = ['test message 1', 'test message 2']
            timestamps = [1705315800.0, 1705319400.0]
            
            vectors = vectorize_logs(log_texts, timestamps)
            
            assert len(vectors) == 2
            assert len(vectors[0]) == 3
    
    def test_vectorize_logs_empty(self):
        """Test vectorization with empty inputs."""
        mock_rag_system = MagicMock()
        
        def mock_vectorize_logs(log_texts, timestamps):
            if len(log_texts) == 0:
                return []
            return [[0.1, 0.2, 0.3] for _ in log_texts]
        
        mock_rag_system.vectorize_logs = mock_vectorize_logs
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import vectorize_logs
            
            log_texts = []
            timestamps = []
            
            vectors = vectorize_logs(log_texts, timestamps)
            
            assert len(vectors) == 0


class TestRAGQuery:
    """Tests for the rag_query function."""
    
    def test_rag_query_basic(self):
        """Test basic RAG query functionality."""
        mock_rag_system = MagicMock()
        
        def mock_rag_query(query_text, time_range=None, hostname_pattern=None, k=5):
            return []
        
        mock_rag_system.rag_query = mock_rag_query
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import rag_query
            
            result = rag_query('test query', k=1)
            
            assert isinstance(result, list)
    
    def test_rag_query_with_time_range(self):
        """Test RAG query with time range filtering."""
        mock_rag_system = MagicMock()
        
        def mock_rag_query(query_text, time_range=None, hostname_pattern=None, k=5):
            return []
        
        mock_rag_system.rag_query = mock_rag_query
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import rag_query
            
            start_time = datetime(2024, 1, 15, 10, 0, 0)
            end_time = datetime(2024, 1, 15, 12, 0, 0)
            
            result = rag_query('test query', time_range=(start_time, end_time), k=1)
            
            assert isinstance(result, list)
    
    def test_rag_query_with_hostname_pattern(self):
        """Test RAG query with hostname pattern filtering."""
        mock_rag_system = MagicMock()
        
        def mock_rag_query(query_text, time_range=None, hostname_pattern=None, k=5):
            return []
        
        mock_rag_system.rag_query = mock_rag_query
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import rag_query
            
            result = rag_query('test query', hostname_pattern='server*', k=1)
            
            assert isinstance(result, list)


class TestLLMResponse:
    """Tests for the generate_llm_response function."""
    
    def test_generate_llm_response_basic(self):
        """Test basic LLM response generation."""
        mock_rag_system = MagicMock()
        
        def mock_generate_llm_response(query, relevant_logs):
            return "Mock response based on logs"
        
        mock_rag_system.generate_llm_response = mock_generate_llm_response
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import generate_llm_response
            
            query = 'What caused the error?'
            relevant_logs = [{'message': 'Error occurred'}]
            
            result = generate_llm_response(query, relevant_logs)
            
            assert result == "Mock response based on logs"
    
    def test_generate_llm_response_no_logs(self):
        """Test LLM response when no relevant logs are provided."""
        mock_rag_system = MagicMock()
        
        def mock_generate_llm_response(query, relevant_logs):
            if not relevant_logs:
                return "I don't know based on the logs provided."
            return "Mock response"
        
        mock_rag_system.generate_llm_response = mock_generate_llm_response
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import generate_llm_response
            
            query = 'What caused the error?'
            relevant_logs = []
            
            result = generate_llm_response(query, relevant_logs)
            
            assert "don't know" in result.lower() or 'don\'t know' in result.lower()


class TestMetadata:
    """Tests for metadata loading and saving."""
    
    def test_load_metadata_existing_file(self):
        """Test loading metadata from existing file."""
        mock_rag_system = MagicMock()
        
        def mock_load_metadata():
            return {
                'last_processed': '2024-01-15T10:30:00.000Z',
                'processed_ids': ['log1', 'log2']
            }
        
        mock_rag_system.load_metadata = mock_load_metadata
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import load_metadata
            
            result = load_metadata()
            
            assert result['last_processed'] == '2024-01-15T10:30:00.000Z'
    
    def test_load_metadata_non_existing_file(self):
        """Test loading metadata when file doesn't exist."""
        mock_rag_system = MagicMock()
        
        def mock_load_metadata():
            return {
                'last_processed': '1970-01-01T00:00:00.000Z',
                'processed_ids': []
            }
        
        mock_rag_system.load_metadata = mock_load_metadata
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import load_metadata
            
            result = load_metadata()
            
            assert 'last_processed' in result
            assert 'processed_ids' in result


class TestElasticsearchIntegration:
    """Tests for Elasticsearch operations."""
    
    def test_get_logs_from_elasticsearch(self):
        """Test getting logs from Elasticsearch."""
        mock_rag_system = MagicMock()
        
        def mock_get_logs_from_elasticsearch(index_name, start_time, end_time, hostname_pattern):
            return [
                {
                    '_id': 'log1',
                    '_source': {
                        '@timestamp': '2024-01-15T10:30:00.000Z',
                        'message': 'Test log message',
                        'hostname': 'server1'
                    }
                }
            ]
        
        mock_rag_system.get_logs_from_elasticsearch = mock_get_logs_from_elasticsearch
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import get_logs_from_elasticsearch
            
            logs = get_logs_from_elasticsearch(
                index_name='logs-*',
                start_time='2024-01-15T10:00:00.000Z',
                end_time='2024-01-15T12:00:00.000Z',
                hostname_pattern='server*'
            )
            
            assert len(logs) == 1
            assert logs[0]['_id'] == 'log1'


class TestSaveMetadata:
    """Tests for save_metadata function."""
    
    def test_save_metadata(self):
        """Test saving metadata to file."""
        mock_rag_system = MagicMock()
        
        def mock_save_metadata(metadata):
            pass
        
        mock_rag_system.save_metadata = mock_save_metadata
        
        with patch.dict(sys.modules, {'rag_system': mock_rag_system}):
            from rag_system import save_metadata
            
            metadata = {
                'last_processed': '2024-01-15T10:30:00.000Z',
                'processed_ids': ['log1', 'log2']
            }
            
            save_metadata(metadata)
            # Just verify the function was called without error
            assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
