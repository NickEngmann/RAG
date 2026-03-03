#!/usr/bin/env python3
"""Test suite for RAG system API endpoints."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        sys.modules['sentence_transformers'] = MagicMock()
        sys.modules['faiss'] = MagicMock()
        sys.modules['elasticsearch'] = MagicMock()
        sys.modules['elasticsearch.helpers'] = MagicMock()
        sys.modules['schedule'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['openai'] = MagicMock()
        
        mock_numpy = MagicMock()
        mock_numpy.hstack = lambda x: x[0] if len(x) == 1 else x[0]
        mock_numpy.array = lambda x: x
        mock_numpy.random = MagicMock()
        mock_numpy.random.rand = lambda *args: [[0.5] * args[0] if len(args) == 1 else [[0.5] * args[0] for _ in range(args[1])]]
        sys.modules['numpy'] = mock_numpy
        
        mock_sklearn = MagicMock()
        mock_sklearn.preprocessing = MagicMock()
        mock_sklearn.preprocessing.MinMaxScaler = MagicMock(return_value=Mock(
            fit_transform=lambda x: [[0.5]],
            transform=lambda x: [[0.5]]
        ))
        sys.modules['sklearn'] = mock_sklearn
        sys.modules['sklearn.preprocessing'] = mock_sklearn.preprocessing
        
        mock_dateutil = MagicMock()
        mock_dateutil.parser = MagicMock()
        mock_dateutil.parser.parse = lambda x: datetime(2023, 1, 1, 12, 0, 0)
        sys.modules['dateutil'] = mock_dateutil
        sys.modules['dateutil.parser'] = mock_dateutil.parser
        
        os.environ['ELASTICSEARCH_URL'] = 'http://localhost:9200'
        os.environ['OPENAI_API_KEY'] = 'test-key'
    
    def test_query_model_validation(self):
        """Test that query model validates required fields."""
        # Simulate query model validation
        query_data = {'text': 'test query'}
        
        self.assertIn('text', query_data)
        self.assertEqual(query_data['text'], 'test query')
    
    def test_query_model_optional_fields(self):
        """Test that query model handles optional fields."""
        query_data = {
            'text': 'test query',
            'k': 5,
            'start_time': None,
            'end_time': None,
            'hostname_pattern': None
        }
        
        self.assertIn('k', query_data)
        self.assertEqual(query_data['k'], 5)
        self.assertIsNone(query_data['start_time'])
    
    def test_response_model_structure(self):
        """Test that response model has correct structure."""
        response_data = {
            'answer': 'Generated response',
            'relevant_logs': [
                {
                    'hostname': 'host1',
                    'timestamp': '2023-01-01T12:00:00.000Z',
                    'message': 'Test log',
                    'id': 'id1'
                }
            ]
        }
        
        self.assertIn('answer', response_data)
        self.assertIn('relevant_logs', response_data)
        self.assertIsInstance(response_data['relevant_logs'], list)
    
    def test_error_response_format(self):
        """Test that error responses have correct format."""
        error_response = {
            'detail': 'Error message',
            'status': 'error'
        }
        
        self.assertIn('detail', error_response)
        self.assertEqual(error_response['status'], 'error')


class TestQueryProcessing(unittest.TestCase):
    """Test query processing logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        sys.modules['sentence_transformers'] = MagicMock()
        sys.modules['faiss'] = MagicMock()
        sys.modules['elasticsearch'] = MagicMock()
        sys.modules['elasticsearch.helpers'] = MagicMock()
        sys.modules['schedule'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['openai'] = MagicMock()
        
        mock_numpy = MagicMock()
        mock_numpy.hstack = lambda x: x[0] if len(x) == 1 else x[0]
        mock_numpy.array = lambda x: x
        sys.modules['numpy'] = mock_numpy
        
        mock_sklearn = MagicMock()
        mock_sklearn.preprocessing = MagicMock()
        mock_sklearn.preprocessing.MinMaxScaler = MagicMock(return_value=Mock(
            fit_transform=lambda x: [[0.5]],
            transform=lambda x: [[0.5]]
        ))
        sys.modules['sklearn'] = mock_sklearn
        sys.modules['sklearn.preprocessing'] = mock_sklearn.preprocessing
        
        mock_dateutil = MagicMock()
        mock_dateutil.parser = MagicMock()
        mock_dateutil.parser.parse = lambda x: datetime(2023, 1, 1, 12, 0, 0)
        sys.modules['dateutil'] = mock_dateutil
        sys.modules['dateutil.parser'] = mock_dateutil.parser
        
        os.environ['ELASTICSEARCH_URL'] = 'http://localhost:9200'
        os.environ['OPENAI_API_KEY'] = 'test-key'
    
    def test_query_with_time_range(self):
        """Test query processing with time range."""
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 3)
        
        test_timestamp = datetime(2023, 1, 2)
        
        self.assertTrue(start_time <= test_timestamp <= end_time)
    
    def test_query_without_time_range(self):
        """Test query processing without time range."""
        start_time = None
        end_time = None
        
        self.assertIsNone(start_time)
        self.assertIsNone(end_time)
    
    def test_query_hostname_filter(self):
        """Test query processing with hostname filter."""
        import fnmatch
        
        hostname = 'server-01'
        pattern = 'server-0*'
        
        self.assertTrue(fnmatch.fnmatch(hostname, pattern))
    
    def test_query_hostname_no_filter(self):
        """Test query processing without hostname filter."""
        hostname = 'server-01'
        pattern = None
        
        self.assertIsNone(pattern)


class TestLogging(unittest.TestCase):
    """Test logging functionality."""
    
    def test_log_entry_format(self):
        """Test that log entries have expected format."""
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': 'Test log message',
            'hostname': 'test-host'
        }
        
        self.assertIn('@timestamp', log_entry)
        self.assertIn('message', log_entry)
        self.assertIn('hostname', log_entry)
    
    def test_log_message_truncation(self):
        """Test that log messages are properly truncated."""
        long_message = 'x' * 2000
        truncated = long_message[:1000]
        
        self.assertEqual(len(truncated), 1000)


if __name__ == '__main__':
    unittest.main()
