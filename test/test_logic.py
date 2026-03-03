#!/usr/bin/env python3
"""Test suite for RAG system logic functions with mocked dependencies."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGLogic(unittest.TestCase):
    """Test cases for RAG system logic functions."""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        # Mock the heavy dependencies before importing
        sys.modules['sentence_transformers'] = MagicMock()
        sys.modules['faiss'] = MagicMock()
        sys.modules['elasticsearch'] = MagicMock()
        sys.modules['elasticsearch.helpers'] = MagicMock()
        sys.modules['schedule'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['openai'] = MagicMock()
        
        # Mock numpy
        mock_numpy = MagicMock()
        mock_numpy.hstack = lambda x: x[0] if len(x) == 1 else x[0]
        mock_numpy.array = lambda x: x
        mock_numpy.random = MagicMock()
        mock_numpy.random.rand = lambda *args: [[0.5] * args[0] if len(args) == 1 else [[0.5] * args[0] for _ in range(args[1])]]
        sys.modules['numpy'] = mock_numpy
        
        # Mock sklearn
        mock_sklearn = MagicMock()
        mock_sklearn.preprocessing = MagicMock()
        mock_sklearn.preprocessing.MinMaxScaler = MagicMock(return_value=Mock(
            fit_transform=lambda x: [[0.5]],
            transform=lambda x: [[0.5]]
        ))
        sys.modules['sklearn'] = mock_sklearn
        sys.modules['sklearn.preprocessing'] = mock_sklearn.preprocessing
        
        # Mock dateutil
        mock_dateutil = MagicMock()
        mock_dateutil.parser = MagicMock()
        mock_dateutil.parser.parse = lambda x: Mock(timestamp=lambda: 1234567890.0)
        sys.modules['dateutil'] = mock_dateutil
        sys.modules['dateutil.parser'] = mock_dateutil.parser
        
        # Import the module after mocking
        import rag_system
        self.rag_system = rag_system
        
        # Mock environment variables
        os.environ['ELASTICSEARCH_URL'] = 'http://localhost:9200'
        os.environ['OPENAI_API_KEY'] = 'test-key'
    
    def test_preprocess_log(self):
        """Test log preprocessing function."""
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': 'Test log message that should be processed correctly',
            'hostname': 'test-host'
        }
        
        result = self.rag_system.preprocess_log(log_entry)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], log_entry['message'])
        self.assertEqual(result[2], log_entry['hostname'])
    
    def test_preprocess_log_truncation(self):
        """Test that log messages are truncated to 1000 characters."""
        long_message = 'x' * 2000
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': long_message,
            'hostname': 'test-host'
        }
        
        result = self.rag_system.preprocess_log(log_entry)
        
        self.assertEqual(len(result[0]), 1000)
    
    def test_load_metadata_file_exists(self):
        """Test loading metadata from existing file."""
        import json
        test_metadata = {
            'last_processed': '2023-01-01T00:00:00.000Z',
            'processed_ids': ['id1', 'id2']
        }
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(test_metadata))):
                result = self.rag_system.load_metadata()
                
                self.assertEqual(result['last_processed'], test_metadata['last_processed'])
                self.assertEqual(set(result['processed_ids']), set(test_metadata['processed_ids']))
    
    def test_load_metadata_file_not_exists(self):
        """Test loading metadata when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            result = self.rag_system.load_metadata()
            
            self.assertEqual(result['last_processed'], '1970-01-01T00:00:00.000Z')
            self.assertEqual(result['processed_ids'], set())
    
    def test_save_metadata(self):
        """Test saving metadata to file."""
        test_metadata = {
            'last_processed': '2023-01-01T00:00:00.000Z',
            'processed_ids': ['id1', 'id2']
        }
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.rag_system.save_metadata(test_metadata)
            
            mock_file.assert_called_once()
            call_args = mock_file.call_args
            self.assertEqual(call_args[0][0], '/mnt/vectordb/metadata.json')
            self.assertEqual(call_args[1]['mode'], 'w')
    
    def test_rag_query_time_range(self):
        """Test RAG query with time range filtering."""
        mock_metadata = {
            '1': {
                'id': 'id1',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test message 1',
                'hostname': 'host1'
            },
            '2': {
                'id': 'id2',
                'timestamp': '2023-01-02T12:00:00.000Z',
                'message': 'Test message 2',
                'hostname': 'host2'
            }
        }
        
        with patch.object(self.rag_system, 'metadata', mock_metadata):
            with patch.object(self.rag_system, 'index') as mock_index:
                mock_index.search = Mock(return_value=[[1, 2], [100, 200]])
                
                from datetime import datetime
                time_range = (datetime(2023, 1, 1), datetime(2023, 1, 3))
                
                result = self.rag_system.rag_query('test query', time_range=time_range, k=2)
                
                self.assertEqual(len(result), 2)
    
    def test_rag_query_hostname_filter(self):
        """Test RAG query with hostname pattern filtering."""
        mock_metadata = {
            '1': {
                'id': 'id1',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test message 1',
                'hostname': 'server-01'
            },
            '2': {
                'id': 'id2',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test message 2',
                'hostname': 'server-02'
            }
        }
        
        with patch.object(self.rag_system, 'metadata', mock_metadata):
            with patch.object(self.rag_system, 'index') as mock_index:
                mock_index.search = Mock(return_value=[[1, 2], [100, 200]])
                
                result = self.rag_system.rag_query('test query', hostname_pattern='server-0*', k=2)
                
                self.assertEqual(len(result), 2)
    
    def test_generate_llm_response(self):
        """Test LLM response generation."""
        relevant_logs = [
            {
                'hostname': 'host1',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test log message'
            }
        ]
        
        with patch.object(self.rag_system.openai, 'ChatCompletion') as mock_chat:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='Test response'))]
            mock_chat.create = Mock(return_value=mock_response)
            
            result = self.rag_system.generate_llm_response('test query', relevant_logs)
            
            self.assertEqual(result, 'Test response')
            mock_chat.create.assert_called_once()
    
    def test_api_rag_query_endpoint(self):
        """Test the FastAPI endpoint for RAG queries."""
        from fastapi.testclient import TestClient
        
        client = TestClient(self.rag_system.app)
        
        with patch.object(self.rag_system, 'rag_query') as mock_rag_query:
            mock_rag_query.return_value = [
                {'hostname': 'host1', 'timestamp': '2023-01-01T12:00:00.000Z', 'message': 'Test'}
            ]
            
            with patch.object(self.rag_system, 'generate_llm_response') as mock_llm:
                mock_llm.return_value = 'Generated response'
                
                response = client.post('/rag_query', json={
                    'text': 'test query',
                    'k': 5
                })
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data['answer'], 'Generated response')
                self.assertIn('relevant_logs', data)
    
    def test_api_rag_query_with_time_range(self):
        """Test the FastAPI endpoint with time range."""
        from fastapi.testclient import TestClient
        
        client = TestClient(self.rag_system.app)
        
        with patch.object(self.rag_system, 'rag_query') as mock_rag_query:
            mock_rag_query.return_value = []
            
            with patch.object(self.rag_system, 'generate_llm_response') as mock_llm:
                mock_llm.return_value = 'No logs found'
                
                response = client.post('/rag_query', json={
                    'text': 'test query',
                    'start_time': '2023-01-01T00:00:00.000Z',
                    'end_time': '2023-01-02T00:00:00.000Z'
                })
                
                self.assertEqual(response.status_code, 200)
    
    def test_api_rag_query_error_handling(self):
        """Test error handling in the API endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(self.rag_system.app)
        
        with patch.object(self.rag_system, 'rag_query') as mock_rag_query:
            mock_rag_query.side_effect = Exception('Test error')
            
            response = client.post('/rag_query', json={
                'text': 'test query'
            })
            
            self.assertEqual(response.status_code, 500)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
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
        mock_dateutil.parser.parse = lambda x: Mock(timestamp=lambda: 1234567890.0)
        sys.modules['dateutil'] = mock_dateutil
        sys.modules['dateutil.parser'] = mock_dateutil.parser
        
        import rag_system
        self.rag_system = rag_system
    
    def test_vectorize_logs_structure(self):
        """Test that vectorize logs creates proper structure."""
        log_texts = ['test message 1', 'test message 2']
        timestamps = [0.5, 0.6]
        
        with patch.object(self.rag_system, 'model') as mock_model:
            mock_model.encode = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            
            result = self.rag_system.vectorize_logs(log_texts, timestamps)
            
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0]), 4)  # 3 embedding dims + 1 timestamp


if __name__ == '__main__':
    unittest.main()
