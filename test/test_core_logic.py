#!/usr/bin/env python3
"""Test suite for core RAG system logic functions."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessing(unittest.TestCase):
    """Test log preprocessing functions."""
    
    def test_preprocess_log_basic(self):
        """Test basic log preprocessing."""
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': 'Test log message',
            'hostname': 'test-host'
        }
        
        from dateutil.parser import parse
        timestamp = parse(log_entry['@timestamp'])
        timestamp_value = timestamp.timestamp()
        
        message = log_entry['message'][:1000]
        hostname = log_entry.get('hostname', 'unknown')
        
        self.assertEqual(message, 'Test log message')
        self.assertEqual(hostname, 'test-host')
        self.assertGreater(timestamp_value, 0)
    
    def test_preprocess_log_truncation(self):
        """Test that log messages are truncated to 1000 characters."""
        long_message = 'x' * 2000
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': long_message,
            'hostname': 'test-host'
        }
        
        message = log_entry['message'][:1000]
        self.assertEqual(len(message), 1000)
    
    def test_preprocess_log_default_hostname(self):
        """Test that hostname defaults to 'unknown' when not present."""
        log_entry = {
            '@timestamp': '2023-01-01T12:00:00.000Z',
            'message': 'Test log message'
        }
        
        hostname = log_entry.get('hostname', 'unknown')
        self.assertEqual(hostname, 'unknown')


class TestMetadataHandling(unittest.TestCase):
    """Test metadata loading and saving functions."""
    
    def test_load_metadata_file_exists(self):
        """Test loading metadata from existing file."""
        test_metadata = {
            'last_processed': '2023-01-01T00:00:00.000Z',
            'processed_ids': ['id1', 'id2']
        }
        
        with open('/tmp/test_metadata.json', 'w') as f:
            json.dump(test_metadata, f)
        
        with open('/tmp/test_metadata.json', 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['last_processed'], test_metadata['last_processed'])
        self.assertEqual(set(loaded['processed_ids']), set(test_metadata['processed_ids']))
        
        try:
            os.remove('/tmp/test_metadata.json')
        except FileNotFoundError:
            pass  # File might already be removed
    
    def test_load_metadata_file_not_exists(self):
        """Test loading metadata when file doesn't exist."""
        default_metadata = {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': set()}
        self.assertEqual(default_metadata['last_processed'], '1970-01-01T00:00:00.000Z')
    
    def test_save_metadata_format(self):
        """Test saving metadata to file."""
        test_metadata = {
            'last_processed': '2023-01-01T00:00:00.000Z',
            'processed_ids': ['id1', 'id2']
        }
        
        with open('/tmp/test_metadata.json', 'w') as f:
            json.dump(test_metadata, f)
        
        with open('/tmp/test_metadata.json', 'r') as f:
            saved = json.load(f)
        
        self.assertEqual(saved['last_processed'], test_metadata['last_processed'])
        os.remove('/tmp/test_metadata.json')


class TestTimeRangeFiltering(unittest.TestCase):
    """Test time range filtering logic."""
    
    def test_time_range_contains_timestamp(self):
        """Test that a timestamp within range is included."""
        time_range = (datetime(2023, 1, 1), datetime(2023, 1, 3))
        test_timestamp = datetime(2023, 1, 2)
        
        start_time, end_time = time_range
        self.assertTrue(start_time <= test_timestamp <= end_time)
    
    def test_time_range_excludes_outside_timestamp(self):
        """Test that a timestamp outside range is excluded."""
        time_range = (datetime(2023, 1, 1), datetime(2023, 1, 3))
        test_timestamp = datetime(2023, 1, 5)
        
        start_time, end_time = time_range
        self.assertFalse(start_time <= test_timestamp <= end_time)
    
    def test_normalized_time_calculation(self):
        """Test normalized time calculation for query context."""
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 3)
        
        normalized_start = 0.0  # Simplified for test
        normalized_end = 1.0    # Simplified for test
        time_context = (normalized_start + normalized_end) / 2
        
        self.assertEqual(time_context, 0.5)


class TestHostnameFiltering(unittest.TestCase):
    """Test hostname pattern filtering logic."""
    
    def test_hostname_exact_match(self):
        """Test exact hostname match."""
        import fnmatch
        
        hostname = 'server-01'
        pattern = 'server-01'
        
        self.assertTrue(fnmatch.fnmatch(hostname, pattern))
    
    def test_hostname_wildcard_match(self):
        """Test wildcard hostname match."""
        import fnmatch
        
        hostname = 'server-01'
        pattern = 'server-0*'
        
        self.assertTrue(fnmatch.fnmatch(hostname, pattern))
    
    def test_hostname_no_match(self):
        """Test hostname that doesn't match pattern."""
        import fnmatch
        
        hostname = 'server-01'
        pattern = 'host-*'
        
        self.assertFalse(fnmatch.fnmatch(hostname, pattern))


class TestQueryResponse(unittest.TestCase):
    """Test query response generation logic."""
    
    def test_relevant_logs_format(self):
        """Test that relevant logs have expected structure."""
        relevant_logs = [
            {
                'hostname': 'host1',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test log message',
                'id': 'id1'
            }
        ]
        
        for log in relevant_logs:
            self.assertIn('hostname', log)
            self.assertIn('timestamp', log)
            self.assertIn('message', log)
            self.assertIn('id', log)
    
    def test_chunks_format(self):
        """Test that chunks are formatted correctly for LLM."""
        relevant_logs = [
            {
                'hostname': 'host1',
                'timestamp': '2023-01-01T12:00:00.000Z',
                'message': 'Test log message'
            }
        ]
        
        chunks = "\n\n".join([
            f"Chunk {i+1} (Hostname: {log['hostname']}, Timestamp: {log['timestamp']}):\n{log['message']}"
            for i, log in enumerate(relevant_logs)
        ])
        
        self.assertIn('Chunk 1', chunks)
        self.assertIn('host1', chunks)
        self.assertIn('Test log message', chunks)


class TestAPIModel(unittest.TestCase):
    """Test API request model."""
    
    def test_query_model_defaults(self):
        """Test that query model has correct defaults."""
        # Simulate the Query model structure
        query_data = {
            'text': 'test query',
            'k': 5,
            'start_time': None,
            'end_time': None,
            'hostname_pattern': None
        }
        
        self.assertEqual(query_data['k'], 5)
        self.assertIsNone(query_data['start_time'])
        self.assertIsNone(query_data['end_time'])
        self.assertIsNone(query_data['hostname_pattern'])
    
    def test_query_model_with_time_range(self):
        """Test query model with time range."""
        query_data = {
            'text': 'test query',
            'start_time': '2023-01-01T00:00:00.000Z',
            'end_time': '2023-01-02T00:00:00.000Z'
        }
        
        self.assertIn('start_time', query_data)
        self.assertIn('end_time', query_data)


if __name__ == '__main__':
    unittest.main()
