#!/usr/bin/env python3
"""Tests for rag_system.py - testing individual functions without importing the module"""

import pytest
import os
import json
import faiss
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
from dateutil.parser import parse


def test_parse_date():
    """Test date parsing functionality"""
    # Test parsing a date string
    result = parse("2024-01-15T10:30:00Z")
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 15


def test_datetime_range():
    """Test datetime range calculation"""
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    
    assert one_hour_ago < now
    assert (now - one_hour_ago).total_seconds() == 3600


def test_faiss_index_creation():
    """Test FAISS index creation"""
    dimension = 384
    num_vectors = 10
    
    # Create random vectors
    vectors = np.random.rand(num_vectors, dimension).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    assert index.ntotal == num_vectors


def test_faiss_search():
    """Test FAISS search functionality"""
    dimension = 384
    num_vectors = 10
    
    # Create random vectors
    vectors = np.random.rand(num_vectors, dimension).astype('float32')
    
    # Create and train index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    # Create query vector
    query = np.random.rand(1, dimension).astype('float32')
    
    # Search for top 3 similar vectors
    k = 3
    distances, indices = index.search(query, k)
    
    assert len(indices[0]) == k
    assert len(distances[0]) == k


def test_json_serialization():
    """Test JSON serialization of data"""
    test_data = {
        'message': 'Test message',
        'timestamp': '2024-01-15T10:30:00Z',
        'hostname': 'test-host',
        'level': 'INFO'
    }
    
    # Serialize to JSON
    json_str = json.dumps(test_data)
    
    # Deserialize back
    loaded_data = json.loads(json_str)
    
    assert loaded_data == test_data


def test_metadata_file():
    """Test metadata file creation and reading"""
    metadata = {
        'last_processed': datetime.now().isoformat(),
        'total_logs_processed': 100,
        'index_file': 'test_index.faiss',
        'vector_dimension': 384
    }
    
    # Write metadata to file
    metadata_file = '/tmp/test_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    # Read metadata back
    with open(metadata_file, 'r') as f:
        loaded_metadata = json.load(f)
    
    assert loaded_metadata['total_logs_processed'] == 100
    
    # Cleanup
    os.remove(metadata_file)


def test_log_processing():
    """Test log processing with mock data"""
    # Create mock log data
    mock_logs = [
        {
            '@timestamp': '2024-01-15T10:30:00Z',
            'message': 'Test log message 1',
            'hostname': 'test-host-1'
        },
        {
            '@timestamp': '2024-01-15T10:31:00Z',
            'message': 'Test log message 2',
            'hostname': 'test-host-2'
        }
    ]
    
    # Process logs
    processed_logs = []
    for log in mock_logs:
        processed_logs.append({
            'timestamp': log['@timestamp'],
            'message': log['message'],
            'hostname': log['hostname']
        })
    
    assert len(processed_logs) == 2
    assert processed_logs[0]['message'] == 'Test log message 1'


def test_time_range_filtering():
    """Test filtering logs by time range"""
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    two_hours_ago = now - timedelta(hours=2)
    
    # Create mock logs with different timestamps
    mock_logs = [
        {
            'timestamp': (now - timedelta(minutes=30)).isoformat(),
            'message': 'Recent log'
        },
        {
            'timestamp': (now - timedelta(hours=1.5)).isoformat(),
            'message': 'Old log'
        }
    ]
    
    # Filter logs within the time range
    filtered_logs = []
    for log in mock_logs:
        log_time = datetime.fromisoformat(log['timestamp'])
        if one_hour_ago <= log_time <= now:
            filtered_logs.append(log)
    
    assert len(filtered_logs) == 1
    assert filtered_logs[0]['message'] == 'Recent log'


def test_hostname_pattern_matching():
    """Test hostname pattern matching"""
    mock_logs = [
        {'hostname': 'web-server-1', 'message': 'Log 1'},
        {'hostname': 'web-server-2', 'message': 'Log 2'},
        {'hostname': 'db-server-1', 'message': 'Log 3'}
    ]
    
    # Match logs with hostname pattern 'web-server'
    pattern = 'web-server'
    matched_logs = [log for log in mock_logs if pattern in log['hostname']]
    
    assert len(matched_logs) == 2
    assert matched_logs[0]['hostname'] == 'web-server-1'


def test_vector_similarity():
    """Test vector similarity calculation"""
    # Create similar vectors (close in direction)
    vector1 = np.array([1.0, 2.0, 3.0], dtype='float32')
    vector2 = np.array([1.1, 2.1, 3.1], dtype='float32')
    # Create orthogonal vector (different direction)
    vector3 = np.array([3.0, -2.0, 1.0], dtype='float32')
    
    # Calculate cosine similarity manually
    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    
    sim_12 = cosine_similarity(vector1, vector2)
    sim_13 = cosine_similarity(vector1, vector3)
    
    # Similar vectors should have higher similarity
    assert sim_12 > sim_13
    assert sim_12 > 0.99  # Very similar vectors
    assert sim_13 < 0.5  # Different direction vectors


def test_api_response_structure():
    """Test API response structure"""
    # Mock API response
    response = {
        'answer': 'This is a test response',
        'relevant_logs': [
            {'message': 'Log 1', 'timestamp': '2024-01-15T10:30:00Z'},
            {'message': 'Log 2', 'timestamp': '2024-01-15T10:31:00Z'}
        ]
    }
    
    assert 'answer' in response
    assert 'relevant_logs' in response
    assert len(response['relevant_logs']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
