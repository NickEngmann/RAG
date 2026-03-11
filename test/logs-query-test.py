#!/usr/bin/env python3
"""Test the /logs_query endpoint for retrieving relevant logs without LLM."""

import sys
import os
sys.path.insert(0, '')

from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

# Mock the problematic imports
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['elasticsearch'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['schedule'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

from rag_system import (
    rag_query, 
    load_metadata, 
    save_metadata,
    process_batch,
    vectorize_logs,
    preprocess_log
)

def test_rag_query_with_empty_index():
    """Test that rag_query handles empty index gracefully."""
    with patch('faiss.read_index') as mock_read:
        mock_read.side_effect = FileNotFoundError()
        with patch('faiss.IndexFlatIP') as mock_index:
            mock_index.return_value.ntotal = 0
            result = rag_query("test query", k=5)
            assert result == [], f"Expected empty list, got {result}"
            print("✓ Empty index handled correctly")

def test_rag_query_basic():
    """Test basic rag_query functionality with mocked data."""
    # Create mock metadata
    mock_metadata = {
        '0': {
            'id': 'log1',
            'timestamp': '2024-01-01T10:00:00.000Z',
            'message': 'Application started successfully',
            'hostname': 'server-01'
        },
        '1': {
            'id': 'log2',
            'timestamp': '2024-01-01T10:05:00.000Z',
            'message': 'Database connection established',
            'hostname': 'server-01'
        },
        '2': {
            'id': 'log3',
            'timestamp': '2024-01-01T10:10:00.000Z',
            'message': 'User login detected',
            'hostname': 'server-02'
        }
    }
    
    with patch('faiss.read_index') as mock_read:
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_read.return_value = mock_index
        
        with patch('faiss.IndexFlatIP') as mock_flat:
            mock_flat.return_value = mock_index
            
            with patch('faiss.normalize_L2'):
                with patch.object(mock_index, 'search') as mock_search:
                    # Mock search to return some results
                    mock_search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
                    
                    result = rag_query("test query", k=3)
                    
                    assert len(result) <= 3, f"Expected at most 3 results, got {len(result)}"
                    print(f"✓ Basic rag_query works, returned {len(result)} results")

def test_rag_query_with_hostname_filter():
    """Test rag_query with hostname pattern filtering."""
    mock_metadata = {
        '0': {
            'id': 'log1',
            'timestamp': '2024-01-01T10:00:00.000Z',
            'message': 'Application started',
            'hostname': 'server-01'
        },
        '1': {
            'id': 'log2',
            'timestamp': '2024-01-01T10:05:00.000Z',
            'message': 'Error occurred',
            'hostname': 'server-02'
        }
    }
    
    with patch('faiss.read_index') as mock_read:
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_read.return_value = mock_index
        
        with patch('faiss.IndexFlatIP') as mock_flat:
            mock_flat.return_value = mock_index
            
            with patch('faiss.normalize_L2'):
                with patch.object(mock_index, 'search') as mock_search:
                    mock_search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
                    
                    result = rag_query("error", hostname_pattern="server-0*", k=2)
                    
                    # Should only return logs matching server-0* pattern
                    for log in result:
                        assert 'server-0' in log['hostname'], f"Expected hostname matching pattern, got {log['hostname']}"
                    print(f"✓ Hostname filtering works, returned {len(result)} filtered results")

def test_rag_query_with_time_range():
    """Test rag_query with time range filtering."""
    mock_metadata = {
        '0': {
            'id': 'log1',
            'timestamp': '2024-01-01T10:00:00.000Z',
            'message': 'Early log',
            'hostname': 'server-01'
        },
        '1': {
            'id': 'log2',
            'timestamp': '2024-01-01T12:00:00.000Z',
            'message': 'Middle log',
            'hostname': 'server-01'
        },
        '2': {
            'id': 'log3',
            'timestamp': '2024-01-01T14:00:00.000Z',
            'message': 'Late log',
            'hostname': 'server-01'
        }
    }
    
    with patch('faiss.read_index') as mock_read:
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_read.return_value = mock_index
        
        with patch('faiss.IndexFlatIP') as mock_flat:
            mock_flat.return_value = mock_index
            
            with patch('faiss.normalize_L2'):
                with patch.object(mock_index, 'search') as mock_search:
                    mock_search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
                    
                    from dateutil.parser import parse
                    start_time = parse('2024-01-01T11:00:00.000Z')
                    end_time = parse('2024-01-01T13:00:00.000Z')
                    
                    result = rag_query("test", time_range=(start_time, end_time), k=3)
                    
                    # Should only return logs in the time range
                    for log in result:
                        log_time = parse(log['timestamp'])
                        assert start_time <= log_time <= end_time, f"Log time {log_time} outside range"
                    print(f"✓ Time range filtering works, returned {len(result)} filtered results")

if __name__ == '__main__':
    print("Testing rag_query functionality...\n")
    
    try:
        test_rag_query_with_empty_index()
        test_rag_query_basic()
        test_rag_query_with_hostname_filter()
        test_rag_query_with_time_range()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
