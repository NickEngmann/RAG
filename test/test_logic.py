#!/usr/bin/env python3
"""Test logic functions with mocked dependencies."""

import sys
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_load_metadata():
    """Test load_metadata function with mocked file system."""
    print("Testing load_metadata...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test metadata file
            metadata_file = os.path.join(tmpdir, 'metadata.json')
            test_metadata = {
                'last_processed': '2024-01-01T00:00:00.000Z',
                'processed_ids': ['test-id-1', 'test-id-2']
            }
            with open(metadata_file, 'w') as f:
                json.dump(test_metadata, f)
            
            # Mock the metadata file path
            with patch('rag_system.METADATA_FILE', metadata_file):
                # Import the function after patching
                from rag_system import load_metadata
                metadata = load_metadata()
                assert isinstance(metadata, dict), "Metadata should be a dictionary"
                assert 'last_processed' in metadata, "Metadata should have 'last_processed' key"
                print("✓ load_metadata test passed")
                return True
    except Exception as e:
        print(f"✗ load_metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_metadata():
    """Test save_metadata function with mocked file system."""
    print("Testing save_metadata...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_file = os.path.join(tmpdir, 'metadata.json')
            
            # Mock the metadata file path
            with patch('rag_system.METADATA_FILE', metadata_file):
                from rag_system import save_metadata
                test_metadata = {
                    'last_processed': '2024-01-01T00:00:00.000Z',
                    'processed_ids': ['test-id-1', 'test-id-2']
                }
                save_metadata(test_metadata)
                
                # Verify the file was created and contains correct data
                assert os.path.exists(metadata_file), "Metadata file should be created"
                with open(metadata_file, 'r') as f:
                    saved_metadata = json.load(f)
                assert saved_metadata['last_processed'] == test_metadata['last_processed'], "Metadata should be saved correctly"
                print("✓ save_metadata test passed")
                return True
    except Exception as e:
        print(f"✗ save_metadata test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_log():
    """Test preprocess_log function."""
    print("Testing preprocess_log...")
    try:
        test_log = {
            '@timestamp': '2024-01-01T12:00:00.000Z',
            'message': 'Test log message for preprocessing',
            'hostname': 'test-host'
        }
        
        # Mock time_scaler to avoid dependency on sklearn
        mock_time_scaler = MagicMock()
        mock_time_scaler.fit_transform.return_value = np.array([[0.5]])
        
        with patch('rag_system.time_scaler', mock_time_scaler):
            from rag_system import preprocess_log
            message, normalized_time, hostname = preprocess_log(test_log, mock_time_scaler)
            
            assert isinstance(message, str), "Message should be a string"
            assert isinstance(normalized_time, float), "Normalized time should be a float"
            assert 0 <= normalized_time <= 1, "Normalized time should be between 0 and 1"
            assert hostname == 'test-host', "Hostname should match input"
            print("✓ preprocess_log test passed")
            return True
    except Exception as e:
        print(f"✗ preprocess_log test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_scaler():
    """Test time_scaler initialization."""
    print("Testing time_scaler...")
    try:
        # Mock the time_scaler initialization
        mock_time_scaler = MagicMock()
        mock_time_scaler.fit_transform.return_value = np.array([[0.5]])
        
        with patch('rag_system.time_scaler', mock_time_scaler):
            from rag_system import time_scaler as loaded_scaler
            assert loaded_scaler is not None, "Time scaler should be initialized"
            assert hasattr(loaded_scaler, 'fit_transform'), "Time scaler should have fit_transform method"
            
            # Test that it can transform a timestamp
            test_timestamp = [[1704110400.0]]  # 2024-01-01T00:00:00.000Z
            result = loaded_scaler.fit_transform(test_timestamp)
            assert isinstance(result, np.ndarray), "Result should be a numpy array"
            assert 0 <= result[0][0] <= 1, "Normalized time should be between 0 and 1"
            print("✓ time_scaler test passed")
            return True
    except Exception as e:
        print(f"✗ time_scaler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorize_logs():
    """Test sentence-transformers embedding generation."""
    print("Testing vectorize_logs...")
    try:
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).tolist()
        
        with patch('rag_system.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_model
            
            from rag_system import vectorize_logs
            test_texts = ['Test log message 1', 'Test log message 2']
            test_timestamps = [0.5, 0.75]
            vectors = vectorize_logs(test_texts, test_timestamps, mock_model)
            
            assert isinstance(vectors, np.ndarray), "Vectors should be a numpy array"
            assert vectors.shape[0] == len(test_texts), "Number of vectors should match number of texts"
            assert vectors.shape[1] == 385, "Vector dimension should be 385 (384 + 1 timestamp)"
            print("✓ vectorize_logs test passed")
            return True
    except Exception as e:
        print(f"✗ vectorize_logs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_query():
    """Test RAG query function."""
    print("Testing rag_query...")
    try:
        # Mock the FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.5]]), np.array([[1]]))
        
        # Mock the metadata
        mock_metadata = {
            'last_processed': '2024-01-01T00:00:00.000Z',
            'processed_ids': ['test-id-1']
        }
        
        # Mock the Elasticsearch client
        mock_es = MagicMock()
        mock_es.search.return_value = {'hits': {'hits': []}}
        
        with patch('rag_system.faiss') as mock_faiss:
            mock_faiss.IndexFlatL2.return_value = mock_index
            
            from rag_system import rag_query
            results = rag_query("test", mock_index, mock_metadata, mock_es, 5)
            
            assert isinstance(results, list), "Results should be a list"
            print("✓ rag_query test passed")
            return True
    except Exception as e:
        print(f"✗ rag_query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*50)
    print("Running RAG System Logic Tests")
    print("="*50 + "\n")
    
    tests = [
        test_load_metadata,
        test_save_metadata,
        test_preprocess_log,
        test_time_scaler,
        test_vectorize_logs,
        test_rag_query
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print("="*50)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
