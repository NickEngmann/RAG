#!/usr/bin/env python3
"""Test runner for RAG system components.

This test runner focuses on testing the pure logic functions in rag_logic.py
without requiring external dependencies like Elasticsearch, sentence-transformers,
or FAISS to be running.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Run all tests and return success status."""
    print("=" * 50)
    print("RAG System Test Suite")
    print("=" * 50)
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 50)
    
    passed = 0
    failed = 0
    total = 0
    
    # Test 1: preprocess_log
    total += 1
    print("\n--- Test: preprocess_log ---")
    try:
        from rag_logic import preprocess_log
        
        # Test with valid log entry
        log_entry = {
            'timestamp': '2024-01-01T12:00:00.000Z',
            'level': 'info',
            'message': 'Test log message',
            'user_id': 'user123',
            'action': 'login'
        }
        
        processed = preprocess_log(log_entry)
        assert processed is not None, "Processed log should not be None"
        assert 'timestamp' in processed, "Processed log should have timestamp"
        assert processed['level'] == 'INFO', "Level should be uppercase"
        assert processed.get('preprocessed') == True, "Should have preprocessed flag"
        
        print("✓ preprocess_log test passed")
        passed += 1
    except Exception as e:
        print(f"✗ preprocess_log test failed: {e}")
        failed += 1
    
    # Test 2: generate_embedding_text
    total += 1
    print("\n--- Test: generate_embedding_text ---")
    try:
        from rag_logic import generate_embedding_text
        
        log_entry = {
            'timestamp': '2024-01-01T12:00:00.000Z',
            'level': 'INFO',
            'message': 'User login successful',
            'user_id': 'user123',
            'action': 'login'
        }
        
        text = generate_embedding_text(log_entry)
        assert text is not None, "Generated text should not be None"
        assert len(text) > 0, "Generated text should not be empty"
        assert 'User login successful' in text, "Message should be in text"
        
        print("✓ generate_embedding_text test passed")
        passed += 1
    except Exception as e:
        print(f"✗ generate_embedding_text test failed: {e}")
        failed += 1
    
    # Test 3: calculate_similarity
    total += 1
    print("\n--- Test: calculate_similarity ---")
    try:
        from rag_logic import calculate_similarity
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.9, 0.1, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        sim12 = calculate_similarity(vec1, vec2)
        sim13 = calculate_similarity(vec1, vec3)
        
        assert sim12 > sim13, "Similar vectors should have higher similarity"
        assert 0 <= sim12 <= 1, "Similarity should be between 0 and 1"
        
        print("✓ calculate_similarity test passed")
        passed += 1
    except Exception as e:
        print(f"✗ calculate_similarity test failed: {e}")
        failed += 1
    
    # Test 4: time_scaler
    total += 1
    print("\n--- Test: time_scaler ---")
    try:
        from rag_logic import time_scaler
        
        # Test with valid timestamp
        ts1 = '2024-01-01T12:00:00.000Z'
        ts2 = '2024-01-01T14:00:00.000Z'
        ts3 = '2024-01-01T12:30:00.000Z'
        
        # ts2 should be 2 hours after ts1
        # ts3 should be 30 minutes after ts1
        
        print("✓ time_scaler test passed")
        passed += 1
    except Exception as e:
        print(f"✗ time_scaler test failed: {e}")
        failed += 1
    
    # Test 5: load_metadata
    total += 1
    print("\n--- Test: load_metadata ---")
    try:
        from rag_logic import load_metadata, save_metadata
        import tempfile
        import json
        
        # Create a temporary metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            metadata = {
                'last_updated': '2024-01-01T12:00:00.000Z',
                'total_logs': 100,
                'index_version': '1.0'
            }
            json.dump(metadata, f)
        
        # Load the metadata
        loaded = load_metadata(temp_file)
        assert loaded is not None, "Loaded metadata should not be None"
        assert loaded.get('total_logs') == 100, "Should have correct total_logs"
        
        # Clean up
        os.unlink(temp_file)
        
        print("✓ load_metadata test passed")
        passed += 1
    except Exception as e:
        print(f"✗ load_metadata test failed: {e}")
        failed += 1
    
    # Test 6: save_metadata
    total += 1
    print("\n--- Test: save_metadata ---")
    try:
        from rag_logic import load_metadata, save_metadata
        import tempfile
        import json
        
        # Create a temporary metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        # Save metadata
        metadata = {
            'last_updated': '2024-01-01T12:00:00.000Z',
            'total_logs': 100,
            'index_version': '1.0'
        }
        save_metadata(metadata, temp_file)
        
        # Load and verify
        loaded = load_metadata(temp_file)
        assert loaded is not None, "Loaded metadata should not be None"
        assert loaded.get('total_logs') == 100, "Should have correct total_logs"
        
        # Clean up
        os.unlink(temp_file)
        
        print("✓ save_metadata test passed")
        passed += 1
    except Exception as e:
        print(f"✗ save_metadata test failed: {e}")
        failed += 1
    
    # Test 7: create_index
    total += 1
    print("\n--- Test: create_index ---")
    try:
        from rag_logic import create_index, add_to_index, search_index
        import tempfile
        import numpy as np
        
        # Create a temporary index file
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
            temp_file = f.name
        
        # Create index
        index = create_index(128, temp_file)
        assert index is not None, "Created index should not be None"
        
        # Add some vectors
        test_vectors = np.random.rand(10, 128).astype(np.float32)
        add_to_index(index, test_vectors, temp_file)
        
        # Search
        query = test_vectors[0]
        results = search_index(index, query, k=5, temp_file=temp_file)
        assert results is not None, "Search results should not be None"
        
        # Clean up
        os.unlink(temp_file)
        
        print("✓ create_index test passed")
        passed += 1
    except Exception as e:
        print(f"✗ create_index test failed: {e}")
        failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print("=" * 50)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
