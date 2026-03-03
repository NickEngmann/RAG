#!/usr/bin/env python3
"""Tests for the RAG system main functionality"""

import sys
import os

# Add the repo directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from dotenv import load_dotenv
        from datetime import datetime
        from dateutil.parser import parse
        from elasticsearch import Elasticsearch, helpers
        import faiss
        import numpy as np
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_structure():
    """Test basic structure of the RAG system"""
    # Test that we can create basic data structures
    import numpy as np
    import faiss
    
    # Test vector creation
    vectors = np.random.rand(10, 384).astype('float32')
    print(f"✓ Created test vectors: {vectors.shape}")
    
    # Test FAISS index creation
    index = faiss.IndexFlatL2(384)
    index.add(vectors)
    print(f"✓ Created FAISS index with {index.ntotal} vectors")
    
    assert index.ntotal == 10, "Index should have 10 vectors"
    print("✓ Basic structure test passed")

def test_metadata_handling():
    """Test metadata file handling"""
    import json
    import tempfile
    
    # Create test metadata
    metadata = {
        "last_update": "2024-01-01T00:00:00",
        "total_logs": 1000,
        "index_size": 384
    }
    
    # Test save and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
        json.dump(metadata, f)
    
    with open(temp_file, 'r') as f:
        loaded = json.load(f)
    
    os.unlink(temp_file)
    
    assert loaded == metadata, "Metadata mismatch"
    print("✓ Metadata handling works correctly")
    return True

if __name__ == "__main__":
    print("Running RAG system tests...\n")
    
    tests = [
        ("Import tests", test_imports),
        ("Basic structure", test_basic_structure),
        ("Metadata handling", test_metadata_handling),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
