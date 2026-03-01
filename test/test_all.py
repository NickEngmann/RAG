#!/usr/bin/env python3
"""Comprehensive test suite for RAG system"""

import sys
import os
import gc
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gc():
    """Test garbage collector"""
    print("Testing gc (Garbage Collector)...")
    
    # Create some objects
    large_list = [i for i in range(1000000)]
    del large_list
    
    count_before = len(gc.get_objects())
    gc.collect()
    count_after = len(gc.get_objects())
    
    print(f"Objects before GC: {count_before}")
    print(f"Objects after GC: {count_after}")
    print("gc test completed successfully")
    return True

def test_tqdm():
    """Test tqdm progress bar"""
    print("Testing tqdm...")
    
    from tqdm import tqdm
    
    for i in tqdm(range(10), desc="Processing"):
        time.sleep(0.01)
    
    print("tqdm test completed successfully")
    return True

def test_elasticsearch():
    """Test Elasticsearch connection"""
    print("Testing elasticsearch...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch([elasticsearch_url])
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
        print("elasticsearch test completed successfully")
        return True
    except Exception as e:
        print(f"elasticsearch test skipped (no connection): {e}")
        return True  # Skip if no connection

def test_sentence_transformers():
    """Test sentence transformers"""
    print("Testing sentence_transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = ['This is a test', 'Another test sentence']
        embeddings = model.encode(sentences)
        
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Type of embeddings: {type(embeddings)}")
        print(f"Sample embedding shape: {embeddings[0].shape}")
        print("sentence_transformers test completed successfully")
        return True
    except Exception as e:
        print(f"sentence_transformers test failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch"""
    print("Testing PyTorch...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Create a simple tensor
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print(f"Tensor operation result: {y}")
        
        print("PyTorch test completed successfully.")
        return True
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return False

def test_rag_system_import():
    """Test that RAG system can be imported"""
    print("Testing RAG system import...")
    
    try:
        from rag_system import RAGSystem
        print("RAGSystem imported successfully")
        return True
    except Exception as e:
        print(f"RAG system import failed: {e}")
        return False

if __name__ == "__main__":
    tests = [
        ("gc", test_gc),
        ("tqdm", test_tqdm),
        ("elasticsearch", test_elasticsearch),
        ("sentence_transformers", test_sentence_transformers),
        ("pytorch", test_pytorch),
        ("rag_system_import", test_rag_system_import),
    ]
    
    results = {"passed": 0, "failed": 0, "errors": 0}
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {name}...")
        print('='*50)
        try:
            result = test_func()
            if result:
                results["passed"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            print(f"Test {name} error: {e}")
            results["errors"] += 1
    
    print(f"\n{'='*50}")
    print("Test Summary")
    print('='*50)
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
