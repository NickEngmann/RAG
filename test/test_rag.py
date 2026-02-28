#!/usr/bin/env python3
"""Test suite for RAG system"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required imports work"""
    from rag_system import (
        Elasticsearch,
        SentenceTransformer,
        load_dotenv,
        os,
        datetime,
        parse,
        helpers,
        faiss,
        numpy,
        logging,
        json,
        tqdm
    )
    print("✓ All imports successful")

def test_environment():
    """Test environment variables are loaded"""
    try:
        from rag_system import load_dotenv
        load_dotenv()
        
        # Check required environment variables exist
        required_vars = [
            'ELASTICSEARCH_HOST',
            'ELASTICSEARCH_PORT',
            'ELASTICSEARCH_USER',
            'ELASTICSEARCH_PASSWORD'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"⚠ Missing environment variables: {missing}")
        else:
            print("✓ Environment variables loaded")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_sentence_transformer():
    """Test sentence transformer model"""
    try:
        from rag_system import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = ['test sentence', 'another test']
        embeddings = model.encode(sentences)
        assert embeddings.shape[0] == 2, "Should have 2 embeddings"
        print("✓ Sentence transformer works")
        return True
    except Exception as e:
        print(f"✗ Sentence transformer test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_imports,
        test_environment,
        test_sentence_transformer
    ]
    
    results = {'passed': 0, 'failed': 0, 'errors': 0}
    
    for test in tests:
        try:
            if test():
                results['passed'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            results['errors'] += 1
    
    print(f"\nResults: {results['passed']} passed, {results['failed']} failed, {results['errors']} errors")
    return results

if __name__ == "__main__":
    run_all_tests()
