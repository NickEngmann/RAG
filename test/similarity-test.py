#!/usr/bin/env python3
"""
Test script for similarity_search module
"""

import sys
sys.path.insert(0, '')

from similarity_search import (
    tokenize,
    compute_tf,
    compute_idf,
    compute_tf_idf_vector,
    cosine_similarity,
    SimilaritySearch
)


def test_tokenize():
    """Test tokenization function."""
    print("Testing tokenize...")
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    expected = ['hello', 'world', 'this', 'is', 'a', 'test']
    assert tokens == expected, f"Expected {expected}, got {tokens}"
    print(f"  Input: '{text}'")
    print(f"  Tokens: {tokens}")
    print("  ✓ tokenize test passed")


def test_compute_tf():
    """Test term frequency computation."""
    print("\nTesting compute_tf...")
    tokens = ['hello', 'world', 'hello', 'test', 'hello']
    tf = compute_tf(tokens)
    print(f"  Tokens: {tokens}")
    print(f"  TF: {tf}")
    assert tf['hello'] == 0.6, f"Expected 0.6 for 'hello', got {tf['hello']}"
    print("  ✓ compute_tf test passed")


def test_cosine_similarity():
    """Test cosine similarity computation."""
    print("\nTesting cosine_similarity...")
    vec1 = {'a': 1.0, 'b': 2.0, 'c': 0.0}
    vec2 = {'a': 1.0, 'b': 2.0, 'c': 0.0}
    vec3 = {'a': 0.0, 'b': 0.0, 'c': 1.0}
    
    sim1 = cosine_similarity(vec1, vec2)
    sim2 = cosine_similarity(vec1, vec3)
    
    print(f"  Similarity (identical vectors): {sim1:.4f}")
    print(f"  Similarity (orthogonal vectors): {sim2:.4f}")
    
    assert abs(sim1 - 1.0) < 0.001, f"Expected ~1.0, got {sim1}"
    assert abs(sim2 - 0.0) < 0.001, f"Expected ~0.0, got {sim2}"
    print("  ✓ cosine_similarity test passed")


def test_similarity_search():
    """Test the SimilaritySearch class."""
    print("\nTesting SimilaritySearch...")
    
    search = SimilaritySearch()
    
    # Add documents
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language"
    ]
    
    for doc in docs:
        search.add_document(doc)
    
    search.build_index()
    
    # Test search
    query = "machine learning python"
    results = search.search(query, top_k=2)
    
    print(f"  Query: '{query}'")
    print(f"  Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.3f}] {doc}")
    
    assert len(results) <= 2, "Should return at most 2 results"
    assert all(score >= 0 for _, score in results), "Scores should be non-negative"
    print("  ✓ SimilaritySearch test passed")


def test_best_match():
    """Test get_best_match function."""
    print("\nTesting get_best_match...")
    
    search = SimilaritySearch()
    search.add_document("The cat sat on the mat")
    search.add_document("The dog chased the ball")
    search.add_document("Machine learning algorithms")
    search.build_index()
    
    query = "cat sitting"
    best_match, score = search.get_best_match(query)
    
    print(f"  Query: '{query}'")
    print(f"  Best match: '{best_match}' (score: {score:.3f})")
    
    assert "cat" in best_match.lower(), "Best match should contain 'cat'"
    assert score > 0, "Score should be positive"
    print("  ✓ get_best_match test passed")


def run_all_tests():
    """Run all similarity search tests."""
    print("=" * 60)
    print("Running Similarity Search Tests")
    print("=" * 60)
    
    tests = [
        test_tokenize,
        test_compute_tf,
        test_cosine_similarity,
        test_similarity_search,
        test_best_match
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
