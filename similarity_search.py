#!/usr/bin/env python3
"""
Similarity Search Module
Implements text similarity search using simple token-based matching.
This is a lightweight alternative to sentence-transformers for environments
where heavy dependencies are not available.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency (TF) for a list of tokens."""
    if not tokens:
        return {}
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    return {word: count / total_tokens for word, count in token_counts.items()}


def compute_idf(documents: List[str], vocabulary: set) -> Dict[str, float]:
    """Compute inverse document frequency (IDF) for terms."""
    idf = {}
    n_docs = len(documents)
    for term in vocabulary:
        doc_count = sum(1 for doc in documents if term in tokenize(doc))
        idf[term] = math.log((n_docs + 1) / (doc_count + 1)) + 1
    return idf


def compute_tf_idf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF vector for a token list."""
    tf = compute_tf(tokens)
    return {term: tf.get(term, 0) * idf.get(term, 0) for term in tf}


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Get all unique terms
    all_terms = set(vec1.keys()) | set(vec2.keys())
    
    # Compute dot product and magnitudes
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


class SimilaritySearch:
    """Lightweight text similarity search using TF-IDF."""
    
    def __init__(self):
        self.documents: List[str] = []
        self.document_vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.vocabulary: set = set()
    
    def add_document(self, text: str):
        """Add a document to the search index."""
        self.documents.append(text)
        tokens = tokenize(text)
        self.vocabulary.update(tokens)
    
    def build_index(self):
        """Build the search index from all documents."""
        if not self.documents:
            return
        
        # Compute IDF for all terms
        self.idf = compute_idf(self.documents, self.vocabulary)
        
        # Compute TF-IDF vectors for all documents
        self.document_vectors = [
            compute_tf_idf_vector(tokenize(doc), self.idf)
            for doc in self.documents
        ]
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for documents similar to the query."""
        if not self.documents:
            return []
        
        # Build index if not already built
        if not self.document_vectors:
            self.build_index()
        
        # Compute query vector
        query_tokens = tokenize(query)
        query_vector = compute_tf_idf_vector(query_tokens, self.idf)
        
        # Compute similarities
        similarities = []
        for i, doc_vector in enumerate(self.document_vectors):
            sim = cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], sim))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_best_match(self, query: str) -> Optional[Tuple[str, float]]:
        """Get the best matching document for a query."""
        results = self.search(query, top_k=1)
        return results[0] if results else None


def demo():
    """Demonstrate the similarity search functionality."""
    print("=" * 60)
    print("Similarity Search Demo")
    print("=" * 60)
    
    # Create search index
    search = SimilaritySearch()
    
    # Add sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Deep learning uses neural networks with many layers",
        "Natural language processing helps computers understand text",
        "The cat sat on the mat near the window",
        "Programming languages enable software development",
        "Data science involves statistics and machine learning"
    ]
    
    for doc in documents:
        search.add_document(doc)
    
    # Build the index
    search.build_index()
    
    # Test queries
    queries = [
        "machine learning python",
        "cat dog animal",
        "programming software development"
    ]
    
    print("\nSearch Results:")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = search.search(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {doc}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
