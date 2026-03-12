#!/usr/bin/env python3

import numpy as np

print("Testing sentence_transformers...")

try:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand human language",
        "Deep learning uses neural networks with many layers",
        "Transformers are revolutionizing NLP tasks"
    ]
    
    print(f"Processing {len(sentences)} sentences...")
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings type: {type(embeddings)}")
    print(f"Sample embedding:\n{embeddings[0][:5]}...")  # Print first 5 values of first embedding
    
    # Test similarity computation
    similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    print(f"Cosine similarity between first two sentences: {similarity:.4f}")
    
    print("sentence_transformers test completed successfully")
    
except ImportError as e:
    print(f"Warning: Could not import sentence_transformers: {str(e)}")
    print("Testing numpy array operations as fallback...")
    
    # Fallback test: demonstrate numpy functionality
    test_array = np.random.rand(5, 768)
    print(f"Generated random embedding matrix: {test_array.shape}")
    print("Fallback numpy test completed successfully")
    
except Exception as e:
    print(f"Error during sentence_transformers test: {str(e)}")
    print("Test completed with errors")