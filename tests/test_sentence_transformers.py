import pytest
from sentence_transformers import SentenceTransformer
import numpy as np


def test_sentence_transformers():
    """Test sentence transformers embedding generation."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = ['This is a test sentence.', 'Another sentence for embedding.']

    embeddings = model.encode(sentences)
    
    # Check shape
    assert embeddings.shape == (2, 384), f"Expected shape (2, 384), got {embeddings.shape}"
    
    # Check type
    assert isinstance(embeddings, np.ndarray), f"Expected numpy.ndarray, got {type(embeddings)}"
    
    # Check sample embedding values
    assert len(embeddings[0][:5]) == 5, "Sample embedding should have at least 5 values"
    
    print("sentence_transformers test completed successfully")
