import pytest
import numpy as np

def test_sentence_transformers():
    """Test that sentence_transformers library is installed and working."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Testing sentence_transformers...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = ['This is a test sentence.', 'Another sentence for embedding.']
        
        embeddings = model.encode(sentences)
        print(f"Shape of embeddings: {embeddings.shape}")
        print(f"Type of embeddings: {type(embeddings)}")
        
        # Verify embeddings have expected shape
        assert embeddings.shape == (2, 384), f"Expected shape (2, 384), got {embeddings.shape}"
        
        # Test cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        print(f"Cosine similarity between sentences: {similarity:.4f}")
        
        # Similarity should be between -1 and 1
        assert -1 <= similarity <= 1, f"Cosine similarity out of range: {similarity}"
        
        print("sentence_transformers test completed successfully")
    except ImportError as e:
        pytest.skip(f"sentence_transformers not installed: {str(e)}")
    except Exception as e:
        pytest.fail(f"sentence_transformers test failed: {str(e)}")