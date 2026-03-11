"""Test summary endpoint with mocked dependencies."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, '')

# Mock the problematic imports before importing rag_system
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['elasticsearch'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['faiss.contrib'] = MagicMock()
sys.modules['faiss.contrib.indexer'] = MagicMock()
sys.modules['faiss.contrib.indexer_pytorch'] = MagicMock()
sys.modules['faiss.contrib.indexer_mkl'] = MagicMock()
sys.modules['faiss.contrib.indexer_openmp'] = MagicMock()
sys.modules['faiss.contrib.indexer_tbb'] = MagicMock()
sys.modules['faiss.contrib.indexer_cuda'] = MagicMock()
sys.modules['faiss.contrib.indexer_cublas'] = MagicMock()
sys.modules['faiss.contrib.indexer_cublaslt'] = MagicMock()
sys.modules['faiss.contrib.indexer_cudnn'] = MagicMock()
sys.modules['faiss.contrib.indexer_cudnnlt'] = MagicMock()

from rag_system import summarize_documents


class TestSummaryEndpoint(unittest.TestCase):
    """Test the summary endpoint."""
    
    @patch('rag_system.get_embedding_model')
    @patch('rag_system.get_faiss_index')
    @patch('rag_system.get_elasticsearch_client')
    def test_summarize_documents(self, mock_es, mock_faiss, mock_embedding):
        """Test that summarize_documents returns expected output."""
        # Mock the embedding model
        mock_embedding.return_value = MagicMock()
        mock_query_embedding = MagicMock()
        mock_query_embedding.reshape = lambda shape: [[0.1] * 384]
        mock_query_embedding.tolist = lambda: [[0.1] * 384]
        mock_embedding.return_value.encode.return_value = mock_query_embedding
        
        # Mock the faiss index
        mock_faiss.return_value = MagicMock()
        mock_distances = MagicMock()
        mock_distances.reshape = lambda shape: [[0.1, 0.2, 0.3]]
        mock_distances.tolist = lambda: [[0.1, 0.2, 0.3]]
        mock_faiss.return_value.search.return_value = ([0], mock_distances)
        
        # Mock the Elasticsearch client
        mock_es.return_value.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_source': {
                            'text': 'Document 1 content',
                            'metadata': {'source': 'test'}
                        }
                    },
                    {
                        '_source': {
                            'text': 'Document 2 content',
                            'metadata': {'source': 'test'}
                        }
                    }
                ]
            }
        }
        
        # Test the function
        result = summarize_documents(
            query="test query",
            top_k=2,
            use_faiss=True,
            use_elasticsearch=True
        )
        
        # Verify result structure
        self.assertIn('summary', result)
        self.assertIn('sources', result)
        self.assertIn('query', result)
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(len(result['sources']), 2)
        self.assertIn('Document 1 content', result['summary'])
        self.assertIn('Document 2 content', result['summary'])


if __name__ == '__main__':
    unittest.main()
