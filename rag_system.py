#!/usr/bin/env python3
"""
RAG System for log processing and querying.

This module provides:
- ElasticsearchConnector: Manages Elasticsearch connections
- TimeScaler: Handles time normalization for log processing
- RAGSystem: Main RAG system for log querying and LLM responses
- preprocess_log: Preprocesses log entries
- load_metadata: Loads metadata from file
- save_metadata: Saves metadata to file
"""

import os
import json
import logging
import threading
import queue
from datetime import datetime
from dateutil.parser import parse
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Lazy imports - only imported when needed to avoid module-level execution issues
_sentencetransformer = None
_faiss = None
_elasticsearch = None
_openai = None


def _get_sentence_transformer():
    """Lazy import of sentence_transformers."""
    global _sentencetransformer
    if _sentencetransformer is None:
        from sentence_transformers import SentenceTransformer
        _sentencetransformer = SentenceTransformer
    return _sentencetransformer


def _get_faiss():
    """Lazy import of faiss."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_elasticsearch():
    """Lazy import of elasticsearch."""
    global _elasticsearch
    if _elasticsearch is None:
        from elasticsearch import Elasticsearch, helpers
        _elasticsearch = Elasticsearch
        _helpers = helpers
    return _elasticsearch, _helpers


def _get_openai():
    """Lazy import of openai."""
    global _openai
    if _openai is None:
        import openai
        _openai = openai
    return _openai


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeScaler:
    """Handles time normalization for log processing using MinMaxScaler."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self._fitted = False
        self._min_time = None
        self._max_time = None
    
    def fit_transform(self, timestamps: List[float]) -> np.ndarray:
        """Fit the scaler and transform timestamps."""
        if not timestamps:
            return np.array([])
        
        timestamps = np.array(timestamps).reshape(-1, 1)
        self._min_time = timestamps.min()
        self._max_time = timestamps.max()
        
        if self._max_time == self._min_time:
            return np.zeros_like(timestamps)
        
        self.scaler.fit(timestamps)
        self._fitted = True
        return self.scaler.transform(timestamps).flatten()
    
    def transform(self, timestamps: List[float]) -> np.ndarray:
        """Transform timestamps using fitted scaler."""
        if not self._fitted:
            raise ValueError("TimeScaler must be fitted before transform")
        
        if not timestamps:
            return np.array([])
        
        timestamps = np.array(timestamps).reshape(-1, 1)
        return self.scaler.transform(timestamps).flatten()
    
    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Inverse transform normalized values back to original timestamps."""
        if not self._fitted:
            raise ValueError("TimeScaler must be fitted before inverse_transform")
        
        if len(normalized) == 0:
            return np.array([])
        
        normalized = normalized.reshape(-1, 1)
        return self.scaler.inverse_transform(normalized).flatten()
    
    def get_time_range(self) -> Tuple[float, float]:
        """Return the min and max time values."""
        return (self._min_time, self._max_time)


class TimeNormalizer:
    """Normalizes timestamps for time-based processing."""
    
    def __init__(self):
        self.scaler = TimeScaler()
        self._normalized_timestamps = {}
    
    def normalize(self, timestamp_str: str) -> float:
        """Normalize a timestamp string to a normalized value."""
        try:
            dt = parse(timestamp_str)
            timestamp = dt.timestamp()
            return timestamp
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
            return datetime.now().timestamp()
    
    def fit_on_logs(self, logs: List[Dict[str, Any]]):
        """Fit the normalizer on a list of log entries."""
        timestamps = []
        for log in logs:
            if 'timestamp' in log:
                ts = self.normalize(log['timestamp'])
                timestamps.append(ts)
        
        if timestamps:
            self.scaler.fit_transform(timestamps)
    
    def get_normalized(self, timestamp_str: str) -> float:
        """Get normalized timestamp value."""
        ts = self.normalize(timestamp_str)
        if self.scaler._fitted:
            return self.scaler.transform([ts])[0]
        return ts


class ElasticsearchConnector:
    """Manages Elasticsearch connections and operations."""
    
    def __init__(self, hosts: List[str] = None, index_name: str = "logs"):
        self.index_name = index_name
        self._es = None
        self._hosts = hosts or ["http://localhost:9200"]
    
    @property
    def es(self):
        """Lazy initialization of Elasticsearch client."""
        if self._es is None:
            es_class, _ = _get_elasticsearch()
            self._es = es_class(hosts=self._hosts)
        return self._es
    
    def search(self, query: Dict[str, Any], size: int = 100) -> List[Dict[str, Any]]:
        """Search Elasticsearch for logs."""
        try:
            response = self.es.search(index=self.index_name, body=query, size=size)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Bulk index documents into Elasticsearch."""
        try:
            es_class, helpers = _get_elasticsearch()
            actions = [
                {
                    '_index': self.index_name,
                    '_source': doc
                }
                for doc in documents
            ]
            helpers.bulk(self.es, actions)
            return True
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return False
    
    def close(self):
        """Close the Elasticsearch connection."""
        if self._es:
            self._es.close()
            self._es = None


class SentenceTransformerEmbedder:
    """Handles embedding generation using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy initialization of sentence transformer model."""
        if self._model is None:
            st = _get_sentence_transformer()
            self._model = st(self.model_name)
        return self._model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return mock embeddings if model fails
            return [[0.0] * 384 for _ in texts]
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else [0.0] * 384


class EmbeddingGenerator:
    """Generates embeddings for log entries."""
    
    def __init__(self, embedder: SentenceTransformerEmbedder = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
    
    def generate(self, logs: List[Dict[str, Any]], text_field: str = "message") -> List[List[float]]:
        """Generate embeddings for a list of logs."""
        texts = [log.get(text_field, "") for log in logs]
        return self.embedder.embed(texts)


class MetadataManager:
    """Manages metadata for the RAG system."""
    
    def __init__(self, metadata_file: str = "metadata.json"):
        self.metadata_file = metadata_file
        self._metadata = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Lazy loading of metadata."""
        if self._metadata is None:
            self._metadata = load_metadata(self.metadata_file)
        return self._metadata
    
    def save(self):
        """Save metadata to file."""
        save_metadata(self._metadata, self.metadata_file)
    
    def update(self, updates: Dict[str, Any]):
        """Update metadata with new values."""
        if self._metadata is None:
            self._metadata = {}
        self._metadata.update(updates)


class LogProcessor:
    """Processes log entries for the RAG system."""
    
    def __init__(self, embedder: EmbeddingGenerator = None):
        self.embedder = embedder or EmbeddingGenerator()
        self._processed_logs = []
    
    def process(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of logs."""
        processed = []
        for log in logs:
            processed_log = preprocess_log(log)
            processed.append(processed_log)
        self._processed_logs.extend(processed)
        return processed
    
    def get_embeddings(self, logs: List[Dict[str, Any]]) -> List[List[float]]:
        """Get embeddings for logs."""
        return self.embedder.generate(logs)


class RAGSystem:
    """Main RAG system for log querying and LLM responses."""
    
    def __init__(self, 
                 es_connector: ElasticsearchConnector = None,
                 embedder: SentenceTransformerEmbedder = None,
                 time_scaler: TimeScaler = None):
        self.es_connector = es_connector or ElasticsearchConnector()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.time_scaler = time_scaler or TimeScaler()
        self._metadata = None
        self._index = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Lazy loading of metadata."""
        if self._metadata is None:
            self._metadata = load_metadata()
        return self._metadata
    
    def query(self, query_text: str, k: int = 5, time_range: Optional[Tuple[datetime, datetime]] = None,
              hostname_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the RAG system for relevant logs."""
        # Get query embedding
        query_embedding = self.embedder.embed_single(query_text)
        
        # Search for similar logs
        relevant_logs = self._search_similar_logs(query_embedding, k, time_range, hostname_pattern)
        
        return relevant_logs
    
    def _search_similar_logs(self, query_embedding: List[float], k: int,
                             time_range: Optional[Tuple[datetime, datetime]],
                             hostname_pattern: Optional[str]) -> List[Dict[str, Any]]:
        """Search for logs similar to the query embedding."""
        if not self.metadata or 'index' not in self.metadata:
            return []
        
        try:
            import faiss
            index = faiss.read_index(self.metadata['index_path'])
            
            # Search for similar vectors
            distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
            
            # Get the log entries
            relevant_logs = []
            for idx in indices[0]:
                if idx < len(self.metadata.get('logs', [])):
                    log = self.metadata['logs'][idx]
                    
                    # Apply time range filter
                    if time_range:
                        log_time = parse(log.get('timestamp', ''))
                        if not (time_range[0] <= log_time <= time_range[1]):
                            continue
                    
                    # Apply hostname filter
                    if hostname_pattern:
                        if not fnmatch.fnmatch(log.get('hostname', ''), hostname_pattern):
                            continue
                    
                    relevant_logs.append(log)
                    if len(relevant_logs) >= k:
                        break
            
            return relevant_logs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def generate_response(self, query: str, relevant_logs: List[Dict[str, Any]]) -> str:
        """Generate an LLM response based on relevant logs."""
        try:
            openai = _get_openai()
            
            system_prompt = """
            You are an AI research assistant analyzing log entries to answer queries accurately and concisely.

            Key instructions:
            1. Use ONLY the information contained in the provided log entries to formulate your response.
            2. If no log entries are provided, or if they contain no relevant information to the query, respond with "I don't know".
            3. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the logs.
            4. Do mention or reference the sources of the information.

            Guidelines for responses:
            1. Provide concise, relevant answers that directly address the query.
            2. Synthesize information from multiple logs if applicable.
            3. Maintain a professional and objective tone.
            4. If the information in the logs is insufficient or contradictory, state this clearly.
            5. If a query is ambiguous, respond based solely on the most likely interpretation given the available logs.
            """
            
            chunks = "\n\n".join([
                f"Log {i+1} (Hostname: {log.get('hostname', 'unknown')}, Timestamp: {log.get('timestamp', 'unknown')}):\n{log.get('message', '')}"
                for i, log in enumerate(relevant_logs)
            ])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nRelevant log entries:\n{chunks}"}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500
            )
            
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return f"Error generating response: {str(e)}"


def preprocess_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a single log entry."""
    processed = log.copy()
    
    # Normalize timestamp
    if 'timestamp' in processed:
        try:
            dt = parse(processed['timestamp'])
            processed['timestamp_normalized'] = dt.timestamp()
        except Exception:
            processed['timestamp_normalized'] = datetime.now().timestamp()
    
    # Extract hostname
    if 'hostname' not in processed:
        processed['hostname'] = 'unknown'
    
    # Extract message
    if 'message' not in processed:
        processed['message'] = ''
    
    return processed


def load_metadata(metadata_file: str = "metadata.json") -> Dict[str, Any]:
    """Load metadata from a JSON file."""
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
    return {}


def save_metadata(metadata: Dict[str, Any], metadata_file: str = "metadata.json"):
    """Save metadata to a JSON file."""
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def fnmatch(pattern: str, string: str) -> bool:
    """Simple fnmatch implementation for hostname pattern matching."""
    import fnmatch as fn
    return fn.fnmatch(string, pattern)


# Module-level variables for backward compatibility
metadata = {}
index = None


def get_metadata() -> Dict[str, Any]:
    """Get metadata dictionary."""
    global metadata
    if not metadata:
        metadata = load_metadata()
    return metadata


def get_index() -> Any:
    """Get FAISS index."""
    global index
    if index is None:
        try:
            import faiss
            metadata_dict = get_metadata()
            if 'index_path' in metadata_dict:
                index = faiss.read_index(metadata_dict['index_path'])
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
    return index


if __name__ == "__main__":
    # Example usage
    print("RAG System Module")
    print("Available classes:")
    print("  - TimeScaler")
    print("  - TimeNormalizer")
    print("  - ElasticsearchConnector")
    print("  - SentenceTransformerEmbedder")
    print("  - EmbeddingGenerator")
    print("  - MetadataManager")
    print("  - LogProcessor")
    print("  - RAGSystem")
    print("\nAvailable functions:")
    print("  - preprocess_log")
    print("  - load_metadata")
    print("  - save_metadata")
