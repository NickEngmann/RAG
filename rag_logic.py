#!/usr/bin/env python3
"""Core RAG logic functions without external dependencies.

This module contains the pure logic functions that can be tested
without requiring Elasticsearch, sentence-transformers, or FAISS.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dateutil.parser import parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_log(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a log entry for embedding generation.
    
    Args:
        log_entry: Raw log entry dictionary
        
    Returns:
        Preprocessed log entry with normalized fields
    """
    processed = log_entry.copy()
    
    # Normalize timestamp
    if 'timestamp' in processed:
        try:
            if isinstance(processed['timestamp'], str):
                dt = parse(processed['timestamp'])
                processed['timestamp'] = dt.isoformat()
                processed['timestamp_unix'] = dt.timestamp()
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {e}")
    
    # Normalize level
    if 'level' in processed:
        processed['level'] = processed['level'].upper()
    
    # Normalize message
    if 'message' in processed:
        processed['message'] = str(processed['message']).strip()
    
    # Add processing metadata
    processed['processed_at'] = datetime.now().isoformat()
    processed['preprocessed'] = True
    
    return processed


def generate_embedding_text(log_entry: Dict[str, Any]) -> str:
    """Generate text for embedding from a log entry.
    
    Args:
        log_entry: Preprocessed log entry
        
    Returns:
        Text string for embedding generation
    """
    parts = []
    
    # Include timestamp
    if 'timestamp' in log_entry:
        parts.append(f"Timestamp: {log_entry['timestamp']}")
    
    # Include level
    if 'level' in log_entry:
        parts.append(f"Level: {log_entry['level']}")
    
    # Include message
    if 'message' in log_entry:
        parts.append(f"Message: {log_entry['message']}")
    
    # Include user info if available
    if 'user_id' in log_entry:
        parts.append(f"User: {log_entry['user_id']}")
    
    # Include action if available
    if 'action' in log_entry:
        parts.append(f"Action: {log_entry['action']}")
    
    return " ".join(parts)


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have same length")
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Magnitudes
    mag1 = sum(a * a for a in embedding1) ** 0.5
    mag2 = sum(b * b for b in embedding2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def create_metadata(log_entry: Dict[str, Any], embedding_id: str) -> Dict[str, Any]:
    """Create metadata for a log entry with embedding.
    
    Args:
        log_entry: Preprocessed log entry
        embedding_id: ID of the embedding
        
    Returns:
        Metadata dictionary
    """
    return {
        'id': embedding_id,
        'timestamp': log_entry.get('timestamp', ''),
        'level': log_entry.get('level', ''),
        'message': log_entry.get('message', ''),
        'user_id': log_entry.get('user_id', ''),
        'action': log_entry.get('action', ''),
        'created_at': datetime.now().isoformat()
    }


def save_metadata(metadata: Dict[str, Any], filepath: str = '/tmp/rag_metadata.json') -> bool:
    """Save metadata to a JSON file.
    
    Args:
        metadata: Metadata dictionary to save
        filepath: Path to save metadata file
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return False


def load_metadata(filepath: str = '/tmp/rag_metadata.json') -> Optional[Dict[str, Any]]:
    """Load metadata from a JSON file.
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from {filepath}")
        return metadata
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return None


class TimeScaler:
    """Scale time values for embedding generation."""
    
    def __init__(self, min_time: float = 0.0, max_time: float = 1.0):
        """Initialize time scaler.
        
        Args:
            min_time: Minimum time value
            max_time: Maximum time value
        """
        self.min_time = min_time
        self.max_time = max_time
    
    def scale(self, value: float, min_val: float, max_val: float) -> float:
        """Scale a value to [min_time, max_time] range.
        
        Args:
            value: Value to scale
            min_val: Minimum value of range
            max_val: Maximum value of range
            
        Returns:
            Scaled value
        """
        if max_val == min_val:
            return self.min_time
        
        normalized = (value - min_val) / (max_val - min_val)
        return self.min_time + normalized * (self.max_time - self.min_time)
    
    def scale_timestamp(self, timestamp: str) -> float:
        """Scale a timestamp string to [min_time, max_time] range.
        
        Args:
            timestamp: ISO format timestamp string
            
        Returns:
            Scaled timestamp value
        """
        try:
            dt = parse(timestamp)
            unix_ts = dt.timestamp()
            return self.scale(unix_ts, 0.0, 9999999999.0)  # Far future
        except Exception as e:
            logger.warning(f"Failed to scale timestamp: {e}")
            return 0.5


def create_embedding_index(embeddings: List[List[float]], metadata_list: List[Dict]) -> Dict[str, Any]:
    """Create an embedding index structure.
    
    Args:
        embeddings: List of embedding vectors
        metadata_list: List of metadata dictionaries
        
    Returns:
        Index structure with embeddings and metadata
    """
    if len(embeddings) != len(metadata_list):
        raise ValueError("Embeddings and metadata must have same length")
    
    index = {
        'embeddings': embeddings,
        'metadata': metadata_list,
        'dimension': len(embeddings[0]) if embeddings else 0,
        'count': len(embeddings)
    }
    
    return index


def search_embedding_index(
    index: Dict[str, Any],
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Search embedding index for similar entries.
    
    Args:
        index: Embedding index structure
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of metadata dictionaries sorted by similarity
    """
    if not index.get('embeddings'):
        return []
    
    embeddings = index['embeddings']
    metadata_list = index['metadata']
    
    # Calculate similarity scores
    scores = []
    for i, embedding in enumerate(embeddings):
        similarity = calculate_similarity(query_embedding, embedding)
        scores.append((i, similarity))
    
    # Sort by similarity (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    results = []
    for idx, score in scores[:top_k]:
        result = metadata_list[idx].copy()
        result['similarity'] = score
        results.append(result)
    
    return results


def process_log_batch(
    log_entries: List[Dict[str, Any]],
    time_scaler: TimeScaler
) -> List[Dict[str, Any]]:
    """Process a batch of log entries.
    
    Args:
        log_entries: List of log entries to process
        time_scaler: Time scaler for timestamp normalization
        
    Returns:
        List of processed log entries
    """
    processed = []
    for entry in log_entries:
        preprocessed = preprocess_log(entry)
        processed.append(preprocessed)
    return processed


def generate_sample_embeddings(embedding_size: int = 384) -> List[List[float]]:
    """Generate sample embeddings for testing.
    
    Args:
        embedding_size: Size of each embedding vector
        
    Returns:
        List of sample embedding vectors
    """
    import random
    random.seed(42)  # For reproducibility
    
    embeddings = []
    for _ in range(10):
        embedding = [random.gauss(0, 1) for _ in range(embedding_size)]
        embeddings.append(embedding)
    
    return embeddings


def create_sample_metadata(count: int = 10) -> List[Dict[str, Any]]:
    """Create sample metadata for testing.
    
    Args:
        count: Number of metadata entries to create
        
    Returns:
        List of sample metadata dictionaries
    """
    import random
    random.seed(42)  # For reproducibility
    
    levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    actions = ['login', 'logout', 'view', 'edit', 'delete']
    
    metadata_list = []
    for i in range(count):
        metadata = {
            'id': f'sample_{i}',
            'timestamp': f'2024-01-{i+1:02d}T12:00:00.000Z',
            'level': random.choice(levels),
            'message': f'Sample log message {i}',
            'user_id': f'user_{i}',
            'action': random.choice(actions)
        }
        metadata_list.append(metadata)
    
    return metadata_list
