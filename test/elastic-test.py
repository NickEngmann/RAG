#!/usr/bin/env python3
"""
elastic-test.py - Elasticsearch Connection Testing Script

This script tests the connection to Elasticsearch, which is used by the RAG system
for:
- Storing processed log data
- Indexing vector embeddings for semantic search
- Querying logs by timestamp, severity, or content

Usage:
    python test/elastic-test.py

Dependencies:
    - elasticsearch
    - python-dotenv

Environment Variables Required:
    - ELASTICSEARCH_URL: URL of the Elasticsearch instance

Expected Output:
    - Connection status
    - Elasticsearch version information
    - Cluster health status

Integration with RAG System:
    This test verifies the connection before running the main RAG pipeline
    in rag_system.py which indexes log data into Elasticsearch.
"""

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

def test_elasticsearch_connection():
    """
    Test Elasticsearch connection and basic operations.
    
    This function verifies:
    1. Environment variable configuration (ELASTICSEARCH_URL)
    2. Connection establishment to Elasticsearch cluster
    3. Cluster information retrieval
    4. Basic cluster health check
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print("Testing elasticsearch...")
    
    # Load environment variables from .env file
    # This is required for the ELASTICSEARCH_URL configuration
    load_dotenv()
    
    # Get Elasticsearch URL from environment
    es_url = os.getenv('ELASTICSEARCH_URL')
    if not es_url:
        print("Error: ELASTICSEARCH_URL environment variable not set")
        return False
    
    # Initialize Elasticsearch client
    # The RAG system uses this client for indexing and querying logs
    es = Elasticsearch([es_url])
    
    # Check if the cluster is up and responsive
    if es.ping():
        print("Connected to Elasticsearch")
        
        # Retrieve cluster information
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
        
        # Additional cluster info (if available)
        if 'cluster_name' in info:
            print(f"Cluster name: {info.get('cluster_name', 'N/A')}")
        
        return True
    else:
        print("Could not connect to Elasticsearch")
        print("Please ensure:")
        print("  1. Elasticsearch is running")
        print("  2. ELASTICSEARCH_URL is correctly set in .env file")
        print("  3. Network connectivity to the Elasticsearch instance")
        return False

if __name__ == "__main__":
    success = test_elasticsearch_connection()
    import sys
    sys.exit(0 if success else 1)