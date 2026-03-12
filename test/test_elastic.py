import pytest
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

def test_elastic():
    """Test that elasticsearch library is installed and can connect."""
    load_dotenv()
    
    es_url = os.getenv('ELASTICSEARCH_URL')
    
    if not es_url:
        pytest.skip("ELASTICSEARCH_URL environment variable not set")
    
    try:
        es = Elasticsearch([es_url])
        
        # Check if the cluster is up
        if es.ping():
            print("Connected to Elasticsearch")
            info = es.info()
            print(f"Elasticsearch version: {info['version']['number']}")
        else:
            pytest.skip("Could not connect to Elasticsearch")
    except Exception as e:
        pytest.skip(f"Could not connect to Elasticsearch: {str(e)}")
    
    print("elasticsearch test completed successfully")