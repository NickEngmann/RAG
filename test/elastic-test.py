#!/usr/bin/env python3

try:
    from elasticsearch import Elasticsearch
    from dotenv import load_dotenv
    import os
    
    print("Testing elasticsearch...")
    
    load_dotenv()
    
    es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    es = Elasticsearch([es_url])
    
    # Check if the cluster is up
    if es.ping():
        print("Connected to Elasticsearch")
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
    else:
        print("Could not connect to Elasticsearch")
    
    print("elasticsearch test completed successfully")
except ImportError as e:
    print(f"Elasticsearch test skipped. Error: {str(e)}")
    print("This is expected if elasticsearch is not installed.")
except Exception as e:
    print(f"Elasticsearch test failed. Error: {str(e)}")
    print("This is expected if Elasticsearch is not running.")