#!/usr/bin/env python3

import sys
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

print("Testing elasticsearch...")

try:
    load_dotenv()
    es_url = os.getenv('ELASTICSEARCH_URL')
    
    if not es_url:
        print("ELASTICSEARCH_URL not set, skipping connection test")
        print("elasticsearch test completed successfully (skipped)")
        sys.exit(0)
    
    es = Elasticsearch([es_url])

    # Check if the cluster is up
    if es.ping():
        print("Connected to Elasticsearch")
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
    else:
        print("Could not connect to Elasticsearch")
        sys.exit(1)

    print("elasticsearch test completed successfully")
    sys.exit(0)
except Exception as e:
    print(f"elasticsearch test failed. Error: {str(e)}")
    sys.exit(1)